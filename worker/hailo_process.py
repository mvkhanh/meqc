from hailo_platform import VDevice, HailoSchedulingAlgorithm, HEF
from multiprocessing import Process, Queue, Event
from utils import _softmax
import numpy as np
import cv2
import math
from functools import partial

# ---------- Helpers ----------
def list_input_names(im):
    """Trả về list tên input. Có fallback cho API khác nhau."""
    return [x.name for x in im.inputs]   # list of objects

def list_output_names(im):
    """Trả về list tên output. Có fallback cho API khác nhau."""
    return [x.name for x in im.outputs]

def io_shape(im, is_input, name):
    """Lấy shape cho input/output theo name (hoặc None nếu single)."""
    if is_input:
        return (im.input(name).shape if name is not None else im.input().shape)
    else:
        return (im.output(name).shape if name is not None else im.output().shape)

def bindings_set_buffer(bindings, is_input, name, buf):
    """Gán buffer cho bindings theo tên (hoặc None nếu single)."""
    if is_input:
        (bindings.input(name) if name is not None else bindings.input()).set_buffer(buf)
    else:
        (bindings.output(name) if name is not None else bindings.output()).set_buffer(buf)

def bindings_get_buffer(bindings, name):
    """Lấy buffer output theo tên (hoặc None nếu single)."""
    return (bindings.output(name) if name is not None else bindings.output()).get_buffer()

class HailoInferProc(Process):
    """
    A dedicated *process* that owns a Hailo VDevice and runs exactly one HEF model.
    Use multi_process_service + shared group_id so multiple processes share the same Hailo8.
    It receives RGB face crops on in_q and returns parsed results on out_q.
    model_type: 'agegender' | 'emotion' | 'gaze'
    """

    def __init__(self, hef_path: str, model_type: str, in_q: Queue, out_q: Queue,
                 group_id: str = "SHARED", timeout_ms: int = 10000):
        super().__init__(daemon=True)
        self.hef_path = hef_path
        self.model_type = model_type
        self.in_q = in_q
        self.out_q = out_q
        self.group_id = group_id
        self.timeout_ms = timeout_ms
        self._stop = Event()

    # ---------- helpers ----------
    @staticmethod
    def decode_gaze3Gaze_bins(pitch_logits, yaw_logits):
        GAZE_BINS = 90
        GAZE_BINWIDTH = 4.0
        GAZE_ANGLE = 180.0
        idx = np.arange(GAZE_BINS, dtype=np.float32)[None, :]
        p_pitch = _softmax(pitch_logits)
        p_yaw = _softmax(yaw_logits)
        pitch_deg = float(np.sum(p_pitch * idx, axis=1)[0] * GAZE_BINWIDTH - GAZE_ANGLE)
        yaw_deg = float(np.sum(p_yaw * idx, axis=1)[0] * GAZE_BINWIDTH - GAZE_ANGLE)
        return yaw_deg, pitch_deg
    
    @staticmethod
    def eye_contact_angle(gaze_vec):
        f = np.array([0, 0, -1], dtype=np.float32); v = gaze_vec / (np.linalg.norm(gaze_vec) + 1e-6)
        cos_ = float(np.clip(np.dot(v, f), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_)))
    
    def _load_quant_info(self):
        """Đọc quant info (scale, zp) cho từng vstream từ HEF."""
        in_qp, out_qp = {}, {}
        try:
            hef = HEF(self.hef_path)
            for info in hef.get_input_vstream_infos():
                qi = info.quant_info
                in_qp[info.name] = (float(qi.qp_scale), float(qi.qp_zp))
            for info in hef.get_output_vstream_infos():
                qi = info.quant_info
                out_qp[info.name] = (float(qi.qp_scale), float(qi.qp_zp))
        except Exception as e:
            print(f"[warn] quant_info not available: {e}")
        return in_qp, out_qp

    def _dequant_output(self, name: str, arr: np.ndarray) -> np.ndarray:
        """(arr_u8/i8 -> float32) theo (scale, zp) của output name. 
        Nếu thiếu info: mặc định scale=1, zp=0."""
        scale, zp = self.out_qp.get(name, (1.0, 0.0))
        return (arr.astype(np.float32) - zp) * scale
    
    # Xem shape input tu opencv, chuyen lai cho dung NHWC, va resize dung shape dau vao
    def _prep(self, face_rgb: np.ndarray, expected_shape) -> np.ndarray:
        """
        Trả về buffer đúng y 'expected_shape' của infer_model.input(name).shape
        và đúng dtype (uint8/float32...). Không tự động thêm batch khi không có.
        """
        es = tuple(int(x) for x in expected_shape)

        H, W, C = es
        img = cv2.resize(face_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        return img.astype(np.uint8)

    def _postprocess(self, out_arrs):
        """Best-effort postprocess for demo; adjust to your HEF's real outputs.
        out_arrs: list[np.ndarray] (one or more outputs).
        Returns a dict depending on model_type.
        """
        
        if self.model_type == 'emotion':
            # Assume logits vector for 7 emotions
            out_arrs = np.asarray(out_arrs)
            EMO_LABELS = ['Angry', 'Fear', 'Happiness', 'Sad', 'Surprise', 'Neutral']
            vec = out_arrs[0].reshape(-1)
            if vec.size == 0:
                return {"emotion": None, "emotion_conf": None}
            prob = _softmax(vec)
            idx = int(prob.argmax())
            conf = float(prob[idx])
            label = EMO_LABELS[idx] if idx < len(EMO_LABELS) else f"cls_{idx}"
            return {"emotion": label, "emotion_conf": conf}
        
        elif self.model_type == 'gaze':
            out_arrs = np.asarray(out_arrs)
            EYE_CONTACT_THRESH_DEG = 30.0
            pitch_logits = out_arrs[0]
            yaw_logits = out_arrs[1]
            yaw_deg, pitch_deg = HailoInferProc.decode_gaze3Gaze_bins(pitch_logits, yaw_logits)
            yaw_rad, pitch_rad = math.radians(yaw_deg), math.radians(pitch_deg)
            gx = math.sin(yaw_rad) * math.cos(pitch_rad); gy = math.sin(pitch_rad); gz = -math.cos(yaw_rad) * math.cos(pitch_rad)
            gaze_vec = np.array([gx, gy, gz], dtype=np.float32); gaze_vec /= (np.linalg.norm(gaze_vec) + 1e-6)
            theta = HailoInferProc.eye_contact_angle(gaze_vec)
            eye_contact = theta <= EYE_CONTACT_THRESH_DEG
            return {
                "eye_contact": eye_contact
            }
        
        else:  # agegender
            AGE_CLASS_OFFSET = 3
            g = np.asarray(out_arrs[2]).squeeze()
            # age_raw = float(np.asarray(out_arrs[1]).squeeze())
            # prob_female = 1.0 / (1.0 + np.exp(-g))
            
            probs = _softmax(g)
            prob_female = float(probs[1])
            if prob_female >= 0.5:
                gender_label = "Female"
                gender_conf = prob_female
            else:
                gender_label = "Male"
                gender_conf = 1.0 - prob_female
            age_class_logits = np.asarray(out_arrs[0])
            age_reg_output = np.asarray(out_arrs[1])
            pred_age_reg = age_reg_output.item()
            pred_remapped_class = np.argmax(age_class_logits)
            pred_original_class = pred_remapped_class + AGE_CLASS_OFFSET
            age_years = (pred_original_class * 5) + pred_age_reg

            # age_years = float(np.clip(age_raw, 0, 100))

            return {"age": age_years, "gender": gender_label, "gender_conf": gender_conf}

    # Dummy callback required by run_async
    @staticmethod
    def _cb(completion_info, bindings):
        # No-op; sync read via bindings.output().get_buffer()
        return

    def run(self):
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = self.group_id
        params.multi_process_service = True

        self.in_qp, self.out_qp = self._load_quant_info()
        if self.out_qp:
            print("[quant] outputs:", {k: self.out_qp[k] for k in self.out_qp})

        with VDevice(params) as vdevice:
            infer_model = vdevice.create_infer_model(self.hef_path)
            infer_model.set_batch_size(1)
            
            input_names = list_input_names(infer_model)
            output_names = list_output_names(infer_model)

            with infer_model.configure() as cmodel:
                while True:
                    # t1 = time()
                    face = self.in_q.get()
                    if face is None:
                        break

                    bindings = cmodel.create_bindings()

                    # Inputs
                    for n in input_names:
                        in_shape = io_shape(infer_model, True, n)
                        in_buf   = self._prep(face, in_shape)
                        bindings_set_buffer(bindings, True, n, in_buf)

                    # Outputs
                    for n in output_names:
                        out_shape = io_shape(infer_model, False, n)
                        out_buf   = np.empty(tuple(int(x) for x in out_shape), dtype=np.uint8)
                        bindings_set_buffer(bindings, False, n, out_buf)

                    try:
                        cmodel.wait_for_async_ready(timeout_ms=self.timeout_ms)
                    except Exception as e:
                        print(f'{self.model_type} cmodel error: {e}')
                    
                    job = cmodel.run_async([bindings], partial(self._cb, bindings=bindings))
                    try:
                        job.wait(self.timeout_ms)
                    except Exception as e:
                        print(f'{self.model_type} job error: {e}')

                    out_raw = [bindings_get_buffer(bindings, n) for n in output_names]
                    out_deq = [self._dequant_output(n, arr) for n, arr in zip(output_names, out_raw)]

                    # Cast sang float32 trong postprocess nếu cần:
                    result = self._postprocess(out_deq)
                    self.out_q.put(result)
                    # print(f'{self.model_type} inference time: {time() - t1}s')
