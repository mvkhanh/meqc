import os
import json
from multiprocessing import Process, Queue, Event
from typing import Optional
import numpy as np
import cv2
import math
from functools import partial
from utils import submit, poll, _softmax
from time import time
from threading import Lock, Thread
from model.face_detector import YuNetFaceDetector
from hailo_platform import VDevice, HailoSchedulingAlgorithm, HEF
# -------------------- Face Recognition (ONNX + FAISS) --------------------
import onnxruntime as ort
from torchvision import transforms
from face_alignment import align
import faiss
import queue as tqueue
from PIL import Image


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

# --- Helper: Save RGB/grayscale face image as WebP with quality ---
def _save_webp_image(rgb: np.ndarray, out_path: str, quality: int = 80):
    """
    Lưu ảnh (RGB hoặc grayscale) thành WebP tại out_path với chất lượng 'quality'.
    Tự tạo thư mục nếu chưa có.
    """
    # Bảo đảm thư mục tồn tại
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Chuyển sang BGR trước khi encode bằng OpenCV
    if rgb.ndim == 2:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".webp", bgr, [cv2.IMWRITE_WEBP_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("encode webp failed")
    with open(out_path, "wb") as f:
        f.write(buf.tobytes())

# -------------------- Face Recognition (ONNX + FAISS) --------------------
class _FaissDB:
    """Minimal FAISS-backed DB with cosine similarity (via inner product on L2-normalized vectors).
    Falls back to NumPy if faiss is unavailable.
    """
    FAISS_NAME = 'faiss.index'
    METADATA_NAME = 'metadata.json'
    @classmethod
    def load(cls, path: str):
        """Load FAISS index and metadata saved by `save`.
        """
        idx = faiss.read_index(path + _FaissDB.FAISS_NAME)
        with open(path + _FaissDB.METADATA_NAME, "r") as f:
            meta = json.load(f)
        dim = int(meta.get("dim", idx.d))
        obj = cls(dim)
        obj.index = idx
        obj.ids = list(meta.get("ids", []))
        obj.next_id = int(meta.get("next_id", len(obj.ids) + 1))
        return obj

    def save(self, path: str):
        """Persist FAISS index and metadata.
        Writes two files: `{path}.index` (FAISS) and `{path}.json` (ids/next_id/meta).
        """
        faiss.write_index(self.index, path + _FaissDB.FAISS_NAME)
        meta = {
            "ids": self.ids,
            "next_id": self.next_id,
            "dim": self.dim,
            "ntotal": int(self.index.ntotal),
        }
        with open(path + _FaissDB.METADATA_NAME, "w") as f:
            json.dump(meta, f)
            
    def __init__(self, dim: int):
        self.dim = dim
        self.ids = []  # index -> person_id
        self.next_id = 1
        self._use_faiss = faiss is not None
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vec: np.ndarray) -> int:
        assert vec.shape == (1, self.dim)
        pid = self.next_id
        self.next_id += 1
        self.index.add(vec)
        self.ids.append(pid)
        return pid

    def search(self, vec: np.ndarray, topk: int = 1):
        if self.index.ntotal == 0:
            return None
        D, I = self.index.search(vec, topk)
        sim = float(D[0][0])
        idx = int(I[0][0])
        if idx < 0:
            return None
        return self.ids[idx], sim

class FaceRecognizer:
    """ONNX embedding + FAISS DB wrapper."""
    def __init__(self, onnx_path: str, sim_thres: float = 0.45,
                 providers=("CPUExecutionProvider",), db_path: Optional[str] = None, autosave: bool = True):
        self.sess = ort.InferenceSession(onnx_path, providers=list(providers))
        self.in_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.sim_thres = float(sim_thres)
        self.db_path = db_path
        self.autosave = bool(autosave)
        # preprocessing identical to PyTorch code
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # Try to load existing DB if provided
        self.db = None  # type: Optional[_FaissDB]
        if self.db_path and os.path.exists(self.db_path):
            try:
                self.db = _FaissDB.load(self.db_path)
                print(f"[recog] loaded FAISS DB: {self.db_path} (ntotal={self.db.index.ntotal})")
            except Exception as e:
                print(f"[recog] failed to load DB '{self.db_path}': {e}")
                self.db = None

    def _preprocess(self, face_rgb: np.ndarray) -> np.ndarray:
        # align.get_aligned_face may accept PIL or ndarray
        img = Image.fromarray(face_rgb) if not isinstance(face_rgb, Image.Image) else face_rgb
        try:
            aligned = align.get_aligned_face(img)
        except Exception as e:
            print(f'Align error: {e}')
            aligned = img

        x = self.transform(aligned).unsqueeze(0).numpy().astype(np.float32)
        return np.ascontiguousarray(x)

    def embed(self, face_rgb: np.ndarray) -> np.ndarray:
        inp = self._preprocess(face_rgb)
        outs = self.sess.run(self.out_names, {self.in_name: inp})
        emb = outs[0].astype(np.float32)
        # L2 normalize
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        if self.db is None:
            self.db = _FaissDB(dim=emb.shape[1])
        return emb

    def identify_or_enroll(self, face_rgb: np.ndarray):
        emb = self.embed(face_rgb)
        hit = self.db.search(emb, topk=1)
        if hit is not None:
            pid, sim = hit
            if sim >= self.sim_thres:
                return pid, sim, False  # recognized
        # enroll new ID
        pid = self.db.add(emb)
        # Persist DB on each new enrollment
        try:
            if self.db_path and self.autosave:
                self.db.save(self.db_path)
        except Exception as e:
            print(f"[recog] warning: failed to save DB: {e}")
        print(f'New person: {pid}')
        return pid, 1.0, True

class RecogThread(Thread):
    """Thread that consumes face crops and updates shared state with person_id."""
    def __init__(self, in_q: tqueue.Queue, shared: dict, stop_event,
                 onnx_path: str, sim_thres: float = 0.45, db_path: Optional[str] = None, autosave: bool = True,
                 images_dir: Optional[str] = None):
        super().__init__(daemon=True)
        self.in_q = in_q
        self.shared = shared
        self.stop_event = stop_event
        self.recog = FaceRecognizer(onnx_path, sim_thres, db_path=db_path, autosave=autosave)
        self.images_dir = images_dir or os.path.join("db", "images")
        os.makedirs(self.images_dir, exist_ok=True)

    def run(self):
        while not self.stop_event.is_set():
            try:
                item = self.in_q.get(timeout=0.1)
            except tqueue.Empty:
                continue
            if item is None:
                break
            try:
                pid, sim, is_new = self.recog.identify_or_enroll(item)
                # Lưu một ảnh WebP/ID nếu chưa tồn tại
                try:
                    out_path = os.path.join(self.images_dir, f"{int(pid)}.webp")
                    if not os.path.exists(out_path):
                        # Cố gắng dùng ảnh đã align; nếu lỗi thì dùng ảnh gốc
                        try:
                            pil_img = Image.fromarray(item) if not isinstance(item, Image.Image) else item
                            aligned = align.get_aligned_face(pil_img)
                            arr = np.array(aligned)
                        except Exception:
                            arr = item  # dùng ảnh crop RGB gốc
                        # Bảo đảm là RGB hoặc gray trước khi lưu
                        if arr.ndim == 3 and arr.shape[2] == 3:
                            arr_rgb = arr
                        elif arr.ndim == 2:
                            arr_rgb = arr
                        else:
                            # Trường hợp lạ: cố gắng chuyển về RGB
                            arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                        _save_webp_image(arr_rgb, out_path, quality=80)
                except Exception as e:
                    print(f"[recog] warn: failed to save face image for ID#{pid}: {e}")
                with self.shared["lock"]:
                    self.shared["result"]["person_id"] = int(pid)
                    self.shared["result"]["ts"] = time()
            except Exception as e:
                # best-effort; keep thread alive
                print(f"[recog] error: {e}")

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
            pred_remapped_class = np.argmax(age_class_logits, axis=1)[0]
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

class DetectWorker(Process):
    """
    Multiprocessing worker:
      - Nhận frame qua in_q
      - YuNet chỉ phát hiện khuôn mặt lớn nhất -> lưu crop vào biến dùng chung
      - Hai *process* Hailo (Age/Gender và Emotion) chạy song song, đọc crop chung để suy luận
      - Kết quả (age, gender, emotion) được lưu vào dict chung
      - Vẽ kết quả lên frame và gửi ra out_q
    """

    def __init__(self, detector_path, agegender_path, detect_every_n: int, face_score_thres=0.8, width=640, height=640,
                                  in_q: Optional[Queue] = None, out_q: Optional[Queue] = None, emotion_path: str = None, gaze_path: str = None,
                 recog_onnx_path: Optional[str] = None, recog_sim_thres: float = 0.45,
                 recog_db_path: Optional[str] = None):
        super().__init__(daemon=False)
        self.detect_every_n = detect_every_n
        self.detector = None
        self.detector_path = detector_path

        self.agegender_path = agegender_path 
        self.emotion_path = emotion_path     
        self.gaze_path = gaze_path

        self.face_score_thres = face_score_thres
        self.width = width
        self.height = height
        self.in_q = in_q
        self.out_q = out_q
        self.frame_idx = 0
        self._stop = Event()

        # Shared state insiqde this process
        self.shared = None

        # IPC queues to Hailo processes
        self._ag_in = None
        self._ag_out = None
        self._emo_in = None
        self._emo_out = None
        self._gaze_in = None
        self._gaze_out = None
        
        # Hailo subprocess handles
        self._ag_proc = None
        self._emo_proc = None
        self._gaze_proc = None
        
        self.recog_onnx_path = recog_onnx_path
        self.recog_sim_thres = float(recog_sim_thres)
        self.recog_db_path = recog_db_path
        self._recog_q = None  # thread queue
        self._recog_thr = None
        self._last_recog_ver = -1


    # -------------------- Helper --------------------
    @staticmethod
    def _largest_box(boxes):
        # boxes: list[(x,y,w,h)]
        if not boxes:
            return None
        return max(boxes, key=lambda b: int(b[2]) * int(b[3]))

    @staticmethod
    def _clamp_box(box, W, H):
        x, y, w, h = map(int, box)
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(W, x + w); y1 = min(H, y + h)
        w2 = max(0, x1 - x0); h2 = max(0, y1 - y0)
        return x0, y0, w2, h2

    def _draw_overlay(self, frame_bgr: np.ndarray):
        H, W = frame_bgr.shape[:2]
        with self.shared["lock"]:
            box = self.shared["last_box"]
            res = dict(self.shared["result"])  # shallow copy
        # Vẽ bbox
        if box is not None:
            x, y, w, h = self._clamp_box(box, W, H)
            if w > 0 and h > 0:
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Chuẩn bị text
        parts = []
        pid = res.get("person_id")
        if pid is not None:
            parts.append(f"ID#{pid}")
            
        if res.get("gender") is not None:
            if res.get("gender_conf") is not None:
                parts.append(f"{res['gender']} {int(round(res['gender_conf']*100))}%")
            else:
                parts.append(f"{res['gender']}")
                
        if res.get("age") is not None:
            parts.append(f"{int(round(res['age']))}y")
            
        if res.get("emotion") is not None:
            if res.get("emotion_conf") is not None:
                parts.append(f"{res['emotion']} {int(round(res['emotion_conf']*100))}%")
            else:
                parts.append(f"{res['emotion']}")
                
        # Eye contact info
        dwell = res.get("eye_contact_dwell")
        ec = res.get("eye_contact")
        if dwell is not None and dwell > 0:
            tag = "Eye+" if (ec and dwell >= 3.0) else "Eye"
            parts.append(f"{tag} {dwell:.1f}s")
        text = " | ".join(parts)

        if text and box is not None:
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            pad = 4
            x, y, w, h = self._clamp_box(box, W, H)
            x_bg0 = x
            y_bg0 = max(0, y - th - baseline - 2 * pad)
            x_bg1 = min(W - 1, x + tw + 2 * pad)
            y_bg1 = y
            cv2.rectangle(frame_bgr, (x_bg0, y_bg0), (x_bg1, y_bg1), (0, 0, 0), thickness=-1)
            cv2.putText(frame_bgr, text, (x + pad, y - baseline - pad),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        return frame_bgr

    # -------------------- Main process loop --------------------
    def run(self):
        # Shared state inside this (DetectWorker) process
        self.shared = {
            "lock": Lock(),
            "latest_face": None,   # numpy RGB crop
            "face_ver": 0,
            "last_box": None,      # (x,y,w,h)
            "result": {
                "age": None,
                "gender": None,
                "gender_conf": None,
                "emotion": None,
                "emotion_conf": None,
                "ts": 0.0,
                "eye_contact": None,
                "eye_contact_dwell": 0.0,
                "person_id": None,
            },
            "eye_on_since": 0.0,
        }

        self.detector = YuNetFaceDetector(model_path=self.detector_path, face_score_thres=self.face_score_thres,
                                          width=self.width, height=self.height)

        # Spawn Hailo subprocesses (one per model) with a shared group id so they share the same Hailo8
        GROUP_ID = "SHARED"  # can be parameterized
        if os.path.exists(self.agegender_path):
            self._ag_in, self._ag_out = Queue(maxsize=1), Queue(maxsize=1)
            self._ag_proc = HailoInferProc(self.agegender_path, 'agegender', self._ag_in, self._ag_out, GROUP_ID)
            self._ag_proc.start()
        else:    
            raise FileNotFoundError(f'Not found {self.agegender_path}')
        
        if os.path.exists(self.emotion_path):
            self._emo_in, self._emo_out = Queue(maxsize=1), Queue(maxsize=1)
            self._emo_proc = HailoInferProc(self.emotion_path, 'emotion', self._emo_in, self._emo_out, GROUP_ID)
            self._emo_proc.start()
        else:
            raise FileNotFoundError(f'Not found {self.emotion_path}')
        
        if os.path.exists(self.gaze_path):
            self._gaze_in, self._gaze_out = Queue(maxsize=1), Queue(maxsize=1)
            self._gaze_proc = HailoInferProc(self.gaze_path, 'gaze', self._gaze_in, self._gaze_out, GROUP_ID)
            self._gaze_proc.start()
        else:
            raise FileNotFoundError(f'Not found {self.gaze_path}')
        
        # Start face recognition thread (ONNX + FAISS)
        if self.recog_onnx_path and os.path.exists(self.recog_onnx_path):
            self._recog_q = tqueue.Queue(maxsize=1)
            self._recog_thr = RecogThread(self._recog_q, self.shared, self._stop,
                                          self.recog_onnx_path, self.recog_sim_thres,
                                          db_path=self.recog_db_path, autosave=True,
                                          images_dir=os.path.join("db", "images"))
            self._recog_thr.start()
            print("[recog] Face recognition thread started")
            if self.recog_db_path:
                print(f"[recog] DB path: {self.recog_db_path}")
        else:
            print(f"[recog] disabled (onnx not found: {self.recog_onnx_path})")


        # Track last face version submitted to each model
        last_ag_ver = -1
        last_emo_ver = -1
        last_gaze_ver = -1
        last_recog_ver = -1

        # Main loop: receive frame, (periodically) detect -> update crop + draw overlay
        while not self._stop.is_set():
            frame_bgr = poll(self.in_q)
            if frame_bgr is None:
                continue

            H, W = frame_bgr.shape[:2]
            need_detect = (self.frame_idx % self.detect_every_n == 0)
            t0 = time()
            if need_detect:
                boxes = self.detector.detect(frame_bgr) or []
                best = self._largest_box(boxes)
                if best is not None:
                    x, y, w, h = self._clamp_box(best, W, H)
                    if w >= 16 and h >= 16:
                        face_crop = frame_bgr[y:y + h, x:x + w]
                        with self.shared["lock"]:
                            # Store as RGB for recognizers
                            self.shared["latest_face"] = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                            self.shared["face_ver"] += 1
                            self.shared["last_box"] = (x, y, w, h)
                else:
                    with self.shared["lock"]:
                        self.shared["last_box"] = None

            # Push latest face to subprocesses (only when face version changes) and pull results
            with self.shared["lock"]:
                ver = self.shared["face_ver"]
                face_rgb = self.shared["latest_face"].copy() if (self.shared["latest_face"] is not None) else None

            if face_rgb is not None:
                if self._ag_in is not None and ver != last_ag_ver:
                    submit(self._ag_in, face_rgb)
                    last_ag_ver = ver
                if self._emo_in is not None and ver != last_emo_ver:
                    submit(self._emo_in, face_rgb)
                    last_emo_ver = ver
                if self._gaze_in is not None and ver != last_gaze_ver:
                    submit(self._gaze_in, face_rgb)
                    last_gaze_ver = ver

            # Try to read outputs without blocking
            from queue import Empty
            try:
                if self._ag_out is not None:
                    res = self._ag_out.get_nowait()
                    with self.shared["lock"]:
                        dst = self.shared["result"]
                        dst["age"] = res.get("age")
                        dst["gender"] = res.get("gender")
                        dst["gender_conf"] = res.get("gender_conf")
                        dst["ts"] = time()
            except Empty:
                pass

            try:
                if self._emo_out is not None:
                    res = self._emo_out.get_nowait()
                    with self.shared["lock"]:
                        dst = self.shared["result"]
                        dst["emotion"] = res.get("emotion")
                        dst["emotion_conf"] = res.get("emotion_conf")
                        dst["ts"] = time()
            except Empty:
                pass

            try:
                if self._gaze_out is not None:
                    res = self._gaze_out.get_nowait()
                    now = time()
                    if isinstance(res, dict):
                        with self.shared["lock"]:
                            ec = bool(res.get("eye_contact", False))
                            if ec:
                                if not self.shared.get("eye_on_since"):
                                    self.shared["eye_on_since"] = now
                                dwell = now - (self.shared["eye_on_since"] or now)
                            else:
                                self.shared["eye_on_since"] = 0.0
                                dwell = 0.0
                            self.shared["result"]["eye_contact"] = ec
                            self.shared["result"]["eye_contact_dwell"] = float(max(0.0, dwell))
                            self.shared["result"]["ts"] = now
                            # If eye contact detected, enqueue latest face for recognition (dedupe by face_ver)
                            if self._recog_q is not None and ec:
                                face_rgb2 = self.shared["latest_face"]
                                ver2 = self.shared["face_ver"]
                                if face_rgb2 is not None and ver2 != last_recog_ver:
                                    try:
                                        self._recog_q.put_nowait(face_rgb2.copy())
                                        last_recog_ver = ver2
                                    except tqueue.Full:
                                        pass
                            else:
                                self.shared["result"]["person_id"] = None
            except Empty:
                pass

            # Always draw overlay from shared results
            out = self._draw_overlay(frame_bgr)
            t1 = time()
            # if t1 > t0:
            #     print(f"FPS: {1.0 / (t1 - t0)} - Inference time: {t1 - t0}s")
            submit(self.out_q, out)
            self.frame_idx += 1

        # Tell subprocesses to stop
        try:
            if self._ag_in is not None:
                self._ag_in.put_nowait(None)
        except Exception:
            pass
        try:
            if self._emo_in is not None:
                self._emo_in.put_nowait(None)
        except Exception:
            pass
        try:
            if self._gaze_in is not None:
                self._gaze_in.put_nowait(None)
        except Exception:
            pass
        try:
            if self._ag_proc is not None:
                self._ag_proc.join(timeout=0.5)
        except Exception:
            pass
        try:
            if self._emo_proc is not None:
                self._emo_proc.join(timeout=0.5)
        except Exception:
            pass
        try:
            if self._gaze_proc is not None:
                self._gaze_proc.join(timeout=0.5)
        except Exception:
            pass
                # Stop recognition thread
        try:
            if self._recog_q is not None:
                self._recog_q.put_nowait(None)
        except Exception:
            pass
        try:
            if self._recog_thr is not None:
                self._recog_thr.join(timeout=0.5)
        except Exception:
            if self._recog_thr.is_alive():
                self._recog_thr.stop()
            pass


    def stop(self):
        self._stop.set()
        try:
            self.in_q.put_nowait(None)
        except Exception:
            pass