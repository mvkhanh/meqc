import os
from multiprocessing import Process, Queue, Event
from typing import Optional
import numpy as np
import cv2
from functools import partial
from utils import submit, poll
from time import time, sleep
from threading import Thread, Lock
from model.face_detector import YuNetFaceDetector
from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType

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


def io_set_format(im, is_input, name, fmt):
    """Set format type cho input/output (từng tên)."""
    if is_input:
        (im.input(name) if name is not None else im.input()).set_format_type(fmt)
    else:
        (im.output(name) if name is not None else im.output()).set_format_type(fmt)


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
    model_type: 'agegender' | 'emotion'
    """
    INPUT_SIZE_MAP = {
        'agegender': (150, 150),
        'emotion': (224, 224),
        'gaze': (448, 448)
    }
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
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x = x - np.max(x)
        e = np.exp(x)
        s = e / (np.sum(e) + 1e-9)
        return s
    
    # Xem shape input tu opencv, chuyen lai cho dung NHWC, va resize dung shape dau vao
    def _prep(self, face_rgb: np.ndarray, expected_shape) -> np.ndarray:
        """Chuẩn hoá input theo expected_shape (NCHW hoặc NHWC). Trả float32."""
        print(f'{self.model_type} - {face_rgb.shape} - {expected_shape}')
        H, W = expected_shape[0], expected_shape[1]
        img = cv2.resize(face_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        img = np.expand_dims(img, 0)  # (1,H,W,3)
        return img.astype(np.float32)

    def _postprocess(self, out_arrs):
        """Best-effort postprocess for demo; adjust to your HEF's real outputs.
        out_arrs: list[np.ndarray] (one or more outputs).
        Returns a dict depending on model_type.
        """
        if self.model_type == 'emotion':
            # Assume logits vector for 7 emotions
            EMO_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            vec = out_arrs[0].reshape(-1)
            if vec.size == 0:
                return {"emotion": None, "emotion_conf": None}
            prob = self._softmax(vec)
            idx = int(prob.argmax())
            conf = float(prob[idx])
            label = EMO_LABELS[idx] if idx < len(EMO_LABELS) else f"cls_{idx}"
            return {"emotion": label, "emotion_conf": conf}
        else:  # agegender
            g = float(np.asarray(out_arrs[1]).squeeze())
            age_raw = float(np.asarray(out_arrs[0]).squeeze())
            prob_female = 1.0 / (1.0 + np.exp(-g))
            if prob_female >= 0.5:
                gender_label = "Female"
                gender_conf = prob_female
            else:
                gender_label = "Male"
                gender_conf = 1.0 - prob_female
            age_years = float(np.clip(age_raw, 0, 100))

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

        with VDevice(params) as vdevice:
            infer_model = vdevice.create_infer_model(self.hef_path)
            infer_model.set_batch_size(1)
            
            input_names = list_input_names(infer_model)
            output_names = list_output_names(infer_model)

            with infer_model.configure() as cmodel:
                while True:
                    face = self.in_q.get()
                    if face is None:
                        break

                    # Bindings
                    bindings = cmodel.create_bindings()
                    
                    # --- Inputs: tạo buffer đúng shape cho từng input name
                    for n in input_names:
                        in_shape = io_shape(infer_model, True, n)
                        in_buf = self._prep(face, in_shape)
                        bindings_set_buffer(bindings, True, n, in_buf)

                    # --- Outputs: cấp buffer cho tất cả outputs
                    for n in output_names:
                        out_shape = io_shape(infer_model, False, n)
                        out_buf = np.empty(out_shape, dtype=np.float32)
                        bindings_set_buffer(bindings, False, n, out_buf)

                    cmodel.wait_for_async_ready(timeout_ms=self.timeout_ms)
                    job = cmodel.run_async(
                        [bindings],
                        partial(HailoInferProc._cb, bindings=bindings)
                    )
                    job.wait(self.timeout_ms)

                    # Read outputs (single output path)
                    out_arrs = [bindings.output().get_buffer()]

                    # Parse & push
                    result = self._postprocess(out_arrs)
                    self.out_q.put(result)


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
                 in_q: Optional[Queue] = None, out_q: Optional[Queue] = None, emotion_path: str = None):
        super().__init__(daemon=False)
        self.detect_every_n = detect_every_n
        self.detector = None
        self.detector_path = detector_path
        # self.agegender / self.emotion (ONNX) are removed; replaced by Hailo processes
        self.agegender_path = agegender_path  # expecting .hef
        self.emotion_path = emotion_path      # expecting .hef

        self.face_score_thres = face_score_thres
        self.width = width
        self.height = height
        self.in_q = in_q
        self.out_q = out_q
        self.frame_idx = 0
        self._stop = Event()

        # Shared state inside this process
        self.shared = None

        # IPC queues to Hailo processes
        self._ag_in = None
        self._ag_out = None
        self._emo_in = None
        self._emo_out = None

        # Hailo subprocess handles
        self._ag_proc = None
        self._emo_proc = None

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

    def _agegender_loop(self):
        if self._ag_in is None or self._ag_out is None:
            return
        last_ver = -1
        from queue import Empty
        while not self._stop.is_set():
            with self.shared["lock"]:
                ver = self.shared["face_ver"]
                face = self.shared["latest_face"].copy() if (self.shared["latest_face"] is not None and ver != last_ver) else None
            if face is not None:
                submit(self._ag_in, face)
                last_ver = ver
            # Non-blocking fetch result
            try:
                res = self._ag_out.get_nowait()
                with self.shared["lock"]:
                    dst = self.shared["result"]
                    dst["age"] = res.get("age")
                    dst["gender"] = res.get("gender")
                    dst["gender_conf"] = res.get("gender_conf")
                    dst["ts"] = time()
            except Empty:
                pass
            sleep(0.003)

    def _emotion_loop(self):
        if self._emo_in is None or self._emo_out is None:
            return
        last_ver = -1
        from queue import Empty
        while not self._stop.is_set():
            with self.shared["lock"]:
                ver = self.shared["face_ver"]
                face = self.shared["latest_face"].copy() if (self.shared["latest_face"] is not None and ver != last_ver) else None
            if face is not None:
                submit(self._emo_in, face)
                last_ver = ver
            # Non-blocking fetch result
            try:
                res = self._emo_out.get_nowait()
                with self.shared["lock"]:
                    dst = self.shared["result"]
                    dst["emotion"] = res.get("emotion")
                    dst["emotion_conf"] = res.get("emotion_conf")
                    dst["ts"] = time()
            except Empty:
                pass
            sleep(0.003)

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
                "ts": 0.0
            }
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

        # Background threads that feed face crops to the subprocesses and pull results back
        ag_thread = Thread(target=self._agegender_loop, daemon=True)
        if self._ag_proc is not None:
            ag_thread.start()
        em_thread = Thread(target=self._emotion_loop, daemon=True)
        if self._emo_proc is not None:
            em_thread.start()

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

            # Always draw overlay from shared results
            out = self._draw_overlay(frame_bgr)
            t1 = time()
            if t1 > t0:
                print(f"FPS: {1.0 / (t1 - t0)} - Inference time: {t1 - t0}s")
            submit(self.out_q, out)
            self.frame_idx += 1

        # Cleanup
        try:
            if ag_thread.is_alive():
                ag_thread.join(timeout=0.1)
        except Exception:
            pass
        try:
            if em_thread.is_alive():
                em_thread.join(timeout=0.1)
        except Exception:
            pass

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
            if self._ag_proc is not None:
                self._ag_proc.join(timeout=0.5)
        except Exception:
            pass
        try:
            if self._emo_proc is not None:
                self._emo_proc.join(timeout=0.5)
        except Exception:
            pass

    def stop(self):
        self._stop.set()
        try:
            self.in_q.put_nowait(None)
        except Exception:
            pass