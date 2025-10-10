import os
from multiprocessing import Process, Queue, Event
from typing import Optional
import numpy as np
import cv2

from utils import submit, poll
from time import time
from threading import Lock
from model.face_detector import YuNetFaceDetector
from model.gesture_detector import GestureDetector
from worker.hailo_process import HailoInferProc
from worker.gesture_worker import GestureProcess
from worker.recog_worker import RecogThread
from worker.send_worker import EventSenderThread

import queue as tqueue
import base64

# --- Helper: Encode RGB/grayscale image to WebP base64 ---
def _encode_webp_base64(rgb: np.ndarray, size=(112, 112), quality: int = 80) -> str:
    """
    Mã hoá ảnh RGB/grayscale thành WebP (resize về 112x112 mặc định) và trả về base64 string.
    """
    if rgb is None:
        return None
    # Đảm bảo có 3 kênh BGR cho OpenCV
    if rgb.ndim == 2:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # Resize về kích thước yêu cầu
    bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".webp", bgr, [cv2.IMWRITE_WEBP_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("encode webp failed")
    return base64.b64encode(buf).decode("ascii")

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
                 recog_db_path: Optional[str] = None, server_ip=None, server_port=None, device_id=None,
                 gesture_path=None):
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

        self.gesture_path = gesture_path
        self.gesture_proc = None
        self.gesture_q = None
        
        # ---- Server config / sender queue ----
        self.server_url = f"http://{server_ip}:{server_port}/api/"
        self.device_id = device_id
        self._sender_q = None
        self._sender_thr = None


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
    
    def draw_dets(self, frame, boxes, scores, clses):
        for (x1, y1, x2, y2), s, c in zip(boxes, scores, clses):
            p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            label = f"{GestureDetector.CLASSES.get(int(c), str(int(c)))} {s:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (p1[0], p1[1]-th-8), (p1[0]+tw+6, p1[1]), (0,255,0), -1)
            cv2.putText(frame, label, (p1[0]+3, p1[1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
        return frame

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
                "is_new": None,
                "event_fired": False,
                "boxes": None,
                "scores": None,
                "clses": None
            },
            "eye_on_since": 0.0,
        }

        self.detector = YuNetFaceDetector(model_path=self.detector_path, face_score_thres=self.face_score_thres,
                                          width=self.width, height=self.height)

        # Start sender thread (non-blocking HTTP poster)
        self._sender_q = tqueue.Queue(maxsize=1)
        self._sender_thr = EventSenderThread(self._sender_q, self.server_url)
        self._sender_thr.start()
        print(f"[sender] Event sender → {self.server_url} (device_id={self.device_id})")

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

        if self.gesture_path and os.path.exists(self.gesture_path):
            self.gesture_q = tqueue.Queue(maxsize=1)
            self.gesture_proc = GestureProcess(self.gesture_q, self.shared, self._stop, self.gesture_path)
            self.gesture_proc.start()
        else:
            print(f"[gesture] onnx not found: {self.gesture_path}")

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
                submit(self.gesture_q, frame_bgr)

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
                    if isinstance(res, dict):
                        with self.shared["lock"]:
                            now = time()
                            ec = bool(res.get("eye_contact", False))
                            if ec:
                                if not self.shared.get("eye_on_since"):
                                    self.shared["eye_on_since"] = now
                                dwell = now - (self.shared["eye_on_since"] or now)
                            else:
                                self.shared["eye_on_since"] = 0.0
                                dwell = 0.0
                                # reset latch để lần nhìn sau có thể gửi tiếp
                                self.shared["result"]["event_fired"] = False

                            self.shared["result"]["eye_contact"] = ec
                            self.shared["result"]["eye_contact_dwell"] = float(max(0.0, dwell))
                            self.shared["result"]["ts"] = now

                            # Nếu đang eye contact và vượt ngưỡng 3.0s, bắn event 1 lần
                            if ec and dwell >= 3.0 and not self.shared["result"]["event_fired"]:
                                # Lấy snapshot metadata
                                pid = self.shared["result"]["person_id"]
                                is_new = bool(self.shared["result"].get("is_new") or False)
                                age_val = self.shared["result"]["age"]
                                gender_val = self.shared["result"]["gender"]
                                emotion_val = self.shared["result"]["emotion"]
                                face_rgb2 = self.shared["latest_face"]

                                # Chuẩn hoá giá trị gửi
                                gender_api = (gender_val.lower() if isinstance(gender_val, str) else None)
                                age_api = int(round(age_val)) if isinstance(age_val, (int, float)) else None
                                person_id_api = (str(pid) if pid is not None else None)

                                # Ảnh webp base64 nếu là người mới
                                face_b64 = None
                                try:
                                    face_b64 = _encode_webp_base64(face_rgb2, size=(112, 112), quality=80)
                                except Exception as e:
                                    print(f"[sender] encode face webp failed: {e}")

                                payload = {
                                    "device_id": self.device_id,
                                    "gaze_seconds": float(dwell),
                                    "is_new": bool(is_new),
                                    "age": age_api,
                                    "gender": gender_api,
                                    "emotion": emotion_val,
                                    "person_id": person_id_api,
                                    "face_b64": face_b64,
                                    "extra": {"source": "detect_worker"}
                                }
                                try:
                                    self._sender_q.put_nowait(payload)
                                    self.shared["result"]["event_fired"] = True
                                except tqueue.Full:
                                    print("[sender] queue full; drop event")
                        # enqueue nhận diện như cũ
                        if self._recog_q is not None and self.shared["result"]["eye_contact"]:
                            with self.shared["lock"]:
                                face_rgb2 = self.shared["latest_face"]
                                ver2 = self.shared["face_ver"]
                            if face_rgb2 is not None and ver2 != last_recog_ver:
                                try:
                                    self._recog_q.put_nowait(face_rgb2.copy())
                                    last_recog_ver = ver2
                                except tqueue.Full:
                                    pass
                        else:
                            with self.shared["lock"]:
                                self.shared["result"]["person_id"] = None
            except Empty:
                pass

            # Always draw overlay from shared results
            out = self._draw_overlay(frame_bgr)
            with self.shared["lock"]:
                g_boxes = self.shared["result"].get("boxes")
                g_scores = self.shared["result"].get("scores")
                g_clses = self.shared["result"].get("clses")
            if g_boxes is not None and len(g_boxes):
                out = self.draw_dets(out, g_boxes, g_scores, g_clses)
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
        try:
            if self.gesture_q is not None:
                self.gesture_q.put_nowait(None)
        except Exception:
            pass
        try:
            if self.gesture_proc is not None:
                self.gesture_proc.join(timeout=0.5)
        except Exception:
            if self.gesture_proc.is_alive():
                self.gesture_proc.stop()
            pass
        # Stop sender thread
        try:
            if self._sender_q is not None:
                self._sender_q.put_nowait(None)
        except Exception:
            pass
        try:
            if self._sender_thr is not None:
                self._sender_thr.join(timeout=0.5)
        except Exception:
            pass


    def stop(self):
        self._stop.set()
        try:
            self.in_q.put_nowait(None)
        except Exception:
            pass