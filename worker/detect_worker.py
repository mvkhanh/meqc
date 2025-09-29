from model.face_detector import YuNetFaceDetector
from model.agegender_recognizer import AgeGenderRecognizer
from model.emotion_recognizer import EmotionRecognizer
from multiprocessing import Process, Queue, Event
from typing import Optional
import numpy as np
import cv2
from utils import submit, poll
from time import time, sleep
from threading import Thread, Lock


class DetectWorker(Process):
    """
    Multiprocessing worker:
      - Nhận frame qua in_q
      - YuNet chỉ phát hiện khuôn mặt lớn nhất -> lưu crop vào biến dùng chung
      - Hai luồng nền chạy song song: Age/Gender và Emotion, đọc crop chung để suy luận
      - Kết quả (age, gender, emotion) được lưu vào dict chung
      - Vẽ kết quả lên frame và gửi ra out_q
    """

    def __init__(self, detector_path, agegender_path, detect_every_n: int, face_score_thres=0.8, width=640, height=640,
                 in_q: Optional[Queue] = None, out_q: Optional[Queue] = None, emotion_path: str = None):
        super().__init__(daemon=True)
        self.detect_every_n = detect_every_n
        self.detector = None
        self.agegender = None
        self.emotion = None

        self.detector_path = detector_path
        self.agegender_path = agegender_path
        self.emotion_path = emotion_path

        self.face_score_thres = face_score_thres
        self.width = width
        self.height = height
        self.in_q = in_q
        self.out_q = out_q
        self.frame_idx = 0
        self._stop = Event()

        # Biến chia sẻ giữa các luồng trong cùng Process
        self.shared = None

    # -------------------- Luồng nền cho Age/Gender --------------------
    def _agegender_loop(self):
        last_ver = -1
        while not self._stop.is_set():
            # Lấy ảnh khuôn mặt mới nhất nếu có phiên bản mới
            with self.shared["lock"]:
                ver = self.shared["face_ver"]
                face = self.shared["latest_face"].copy() if (self.shared["latest_face"] is not None and ver != last_ver) else None
            if face is None:
                sleep(0.005)
                continue
            try:
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                age_years, gender_label, gender_conf = self.agegender.detect(face_rgb)
            except Exception:
                age_years, gender_label, gender_conf = None, None, None

            with self.shared["lock"]:
                res = self.shared["result"]
                res["age"] = age_years
                res["gender"] = gender_label
                res["gender_conf"] = gender_conf
                res["ts"] = time()
            last_ver = ver
        # end while

    # -------------------- Luồng nền cho Emotion --------------------
    def _emotion_loop(self):
        if self.emotion is None:
            return
        last_ver = -1
        while not self._stop.is_set():
            with self.shared["lock"]:
                ver = self.shared["face_ver"]
                face = self.shared["latest_face"].copy() if (self.shared["latest_face"] is not None and ver != last_ver) else None
            if face is None:
                sleep(0.005)
                continue
            try:
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                emo_label, emo_conf = self.emotion.detect(face_rgb)
            except Exception:
                emo_label, emo_conf = None, None

            with self.shared["lock"]:
                res = self.shared["result"]
                res["emotion"] = emo_label
                res["emotion_conf"] = emo_conf
                res["ts"] = time()
            last_ver = ver
        # end while

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

    # -------------------- Main process loop --------------------
    def run(self):
        self.shared = {
            "lock": Lock(),
            "latest_face": None,   # numpy BGR crop
            "face_ver": 0,         # tăng mỗi khi có crop mới
            "last_box": None,      # (x,y,w,h) của khuôn mặt lớn nhất lần detect gần nhất
            "result": {            # Kết quả tổng hợp để YuNet vẽ lên frame
                "age": None,
                "gender": None,
                "gender_conf": None,
                "emotion": None,
                "emotion_conf": None,
                "ts": 0.0
            }
        }
        # Load models
        self.detector = YuNetFaceDetector(model_path=self.detector_path, face_score_thres=self.face_score_thres,
                                          width=self.width, height=self.height)
        self.agegender = AgeGenderRecognizer(model_path=self.agegender_path)
        self.emotion = EmotionRecognizer(model_path=self.emotion_path) if self.emotion_path else None

        # Start background threads
        ag_thread = Thread(target=self._agegender_loop, daemon=True)
        ag_thread.start()
        em_thread = None
        if self.emotion is not None:
            em_thread = Thread(target=self._emotion_loop, daemon=True)
            em_thread.start()

        # Main loop: nhận frame, (định kỳ) detect -> cập nhật crop + vẽ overlay
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
                            self.shared["latest_face"] = face_crop.copy()
                            self.shared["face_ver"] += 1
                            self.shared["last_box"] = (x, y, w, h)
                else:
                    with self.shared["lock"]:
                        self.shared["last_box"] = None

            # Luôn vẽ overlay từ kết quả chung (nếu có)
            out = self._draw_overlay(frame_bgr)
            t1 = time()
            if t1 > t0:
                print(f"FPS: {1.0 / (t1 - t0)} - Inference time: {t1 - t0}s")
            submit(self.out_q, out)
            self.frame_idx += 1

        # Kết thúc
        try:
            if ag_thread.is_alive():
                ag_thread.join(timeout=0.1)
        except Exception:
            pass
        try:
            if em_thread is not None and em_thread.is_alive():
                em_thread.join(timeout=0.1)
        except Exception:
            pass

    def stop(self):
        self._stop.set()
        try:
            self.in_q.put_nowait(None)
        except Exception:
            pass