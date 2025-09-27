from model.face_detector import YuNetFaceDetector
from model.agegender_recognizer import AgeGenderRecognizer
from model.emotion_recognizer import EmotionRecognizer
from multiprocessing import Process, Queue, Event
from typing import Optional
import numpy as np
import cv2
from utils import submit, poll

class DetectWorker(Process):
    """
    Multiprocessing worker: receive frames via in_q, run detection, send annotated frames via out_q.
    Keeps only the newest frame in both queues to avoid backlog.
    """
    def __init__(self, detector_path, agegender_path, detect_every_n: int, face_score_thres=0.8, width=640, height=640,
                 in_q: Optional[Queue] = None, out_q: Optional[Queue] = None, emotion_path: str = None):
        super().__init__(daemon=True)
        self.detect_every_n = detect_every_n
        self.detector = None
        self.agegender = None
        self.detector_path = detector_path
        self.agegender_path = agegender_path
        self.face_score_thres = face_score_thres
        self.width = width
        self.height = height
        self.in_q = in_q
        self.out_q = out_q
        self.frame_idx = 0
        self._stop = Event()
        self.emotion_path = emotion_path
        self.emotion = None

    # ----- Child process lifecycle -----
    def _init_in_child(self):
        # Construct heavy objects in child process
        self.detector = self.detector_cls()

    def run(self):
        self.detector = YuNetFaceDetector(model_path=self.detector_path, face_score_thres=self.face_score_thres,
                                          width=self.width, height=self.height)
        self.agegender = AgeGenderRecognizer(model_path=self.agegender_path)
        self.emotion = EmotionRecognizer(model_path=self.emotion_path) if self.emotion_path else None
        static = {"boxes": [], "attrs": []}  # attrs: [(age, gender_label, gender_conf, emo_label, emo_conf), ...] aligned với boxes
        while not self._stop.is_set():
            frame = poll(self.in_q)
            if frame is None:
                continue

            out = self.annotate_and_encode(frame, frame_idx=self.frame_idx, static=static)
            self.frame_idx += 1

            if out is not None:
                submit(self.out_q, out)

    def stop(self):
        self._stop.set()
        try:
            self.in_q.put_nowait(None)
        except Exception:
            pass
        
    # ----- Detection & drawing -----
    def annotate_and_encode(self, frame_bgr: np.ndarray, frame_idx = 0, static=None):
        if static is None:
            static = {"boxes": [], "attrs": []}

        # Chỉ detect lại theo chu kỳ hoặc khi chưa có boxes
        need_detect = (frame_idx % self.detect_every_n == 0) or (not static.get("boxes"))

        if need_detect:
            # 1) Face detection (YuNet)
            boxes = self.detector.detect(frame_bgr) or []

            # 2) Age/Gender cho từng bbox
            attrs = []
            H, W = frame_bgr.shape[:2]
            for (x, y, w, h) in boxes:
                # Clamp bbox trong khung hình
                x0 = max(0, x); y0 = max(0, y)
                x1 = min(W, x + w); y1 = min(H, y + h)
                if x1 - x0 < 16 or y1 - y0 < 16:
                    attrs.append((None, None, None, None, None))
                    continue
                face_crop = frame_bgr[y0:y1, x0:x1]
                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                try:
                    age_years, gender_label, gender_conf = self.agegender.detect(face_crop_rgb)
                except Exception:
                    age_years, gender_label, gender_conf = None, None, None
                # Emotion
                emo_label, emo_conf = (None, None)
                if self.emotion is not None:
                    try:
                        emo_label, emo_conf = self.emotion.detect(face_crop_rgb)
                    except Exception:
                        emo_label, emo_conf = None, None
                attrs.append((age_years, gender_label, gender_conf, emo_label, emo_conf))

            static["boxes"] = boxes
            static["attrs"] = attrs

        # 3) Vẽ kết quả lên frame
        boxes = static.get("boxes", [])
        attrs = static.get("attrs", [])
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Lấy info nếu có
            if i < len(attrs):
                age_years, gender_label, gender_conf, emo_label, emo_conf = attrs[i]
            else:
                age_years, gender_label, gender_conf, emo_label, emo_conf = None, None, None, None, None

            # Chuẩn bị text
            parts = []
            if gender_label is not None and gender_conf is not None:
                parts.append(f"{gender_label} {int(round(gender_conf*100))}%")
            elif gender_label is not None:
                parts.append(f"{gender_label}")
            if age_years is not None:
                parts.append(f"{int(round(age_years))}y")
            if emo_label is not None and emo_conf is not None:
                parts.append(f"{emo_label} {int(round(emo_conf*100))}%")
            elif emo_label is not None:
                parts.append(f"{emo_label}")
            text = " | ".join(parts)

            if text:
                # Vẽ nền mờ phía trên bbox cho dễ đọc
                (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                pad = 4
                x_bg0 = x
                y_bg0 = max(0, y - th - baseline - 2*pad)
                x_bg1 = min(frame_bgr.shape[1]-1, x + tw + 2*pad)
                y_bg1 = y
                cv2.rectangle(frame_bgr, (x_bg0, y_bg0), (x_bg1, y_bg1), (0, 0, 0), thickness=-1)
                cv2.putText(frame_bgr, text, (x + pad, y - baseline - pad),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        return frame_bgr