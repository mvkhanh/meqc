import os
from typing import List, Tuple
import numpy as np
import cv2

class YuNetFaceDetector:
    def __init__(self, model_path, face_score_thres=0.8, width=640, height=640):
        if not os.path.exists(model_path):
            raise RuntimeError(f"Không tìm thấy file {model_path}")
        self.det = cv2.FaceDetectorYN_create(model=model_path, config="", input_size=(width, height), score_threshold=face_score_thres)
        self.det.setInputSize((width, height))
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int,int,int,int]]:
        h, w = frame.shape[:2]
        # đảm bảo input size khớp frame hiện tại
        self.det.setInputSize((w, h))
        retval, faces = self.det.detect(frame)
        if faces is None or len(faces) == 0:
            return []
        # YuNet trả Nx15: [x,y,w,h,score, 10 landmark]
        boxes = faces[:, :4].astype(int).tolist()
        return boxes