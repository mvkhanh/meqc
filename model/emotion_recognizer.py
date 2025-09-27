import os
from typing import Tuple
import numpy as np
import onnxruntime as ort
import torchvision.transforms as T
from PIL import Image


class EmotionRecognizer:
    EMO_LABELS = ['Angry', 'Fear', 'Happiness', 'Sad', 'Surprise', 'Neutral']

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise RuntimeError(f"Không tìm thấy file {model_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
        self.emo_sess = ort.InferenceSession(model_path, providers=providers)

        EMO_IN = self.emo_sess.get_inputs()[0]
        self.EMO_IN_NAME = EMO_IN.name
        # Lấy H, W từ onnx (fallback 224 nếu dynamic)
        _, _, EMO_H, EMO_W = [int(dim) if isinstance(dim, int) and dim > 0 else 224 for dim in EMO_IN.shape]
        self.height = int(EMO_H)
        self.width = int(EMO_W)
        self.transform_emo = self.create_transform()

    def create_transform(self):
        norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = T.Compose([
            T.Resize((self.height, self.width)),
            T.ToTensor(),
            norm
        ])
        return transform

    def preprocess(self, face_crop_rgb) -> np.ndarray:
        """
        face_crop_rgb: numpy RGB (H,W,3)
        returns: NCHW float32 np.array shape (1,3,H,W)
        """
        tensor = self.transform_emo(Image.fromarray(face_crop_rgb)).unsqueeze(0).numpy()
        return tensor.astype(np.float32, copy=False)

    def detect(self, face_crop_rgb) -> Tuple[str, float]:
        """
        Trả về (emotion_label, confidence)
        """
        inp = self.preprocess(face_crop_rgb)
        logits = self.emo_sess.run(None, {self.EMO_IN_NAME: inp})[0]  # (1, C)

        # softmax an toàn số
        x = logits.reshape(-1)
        x = x - np.max(x)
        e = np.exp(x, dtype=np.float64)
        probs = (e / e.sum()).astype(np.float32)

        pred_idx = int(np.argmax(probs))
        emotion_label = EmotionRecognizer.EMO_LABELS[pred_idx] if pred_idx < len(self.EMO_LABELS) else str(pred_idx)
        confidence = float(probs[pred_idx])
        return emotion_label, confidence