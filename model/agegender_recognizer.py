import os
from typing import Tuple
import numpy as np
import torchvision.transforms as T
import onnxruntime as ort
from PIL import Image

class AgeGenderRecognizer:
        
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise RuntimeError(f"Không tìm thấy file {model_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
        self.age_gender_sess = ort.InferenceSession(model_path, providers=providers)
        AG_IN = self.age_gender_sess.get_inputs()[0]
        self.AG_IN_NAME = AG_IN.name
        _, _, AG_H, AG_W = [int(dim) if isinstance(dim, int) and dim > 0 else 224 for dim in AG_IN.shape]
        self.height = AG_H
        self.width = AG_W
        self.transform_ag = self.create_transform()
    
    def create_transform(self):
        norm = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform_ag = T.Compose([
            T.Resize((self.height, self.width)),
            T.ToTensor(),
            norm
        ])
        return transform_ag

    def detect(self, face_crop: np.ndarray) -> Tuple[float, str, float]:
        tensor = self.transform_ag(Image.fromarray(face_crop)).unsqueeze(0).numpy()
        outs = self.age_gender_sess.run(None, {self.AG_IN_NAME: tensor})
        age_raw = float(np.asarray(outs[0]).squeeze())
        gender_logit = float(np.asarray(outs[1]).squeeze())

        age_years = float(np.clip(age_raw, 0, 100))

        prob_female = 1.0 / (1.0 + np.exp(-gender_logit))

        if prob_female >= 0.5:
            gender_label = "Female"
            gender_conf = prob_female
        else:
            gender_label = "Male"
            gender_conf = 1.0 - prob_female
        return age_years, gender_label, gender_conf