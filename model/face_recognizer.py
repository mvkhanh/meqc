import onnxruntime as ort
from torchvision import transforms
from PIL import Image
import numpy as np
from face_alignment import align
from db.face_db import _FaissDB

class FaceRecognizer:
    """ONNX embedding + FAISS DB wrapper."""
    def __init__(self, onnx_path: str, sim_thres: float = 0.45,
                 providers=("CUDAExecutionProvider", "CPUExecutionProvider")):
        self.sess = ort.InferenceSession(onnx_path, providers=list(providers))
        self.in_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.sim_thres = float(sim_thres)
        # preprocessing identical to PyTorch code
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.db = None  # _FaissDB, lazily created after first embedding

    def _preprocess(self, face_rgb: np.ndarray) -> np.ndarray:
        # align.get_aligned_face may accept PIL or ndarray
        img = Image.fromarray(face_rgb) if not isinstance(face_rgb, Image.Image) else face_rgb
        aligned = align.get_aligned_face(img)
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
        return pid, 1.0, True
