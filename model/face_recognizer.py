import onnxruntime as ort
from typing import Optional
from torchvision import transforms
import os
import numpy as np
from PIL import Image
from db.face_db import _FaissDB
from face_alignment import align

class FaceRecognizer:
    """ONNX embedding + FAISS DB wrapper."""
    def __init__(self, onnx_path: str, sim_thres: float = 0.45,
                 providers=("CPUExecutionProvider",), db_path: Optional[str] = None, autosave: bool = True):
        self.sess = ort.InferenceSession(onnx_path, providers=list(providers))
        self.in_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]

        # lấy kích thước input model
        in0 = self.sess.get_inputs()[0]
        self.in_shape = tuple(x for x in (in0.shape or []))
        self.is_nchw = (len(self.in_shape) == 4 and self.in_shape[1] in (1, 3))
        if self.is_nchw:
            self.exp_h, self.exp_w = int(self.in_shape[2]), int(self.in_shape[3])
        else:
            if len(self.in_shape) == 4:
                self.exp_h, self.exp_w = int(self.in_shape[1]), int(self.in_shape[2])
            else:
                self.exp_h, self.exp_w = 112, 112

        self.sim_thres = float(sim_thres)
        self.db_path = db_path
        self.autosave = bool(autosave)

        # ✅ thêm resize vào transform
        self.transform = transforms.Compose([
            transforms.Resize((self.exp_h, self.exp_w)),
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
        if aligned is None:
            aligned = face_rgb
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