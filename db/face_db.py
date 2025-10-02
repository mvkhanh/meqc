import faiss
import numpy as np

# -------------------- Face Recognition (ONNX + FAISS) --------------------
class _FaissDB:
    """Minimal FAISS-backed DB with cosine similarity (via inner product on L2-normalized vectors).
    Falls back to NumPy if faiss is unavailable.
    """
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
