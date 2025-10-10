import faiss
import json
import numpy as np
# -------------------- Face Recognition (ONNX + FAISS) --------------------
class _FaissDB:
    """Minimal FAISS-backed DB with cosine similarity (via inner product on L2-normalized vectors).
    Falls back to NumPy if faiss is unavailable.
    """
    FAISS_NAME = 'faiss.index'
    METADATA_NAME = 'metadata.json'
    @classmethod
    def load(cls, path: str):
        """Load FAISS index and metadata saved by `save`.
        """
        idx = faiss.read_index(path + _FaissDB.FAISS_NAME)
        with open(path + _FaissDB.METADATA_NAME, "r") as f:
            meta = json.load(f)
        dim = int(meta.get("dim", idx.d))
        obj = cls(dim)
        obj.index = idx
        obj.ids = list(meta.get("ids", []))
        obj.next_id = int(meta.get("next_id", len(obj.ids) + 1))
        return obj

    def save(self, path: str):
        """Persist FAISS index and metadata.
        Writes two files: `{path}.index` (FAISS) and `{path}.json` (ids/next_id/meta).
        """
        faiss.write_index(self.index, path + _FaissDB.FAISS_NAME)
        meta = {
            "ids": self.ids,
            "next_id": self.next_id,
            "dim": self.dim,
            "ntotal": int(self.index.ntotal),
        }
        with open(path + _FaissDB.METADATA_NAME, "w") as f:
            json.dump(meta, f)
            
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