import queue as pyqueue
import imagezmq, zmq
import numpy as np

def submit(in_q, frame_bgr):
    try:
        if in_q.full():
            _ = in_q.get_nowait()  # drop oldest
        in_q.put_nowait(frame_bgr)
    except Exception:
        pass
    
def poll(out_q):
    """Drain out_q and return the latest object (or None)."""
    last = None
    try:
        while True:
            last = out_q.get_nowait()
    except pyqueue.Empty:
        return last

def _softmax(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x = x - np.max(x)
        e = np.exp(x)
        s = e / (np.sum(e) + 1e-9)
        return s