from threading import Thread
import queue as tqueue
from typing import Optional
from model.face_recognizer import FaceRecognizer
import os
from time import time
from PIL import Image
from face_alignment import align
import numpy as np
import cv2

# --- Helpers to normalize image inputs ---
def ensure_pil_rgb(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 2:  # gray -> 3 channels
            arr = np.stack([arr] * 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(x)}")

# --- Helper: Save RGB/grayscale face image as WebP with quality ---
def _save_webp_image(rgb: np.ndarray, out_path: str, quality: int = 80):
    """
    Lưu ảnh (RGB hoặc grayscale) thành WebP tại out_path với chất lượng 'quality'.
    Tự tạo thư mục nếu chưa có.
    """
    # Bảo đảm thư mục tồn tại
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Chuyển sang BGR trước khi encode bằng OpenCV
    if rgb.ndim == 2:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".webp", bgr, [cv2.IMWRITE_WEBP_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("encode webp failed")
    with open(out_path, "wb") as f:
        f.write(buf.tobytes())

class RecogThread(Thread):
    """Thread that consumes face crops and updates shared state with person_id."""
    def __init__(self, in_q: tqueue.Queue, shared: dict, stop_event,
                 onnx_path: str, sim_thres: float = 0.45, db_path: Optional[str] = None, autosave: bool = True,
                 images_dir: Optional[str] = None):
        super().__init__(daemon=True)
        self.in_q = in_q
        self.shared = shared
        self.stop_event = stop_event
        self.recog = FaceRecognizer(onnx_path, sim_thres, db_path=db_path, autosave=autosave)
        self.images_dir = images_dir or os.path.join("db", "images")
        os.makedirs(self.images_dir, exist_ok=True)

    def run(self):
        while not self.stop_event.is_set():
            try:
                item = self.in_q.get(timeout=0.1)
            except tqueue.Empty:
                continue
            if item is None:
                break
            pid, sim, is_new = self.recog.identify_or_enroll(item)
            with self.shared["lock"]:
                self.shared["result"]["person_id"] = int(pid)
                self.shared["result"]["is_new"] = bool(is_new)
                self.shared["result"]["ts"] = time()
            # Lưu một ảnh WebP/ID nếu chưa tồn tại
            try:
                out_path = os.path.join(self.images_dir, f"{int(pid)}.webp")
                if not os.path.exists(out_path):
                    # Cố gắng dùng ảnh đã align; nếu lỗi thì dùng ảnh gốc
                    try:
                        pil_img = Image.fromarray(item) if not isinstance(item, Image.Image) else item
                        aligned = align.get_aligned_face(pil_img)
                        # Always end up with a valid PIL RGB image
                        pil_aligned = ensure_pil_rgb(aligned) if aligned is not None else ensure_pil_rgb(pil_img)
                        arr_rgb = np.array(pil_aligned, dtype=np.uint8)  # RGB uint8
                    except Exception:
                        # fallback to the original crop
                        pil_fallback = ensure_pil_rgb(item)
                        arr_rgb = np.array(pil_fallback, dtype=np.uint8)
                    _save_webp_image(arr_rgb, out_path, quality=80)
            except Exception as e:
                print(f"[recog] warn: failed to save face image for ID#{pid}: {e}")