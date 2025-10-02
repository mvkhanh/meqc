from time import time
from threading import Thread
from queue import Queue
from multiprocessing import Event
from model.face_recognizer import FaceRecognizer

class RecogThread(Thread):
    """Thread that consumes face crops and updates shared state with person_id."""
    def __init__(self, in_q: Queue, shared: dict, stop_event: Event,
                 onnx_path: str, sim_thres: float = 0.45):
        super().__init__(daemon=True)
        self.in_q = in_q
        self.shared = shared
        self.stop_event = stop_event
        self.recog = FaceRecognizer(onnx_path, sim_thres)

    def run(self):
        while not self.stop_event.is_set():
            try:
                item = self.in_q.get(timeout=0.1)
            except Queue.Empty:
                continue
            if item is None:
                break
            try:
                pid, sim, is_new = self.recog.identify_or_enroll(item)
                with self.shared["lock"]:
                    self.shared["result"]["person_id"] = int(pid)
                    self.shared["result"]["ts"] = time()
            except Exception as e:
                # best-effort; keep thread alive
                print(f"[recog] error: {e}")
