'''
Hand gesture detection thread
'''
import queue as tqueue
from threading import Thread
from model.gesture_detector import GestureDetector
class GestureProcess(Thread):
    """Thread that consumes face crops and updates shared state with person_id."""
    def __init__(self, in_q: tqueue.Queue, shared: dict, stop_event,
                 onnx_path: str):
        super().__init__(daemon=True)
        self.in_q = in_q
        self.shared = shared
        self.stop_event = stop_event
        self.det = GestureDetector(onnx_path)

    def run(self):
        while not self.stop_event.is_set():
            try:
                item = self.in_q.get(timeout=0.1)
            except tqueue.Empty:
                continue
            if item is None:
                break
            boxes, scores, clses = self.det.detect(item)
            with self.shared["lock"]:
                self.shared["result"]["boxes"] = boxes
                self.shared["result"]["scores"] = scores
                self.shared["result"]["clses"] = clses