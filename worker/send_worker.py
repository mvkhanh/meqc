'''
Sending data to server
'''
import requests
from threading import Thread
import queue as tqueue
from multiprocessing import Event


# --- Event sender thread ---

# Thread gửi payload JSON lên server FastAPI khi đủ điều kiện (nhìn >= 3s).
# Dùng queue để không block DetectWorker loop.
class EventSenderThread(Thread):
    """
    Thread gửi payload JSON lên server FastAPI khi đủ điều kiện (nhìn &gt;= 3s).
    Dùng queue để không block DetectWorker loop.
    """
    def __init__(self, in_q: tqueue.Queue, server_url: str):
        super().__init__(daemon=True)
        self.in_q = in_q
        self.server_url = server_url
        self.stop_event = Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        while not self.stop_event.is_set():
            try:
                item = self.in_q.get(timeout=0.1)
            except tqueue.Empty:
                continue
            if item is None:
                break
            try:
                r = requests.post(self.server_url, json=item, timeout=3)
                # Có thể log r.status_code nếu cần
            except Exception as e:
                print(f"[sender] post error: {e}")