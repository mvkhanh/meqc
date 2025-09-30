#!/usr/bin/env python3
import argparse
import time
import cv2
from multiprocessing import Process, Queue, Event
from worker.detect_worker import DetectWorker
from utils import submit, poll, create_sender

class CaptureWorker(Process):
    def __init__(self, in_q: Queue, width: int=640, height: int=640):
        super().__init__(daemon=True)
        self._stop = Event()
        self.in_q = in_q
        self.width = width
        self.height = height
        self.cam = None
        
    def run(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # print(f"FPS: {self.cam.get(cv2.CAP_PROP_FPS)}")
        try:
            while not self._stop.is_set():
                ok, frame_bgr = self.cam.read()
                if not ok:
                    time.sleep(0.02); continue
                submit(self.in_q, frame_bgr)
        finally:
            self.cam.release()
            
    def stop(self):
        self._stop.set()

def _check_quit_key():
    k = cv2.waitKey(1) & 0xFF
    return k in (27, ord('q'), ord('Q'))

def main(args):
    if args.server:
        sender = create_sender(args.server, args.port)
        consecutive_fail = 0
        MAX_FAILS = 3  # quÃ¡ 3 láº§n lá»i liÃªn tiáº¿p thÃ¬ dá»«ng
        
    else:
        cv2.namedWindow('Streaming', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Streaming', width=args.width, height=args.height)
    
    in_q = Queue(maxsize=1)
    out_q = Queue(maxsize=1)
    capture_worker = CaptureWorker(in_q=in_q, width=args.width, height=args.height)
    detect_worker = DetectWorker(detector_path=args.face_detection_model, detect_every_n=args.den,
                                 face_score_thres=args.face_thres, agegender_path=args.agegender_model,
                                 width=args.width, height=args.height, in_q=in_q, out_q=out_q,
                                 emotion_path=args.emotion_model)
    
    capture_worker.start()
    detect_worker.start()
    try:
        while True:
            frame = poll(out_q)

            # Chưa có frame mới -> đợi nhẹ rồi tiếp
            if frame is None:
                if args.server:
                    time.sleep(0.02)
                else:
                    if _check_quit_key():
                        break
                continue

            # ĐÃ có frame -> gửi hoặc hiển thị
            if args.server:
                ok, tmp = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
                jpg = tmp.tobytes() if ok else None
                if jpg is None:
                    continue

                try:
                    _ = sender.send_jpg(args.name, jpg)
                    consecutive_fail = 0  # reset khi gửi OK
                except Exception as e:
                    consecutive_fail += 1
                    print(f"[CLIENT] send_jpg failed ({consecutive_fail}/{MAX_FAILS}): {e}")
                    if consecutive_fail >= MAX_FAILS:
                        print("[CLIENT] Server unreachable. Stopping client.")
                        break

                    time.sleep(0.5)
                    # Recreate socket
                    try:
                        sender.zmq_socket.close(0)
                        sender.zmq_context.term()
                    except Exception:
                        pass
                    sender = create_sender(args.server, args.port)
            else:
                cv2.imshow('Streaming', frame)
                if _check_quit_key():
                    break
                    
    except KeyboardInterrupt:
        pass
    finally:
        if args.server:
            try:
                sender.zmq_socket.close(0)
            except Exception:
                pass
            try:
                sender.zmq_context.term()
            except Exception:
                pass
        else:
            cv2.destroyAllWindows()
        
        # Graceful shutdown of workers
        try:
            detect_worker.stop()
        except Exception:
            pass
        try:
            capture_worker.stop()
        except Exception:
            pass

        # Join and, if needed, terminate
        for p in (detect_worker, capture_worker):
            try:
                p.join(timeout=1.0)
            except Exception:
                pass
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Raspberry Pi 5 Realtime Client")
    
    parser.add_argument("--server", default="", help="IP/host cua PC server, bo trong de hien thi local")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--name", default="pi")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--den", type=int, default=3, help="detect_every_n")
    parser.add_argument("--quality", type=int, default=80, help='Image quality when send to server')
    parser.add_argument("--face-thres", type=float, default=0.8, help='Threshold for face detection')
    parser.add_argument("--face-detection-model", default='ckpt/face_detection_yunet_2023mar.onnx', help="Tham so cua yunet")
    parser.add_argument("--emotion-model", default='ckpt/emotion.hef', help="Tham so cua emotion")
    parser.add_argument("--agegender-model", default='ckpt/agegender.hef', help="Tham so cua age gender")
    # parser.add_argument("--gaze-model", default='ckpt/resnet34_gaze.opset17.onnx', help="Tham so cua gaze")

    args = parser.parse_args()
      
    main(args)
