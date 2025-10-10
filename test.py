# yolo10_cam.py
import cv2
import time
import numpy as np
import onnxruntime as ort

# ====== Cấu hình ======
ONNX_PATH = "ckpt/YOLOv10n_gestures.onnx"
CONF_THRES = 0.25

CLASSES = {
    0:"grabbing",1:"grip",2:"holy",3:"point",4:"call",5:"three3",6:"timeout",7:"xsign",
    8:"hand_heart",9:"hand_heart2",10:"little_finger",11:"middle_finger",12:"take_picture",
    13:"dislike",14:"fist",15:"four",16:"like",17:"mute",18:"ok",19:"one",20:"palm",
    21:"peace",22:"peace_inverted",23:"rock",24:"stop",25:"stop_inverted",
    26:"three",27:"three2",28:"two_up",29:"two_up_inverted",30:"three_gun",
    31:"thumb_index",32:"thumb_index2",33:"no_gesture"
}

# ====== Helpers ======
def get_input_size_and_layout(sess, default=224):
    s = sess.get_inputs()[0].shape
    # s dạng [N,3,H,W] hoặc [N,H,W,3] hoặc dynamic
    if len(s) == 4 and isinstance(s[2], int) and isinstance(s[3], int):
        size = int(s[2]) if s[2] == s[3] else default
    else:
        size = default
    layout = "NCHW" if (len(s) == 4 and s[1] == 3) else "NHWC"
    return size, layout

def letterbox_like_ultra(img_bgr, size=224, pad_val=114):
    h, w = img_bgr.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    im = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    dw = (size - nw) / 2
    dh = (size - nh) / 2
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))
    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))

    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            borderType=cv2.BORDER_CONSTANT,
                            value=(pad_val, pad_val, pad_val))
    return im, r, left, top

def preprocess(img_bgr, size, layout):
    im_lb, r, pad_left, pad_top = letterbox_like_ultra(img_bgr, size=size)
    im = cv2.cvtColor(im_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if layout == "NCHW":
        im = np.transpose(im, (2, 0, 1))  # CHW
    # add batch & contiguous
    im = np.expand_dims(np.ascontiguousarray(im), 0)
    return im, r, pad_left, pad_top

def to_N6(out):
    """Chuẩn hoá output về (N,6)"""
    a = np.squeeze(out)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim == 2:
        if a.shape[1] == 6:
            return a
        if a.shape[0] == 6:
            return a.T
    raise RuntimeError(f"Unexpected output shape {a.shape}, cần (N,6) hoặc (6,N).")

def unletterbox_xyxy(boxes, r, pad_left, pad_top, w0, h0):
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top ) / r
    boxes[:, 0::2]   = np.clip(boxes[:, 0::2], 0, w0)
    boxes[:, 1::2]   = np.clip(boxes[:, 1::2], 0, h0)
    return boxes

def draw_dets(frame, boxes, scores, clses):
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, clses):
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        label = f"{CLASSES.get(int(c), str(int(c)))} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (p1[0], p1[1]-th-8), (p1[0]+tw+6, p1[1]), (0,255,0), -1)
        cv2.putText(frame, label, (p1[0]+3, p1[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
    return frame

# ====== Main ======
def main():
    # Session (ưu tiên CUDA nếu có, fallback CPU)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(ONNX_PATH, providers=providers)
    except Exception:
        sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    SIZE, LAYOUT = get_input_size_and_layout(sess, default=224)

    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    t0, fps = time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h0, w0 = frame.shape[:2]
        inp, r, pad_left, pad_top = preprocess(frame, SIZE, LAYOUT)
        out = sess.run(None, {inp_name: inp})[0]
        pred = to_N6(out)

        boxes  = pred[:, :4].astype(np.float32).copy()   # xyxy @ input-space
        scores = pred[:, 4].astype(np.float32)
        clses  = pred[:, 5].astype(np.int32)

        m = scores >= CONF_THRES
        boxes, scores, clses = boxes[m], scores[m], clses[m]
        boxes = unletterbox_xyxy(boxes, r, pad_left, pad_top, w0, h0)

        vis = draw_dets(frame.copy(), boxes, scores, clses)

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, (now - t0)))
        t0 = now
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("YOLOv10 ONNX (cam0)", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()