# hailo_quicktest.py
import argparse
import numpy as np
from functools import partial
from multiprocessing import Process
from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType


# ---------- Helpers ----------
def list_input_names(im):
    """Trả về list tên input. Có fallback cho API khác nhau."""
    if hasattr(im, "get_input_names"):
        return list(im.get_input_names())
    if hasattr(im, "inputs"):
        try:
            return list(im.inputs.keys())            # dict-like
        except Exception:
            try:
                return [x.name for x in im.inputs]   # list of objects
            except Exception:
                pass
    # Fallback: single input (không tên)
    return [None]


def list_output_names(im):
    """Trả về list tên output. Có fallback cho API khác nhau."""
    if hasattr(im, "get_output_names"):
        return list(im.get_output_names())
    if hasattr(im, "outputs"):
        try:
            return list(im.outputs.keys())
        except Exception:
            try:
                return [x.name for x in im.outputs]
            except Exception:
                pass
    # Fallback: single output (không tên)
    return [None]


def io_shape(im, is_input, name):
    """Lấy shape cho input/output theo name (hoặc None nếu single)."""
    if is_input:
        return (im.input(name).shape if name is not None else im.input().shape)
    else:
        return (im.output(name).shape if name is not None else im.output().shape)


def io_set_format(im, is_input, name, fmt):
    """Set format type cho input/output (từng tên)."""
    if is_input:
        (im.input(name) if name is not None else im.input()).set_format_type(fmt)
    else:
        (im.output(name) if name is not None else im.output()).set_format_type(fmt)


def bindings_set_buffer(bindings, is_input, name, buf):
    """Gán buffer cho bindings theo tên (hoặc None nếu single)."""
    if is_input:
        (bindings.input(name) if name is not None else bindings.input()).set_buffer(buf)
    else:
        (bindings.output(name) if name is not None else bindings.output()).set_buffer(buf)


def bindings_get_buffer(bindings, name):
    """Lấy buffer output theo tên (hoặc None nếu single)."""
    return (bindings.output(name) if name is not None else bindings.output()).get_buffer()


# ---------- Callback ----------
def cb_print_stats(completion_info, bindings, output_names):
    """Callback: in thống kê cho từng output để xác nhận pipeline OK."""
    if completion_info.exception:
        print("[callback] ❌ Exception:", completion_info.exception)
        return

    for idx, name in enumerate(output_names):
        try:
            out = bindings_get_buffer(bindings, name)
            nm = name if name is not None else "<default>"
            print(f"[callback] ✓ out[{idx}] '{nm}' shape={tuple(out.shape)} "
                  f"min={float(out.min()):.6f} max={float(out.max()):.6f} mean={float(out.mean()):.6f}")
        except Exception as e:
            print(f"[callback] (note) Could not read stats for output[{idx}] name={name}: {e}")


# ---------- Worker ----------
def infer_worker(args):
    # 1) VDevice (có thể chia sẻ giữa nhiều process)
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    params.group_id = args.group_id
    if args.mps:
        params.multi_process_service = True

    with VDevice(params) as vdev:
        # 2) Load HEF
        infer_model = vdev.create_infer_model(args.hef)

        # 3) (Optional) Batch size
        if args.batch and args.batch > 0:
            infer_model.set_batch_size(args.batch)

        # 4) Lấy danh sách tên I/O
        input_names = list_input_names(infer_model)
        output_names = list_output_names(infer_model)

        # 5) Set format cho tất cả I/O (FLOAT32 theo sample)
        for n in input_names:
            io_set_format(infer_model, True, n, FormatType.FLOAT32)
        for n in output_names:
            io_set_format(infer_model, False, n, FormatType.FLOAT32)

        # (Log ngắn gọn)
        try:
            in_desc = [f"{n if n else '<default>'}:{tuple(io_shape(infer_model, True, n))}" for n in input_names]
            out_desc = [f"{n if n else '<default>'}:{tuple(io_shape(infer_model, False, n))}" for n in output_names]
            print(f"[{args.proc_name}] inputs={in_desc}")
            print(f"[{args.proc_name}] outputs={out_desc}")
        except Exception as e:
            print(f"[{args.proc_name}] (note) Could not print I/O shapes: {e}")

        # 6) Configure & chạy async
        with infer_model.configure() as cmodel:
            jobs = []
            for i in range(args.frames):
                bindings = cmodel.create_bindings()

                # a) Gán buffer cho tất cả input
                for n in input_names:
                    shape = io_shape(infer_model, True, n)
                    in_buf = np.random.rand(*shape).astype(np.float32)
                    bindings_set_buffer(bindings, True, n, in_buf)

                # b) Tạo buffer cho tất cả output
                for n in output_names:
                    shape = io_shape(infer_model, False, n)
                    out_buf = np.empty(shape, dtype=np.float32)
                    bindings_set_buffer(bindings, False, n, out_buf)

                # c) Đợi pipeline sẵn sàng và run async
                cmodel.wait_for_async_ready(timeout_ms=args.timeout_ms)
                job = cmodel.run_async(
                    [bindings],
                    partial(cb_print_stats, bindings=bindings, output_names=output_names)
                )
                jobs.append(job)

            # 7) Đợi toàn bộ job
            for j in jobs:
                j.wait(args.timeout_ms)

        print(f"[proc {args.proc_name}] ✓ Done {args.frames} frame(s), batch={args.batch}")


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Minimal Hailo HEF async inference quick test (multi I/O).")
    ap.add_argument("--hef", required=True, help="Đường dẫn file .hef")
    ap.add_argument("--frames", type=int, default=4, help="Số frame/inference mỗi process")
    ap.add_argument("--batch", type=int, default=1, help="Batch size đặt cho infer_model")
    ap.add_argument("--procs", type=int, default=1, help="Số process song song")
    ap.add_argument("--mps", action="store_true", help="Bật multi_process_service để chia sẻ Hailo giữa nhiều process")
    ap.add_argument("--group-id", default="SHARED", help="Group ID dùng chung khi bật MPS")
    ap.add_argument("--timeout-ms", type=int, default=10000, help="Timeout cho wait_for_async_ready & job.wait")
    args = ap.parse_args()

    if args.procs <= 1:
        args.proc_name = "P0"
        print("▶️  Start single-process inference test")
        infer_worker(args)
    else:
        print(f"▶️  Start multi-process inference test: procs={args.procs}, MPS={'ON' if args.mps else 'OFF'}")
        pool = []
        for i in range(args.procs):
            a = argparse.Namespace(**vars(args))
            a.proc_name = f"P{i}"
            p = Process(target=infer_worker, args=(a,), daemon=False)
            pool.append(p)
        for p in pool:
            p.start()
        for p in pool:
            p.join()

    print("✅ All done.")


if __name__ == "__main__":
    main()