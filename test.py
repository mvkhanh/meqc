# hailo_quicktest.py
import argparse
import numpy as np
from functools import partial
from multiprocessing import Process
from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType


def cb_print_stats(completion_info, bindings):
    """Callback: in ra thống kê output để xác nhận pipeline chạy OK."""
    if completion_info.exception:
        print("[callback] ❌ Exception:", completion_info.exception)
        return
    out = bindings.output().get_buffer()
    # In stats gọn nhẹ
    try:
        print(f"[callback] ✓ Output shape={tuple(out.shape)} "
              f"min={float(out.min()):.5f} max={float(out.max()):.5f} mean={float(out.mean()):.5f}")
    except Exception as e:
        print("[callback] (note) Could not compute stats:", e)


def infer_worker(args):
    # 1) Tạo VDevice với chia sẻ giữa nhiều process (nếu bật)
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    params.group_id = args.group_id
    if args.mps:
        params.multi_process_service = True

    timeout_ms = args.timeout_ms

    with VDevice(params) as vdev:
        # 2) Load model từ HEF
        infer_model = vdev.create_infer_model(args.hef)

        # 3) (Tuỳ chọn) set batch size
        if args.batch and args.batch > 0:
            infer_model.set_batch_size(args.batch)

        # 4) Kiểu dữ liệu I/O (giữ theo mẫu hướng dẫn)
        infer_model.input().set_format_type(FormatType.FLOAT32)
        infer_model.output().set_format_type(FormatType.FLOAT32)

        # 5) Configure model -> chạy async
        with infer_model.configure() as cmodel:
            jobs = []
            for i in range(args.frames):
                # a) Bindings + buffer
                bindings = cmodel.create_bindings()

                in_shape = infer_model.input().shape
                out_shape = infer_model.output().shape  # chỉ để log
                print(out_shape)
                # Sinh input ngẫu nhiên theo đúng shape, float32
                in_buf = np.random.rand(*in_shape).astype(np.float32)
                out_buf = np.empty(out_shape, dtype=np.float32)

                bindings.input().set_buffer(in_buf)
                bindings.output().set_buffer(out_buf)

                # b) Đợi pipeline sẵn sàng rồi run async
                cmodel.wait_for_async_ready(timeout_ms=timeout_ms)
                job = cmodel.run_async([bindings], partial(cb_print_stats, bindings=bindings))
                jobs.append(job)

            # 6) Đợi tất cả job hoàn thành (đảm bảo callback đã chạy)
            for j in jobs:
                j.wait(timeout_ms)

    print(f"[proc {args.proc_name}] ✓ Done {args.frames} frame(s), batch={args.batch}")


def main():
    ap = argparse.ArgumentParser(description="Minimal Hailo HEF async inference quick test.")
    ap.add_argument("--hef", required=True, help="Đường dẫn file .hef")
    ap.add_argument("--frames", type=int, default=4, help="Số frame giả lập/inference mỗi process")
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