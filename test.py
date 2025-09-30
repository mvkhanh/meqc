#!/usr/bin/env python3
"""
Quick test runner for a Hailo HEF.
- Loads a HEF you provide
- Allocates a random input tensor matching the model's input shape
- Binds ALL outputs by name (handles multi-output models)
- Runs one async inference and prints output names, shapes, and a small value preview

Usage:
  python test.py --hef ckpt/agegender.hef --timeout 10000
Optional:
  --seed 0            # for reproducibility of random input
  --float32           # try to set input/output FormatType to FLOAT32 (default on)
  --no-float32        # do not set FLOAT32 (use model defaults)
"""
import argparse
import numpy as np

try:
    from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType
except Exception as e:
    raise SystemExit(f"Hailo SDK not available: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", required=True, help="Path to .hef file")
    ap.add_argument("--timeout", type=int, default=10000, help="Timeout ms for async job")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for input")
    ap.add_argument("--float32", dest="set_fp32", action="store_true", default=True,
                    help="Try to set input/output FormatType to FLOAT32 (default)")
    ap.add_argument("--no-float32", dest="set_fp32", action="store_false",
                    help="Do not set FLOAT32 on streams")
    args = ap.parse_args()

    np.random.seed(args.seed)

    # Create a VDevice with multi-process sharing allowed (harmless here)
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    params.group_id = "SHARED"
    params.multi_process_service = True

    with VDevice(params) as vdevice:
        infer_model = vdevice.create_infer_model(args.hef)
        infer_model.set_batch_size(1)

        if args.set_fp32:
            try:
                infer_model.input().set_format_type(FormatType.FLOAT32)
                infer_model.output().set_format_type(FormatType.FLOAT32)
            except Exception:
                pass

        with infer_model.configure() as configured:
            # ---- Input binding ----
            try:
                in_info = infer_model.input()
                in_shape = tuple(int(d) for d in in_info.shape)
                in_dtype = np.float32
            except Exception:
                # Fallback: ask configured for input vstream infos
                infos_in = configured.get_input_vstream_infos()
                if not infos_in:
                    raise RuntimeError("Cannot query input vstream info")
                in_shape = tuple(int(d) for d in infos_in[0].shape)
                in_dtype = np.float32

            # Allocate random input matching the model's expected input shape
            x = np.random.rand(*in_shape).astype(in_dtype)
            in_b = configured.create_input_binding()
            in_b.set_buffer(x)

            # ---- Output bindings (multi-output aware) ----
            out_names = []
            output_bindings = []
            try:
                infos = configured.get_output_vstream_infos()
                out_names = [info.name for info in infos]
            except Exception:
                try:
                    out_names = infer_model.get_sorted_output_names()
                except Exception:
                    out_names = []

            if out_names:
                for name in out_names:
                    try:
                        out_info = infer_model.output(name)
                        out_shape = tuple(int(d) for d in out_info.shape)
                        out_dtype = np.float32
                    except Exception:
                        out_shape, out_dtype = tuple(int(d) for d in infer_model.output().shape), np.float32
                    buf = np.empty(out_shape, dtype=out_dtype)
                    out_b = configured.create_output_binding(name)
                    out_b.set_buffer(buf)
                    output_bindings.append(out_b)
                bindings_list = [in_b] + output_bindings
            else:
                # Single-output fallback
                out_info = infer_model.output()
                out_shape = tuple(int(d) for d in out_info.shape)
                out_dtype = np.float32
                single_buf = np.empty(out_shape, dtype=out_dtype)
                single_out_b = configured.create_output_binding()
                single_out_b.set_buffer(single_buf)
                output_bindings = [single_out_b]
                bindings_list = [in_b, single_out_b]

            # ---- Run once ----
            configured.wait_for_async_ready(timeout_ms=args.timeout)
            job = configured.run_async(bindings_list, lambda *_: None)
            job.wait(args.timeout)

            # ---- Collect & print ----
            print("\n=== Inference Results ===")
            if out_names:
                for name, b in zip(out_names, output_bindings):
                    arr = b.get_buffer()
                    print(f"- {name}: shape={arr.shape}, dtype={arr.dtype}")
                    # Print a tiny preview (first up to 6 values)
                    flat = arr.reshape(-1)
                    preview = ", ".join(f"{v:.4f}" for v in flat[:6])
                    print(f"  preview: [{preview}]\n")
            else:
                arr = output_bindings[0].get_buffer()
                print(f"- output: shape={arr.shape}, dtype={arr.dtype}")
                flat = arr.reshape(-1)
                preview = ", ".join(f"{v:.4f}" for v in flat[:6])
                print(f"  preview: [{preview}]\n")


if __name__ == "__main__":
    main()
