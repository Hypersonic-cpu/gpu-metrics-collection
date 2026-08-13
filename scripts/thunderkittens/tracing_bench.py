#!/usr/bin/env python3
"""MHA-only profiling copy of Accel-Sim's tracing_bench.py."""

import argparse
import importlib.util
import os
import pathlib
import sys
import time


ADAPTER_ROOT = pathlib.Path(os.environ.get(
    "HTM_TK_ADAPTER_ROOT",
    pathlib.Path.home() / "htm-workspace/htm-accel-sim/util/tracer_nvbit/workloads/thunderkittens",
)).resolve()
sys.path.insert(0, str(ADAPTER_ROOT))
spec = importlib.util.spec_from_file_location(
    "htm_original_tracing_bench", ADAPTER_ROOT / "tracing_bench.py"
)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="MHA_SMOKE")
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args(argv)
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    manifest = original.load_manifest()
    case = original.find_case(manifest, args.case)
    if case["family"] != "MHA":
        parser.error("this profiling copy supports MHA cases only")

    tk_root = pathlib.Path(os.environ.get(
        "TK_ROOT", pathlib.Path.home() / "Repos/ThunderKittens"
    )).resolve()
    original.verify_tk_checkout(tk_root)
    original.stage(0, "process_start")

    import torch

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    module = original.load_extension(tk_root, "MHA")
    started = time.monotonic()
    original.stage(0, "gpu_alloc_begin")
    main_launch, _ = original.prepare_mha(case, module, device)
    original.stage(0, "gpu_alloc_end", started)
    torch.cuda.synchronize()

    roi_name = "HTM_TARGET:{}".format(args.case)
    print("[rank=0] ROI_READY case={} iterations={}".format(
        args.case, args.iterations), flush=True)
    torch.cuda.nvtx.range_push(roi_name)
    print("[rank=0] ROI_BEGIN case={} name={} iterations={}".format(
        args.case, roi_name, args.iterations), flush=True)
    for _ in range(args.iterations):
        main_launch()
    torch.cuda.synchronize()
    print("[rank=0] ROI_END case={} iterations={}".format(
        args.case, args.iterations), flush=True)
    torch.cuda.nvtx.range_pop()
    print("[rank=0] EXIT_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
