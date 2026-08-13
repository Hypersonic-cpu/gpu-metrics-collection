#!/usr/bin/env python3
"""Long-running single/8-GPU TK benchmark with ten native timing samples."""

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
    parser.add_argument("--case", required=True)
    parser.add_argument("--iterations", type=int, default=1_000_000_000)
    parser.add_argument("--native-samples", type=int, default=10)
    args = parser.parse_args(argv)
    if args.iterations <= 0 or not 0 < args.native_samples <= args.iterations:
        parser.error("require 0 < --native-samples <= --iterations")

    case = original.find_case(original.load_manifest(), args.case)
    tk_root = pathlib.Path(os.environ.get(
        "TK_ROOT", pathlib.Path.home() / "Repos/ThunderKittens"
    )).resolve()
    original.verify_tk_checkout(tk_root)

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world_size != case["world_size"]:
        raise RuntimeError("case requires world_size {}, got {}".format(
            case["world_size"], world_size))

    import torch

    distributed = world_size > 1
    if distributed:
        torch.distributed.init_process_group("nccl", device_id=rank)
    torch.cuda.set_device(rank)
    device = torch.device("cuda:{}".format(rank))
    module = original.load_extension(tk_root, case["family"])
    preparers = {
        "MHA": lambda: original.prepare_mha(case, module, device),
        "AG": lambda: original.prepare_ag(case, module, device, rank, world_size),
        "RS": lambda: original.prepare_rs(case, module, device, rank, world_size),
        "MoE": lambda: original.prepare_moe(case, module, device, rank, world_size),
    }
    main_launch, cleanup_launch = preparers[case["family"]]()
    if cleanup_launch is not None:
        raise RuntimeError("profiling loop expects the current combined binding")
    if distributed:
        torch.distributed.barrier()
    torch.cuda.synchronize()
    print("[rank={}] BENCH_READY case={} family={} iterations={}".format(
        rank, args.case, case["family"], args.iterations), flush=True)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for index in range(args.native_samples):
        if distributed:
            torch.distributed.barrier()
        start_event.record()
        main_launch()
        end_event.record()
        end_event.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        if distributed:
            sample = torch.tensor(elapsed_ms, dtype=torch.float64, device=device)
            torch.distributed.all_reduce(sample, op=torch.distributed.ReduceOp.MAX)
            elapsed_ms = sample.item()
        if rank == 0:
            print("NATIVE_SAMPLE index={} latency_ms={:.6f}".format(
                index, elapsed_ms), flush=True)

    if distributed:
        torch.distributed.barrier()
    torch.cuda.synchronize()
    roi_name = "HTM_TARGET:{}".format(args.case)
    if rank == 0:
        print("ROI_LOOP_BEGIN case={} name={} remaining={}".format(
            args.case, roi_name, args.iterations - args.native_samples), flush=True)

    torch.cuda.nvtx.range_push(roi_name)
    for _ in range(args.native_samples, args.iterations):
        main_launch()
        torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    if distributed:
        torch.distributed.destroy_process_group()
    if rank == 0:
        print("BENCH_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
