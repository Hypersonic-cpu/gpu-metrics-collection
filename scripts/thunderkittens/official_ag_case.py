#!/usr/bin/env python3
"""Run one shape through ThunderKittens' unmodified official AG benchmark."""

import argparse
import importlib.util
import os
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--num-comm-sms", type=int, default=8)
    args = parser.parse_args()
    if args.size <= 0 or args.size % 1024:
        parser.error("--size must be a positive multiple of 1024")
    if args.size % 2048:
        parser.error("H100 AG requires --size to be a multiple of 2048")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("invalid warmup/iteration count")

    ag_dir = pathlib.Path(os.environ.get(
        "TK_ROOT", pathlib.Path.home() / "Repos/ThunderKittens"
    )) / "kernels/parallel/ag_gemm"
    sys.path.insert(0, str(ag_dir))
    spec = importlib.util.spec_from_file_location(
        "tk_official_ag_benchmark", ag_dir / "benchmark.py"
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    rank, world = benchmark.init_distributed_environment()
    if world != 8:
        raise RuntimeError("official H100 AG benchmark requires 8 ranks")
    if rank == 0:
        print(
            "OFFICIAL_AG_BEGIN size={} M={} K={} N={} warmup={} iterations={} "
            "check_correctness=false do_profile=false".format(
                args.size, args.size, args.size, args.size // world,
                args.warmup, args.iterations,
            ),
            flush=True,
        )
    benchmark.run(
        args.size,
        args.size,
        args.size // world,
        args.num_comm_sms,
        rank,
        world,
        num_warmup_iters=args.warmup,
        num_iters=args.iterations,
        check_correctness=False,
        do_profile=False,
    )
    benchmark.destroy_distributed_environment()
    return 0


if __name__ == "__main__":
    sys.exit(main())
