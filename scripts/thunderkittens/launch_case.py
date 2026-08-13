#!/usr/bin/env python3
"""MHA profiling copy of Accel-Sim's ThunderKittens launch_case.py."""

import argparse
import os
import pathlib
import subprocess
import sys


HERE = pathlib.Path(__file__).resolve().parent
ADAPTER_ROOT = pathlib.Path(os.environ.get(
    "HTM_TK_ADAPTER_ROOT",
    pathlib.Path.home() / "htm-workspace/htm-accel-sim/util/tracer_nvbit/workloads/thunderkittens",
)).resolve()
sys.path.insert(0, str(ADAPTER_ROOT))
from validate_manifest import find_case, load_manifest  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="MHA_SMOKE")
    parser.add_argument("--devices", default="5")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if len(devices) != 1 or args.num_gpus != 1:
        parser.error("this profiling copy supports exactly one MHA GPU")
    case = find_case(load_manifest(), args.case)
    if case["family"] != "MHA" or case["world_size"] != 1:
        parser.error("this profiling copy supports MHA cases only")

    print(
        "case={} family=MHA physical_gpu={} iterations={} expected_main={}".format(
            args.case, devices[0], args.iterations,
            case["expected_source_main_kernel"],
        ),
        flush=True,
    )
    if args.dry_run:
        return 0

    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("ARCH", "SM90")
    environment["CUDA_VISIBLE_DEVICES"] = devices[0]
    command = [
        sys.executable,
        str(HERE / "tracing_bench.py"),
        "--case", args.case,
        "--iterations", str(args.iterations),
    ]
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
