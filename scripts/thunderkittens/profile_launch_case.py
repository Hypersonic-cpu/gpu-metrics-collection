#!/usr/bin/env python3
"""Launch a manifest-selected TK profiling case on explicit physical GPUs."""

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
    parser.add_argument("--case", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--iterations", type=int, default=1_000_000_000)
    parser.add_argument("--native-samples", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices or any(not item.isdigit() for item in devices):
        parser.error("--devices must be comma-separated physical GPU IDs")
    if len(set(devices)) != len(devices):
        parser.error("--devices contains duplicates")
    case = find_case(load_manifest(), args.case)
    if len(devices) != case["world_size"]:
        parser.error("{} requires {} GPUs, got {}".format(
            args.case, case["world_size"], len(devices)))

    print("PROFILE_CASE case={} family={} devices={} processes={} iterations={}".format(
        args.case, case["family"], ",".join(devices), len(devices),
        args.iterations), flush=True)
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("ARCH", "SM90")
    bench = [str(HERE / "profile_bench.py"), "--case", args.case,
             "--iterations", str(args.iterations),
             "--native-samples", str(args.native_samples)]
    if len(devices) == 1:
        command = [sys.executable] + bench
    else:
        torchrun = pathlib.Path(sys.executable).with_name("torchrun")
        command = [str(torchrun), "--standalone",
                   "--nproc_per_node={}".format(len(devices))] + bench
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
