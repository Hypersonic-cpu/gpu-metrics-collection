#!/usr/bin/env python3
"""Validate and summarize one collected official P0 profiling result."""

import argparse
import csv
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "thunderkittens"))
from p0_manifest import load_manifest  # noqa: E402


def run(command, env=None):
    print("[VALIDATE CMD] {}".format(" ".join(map(str, command))), flush=True)
    subprocess.run(list(map(str, command)), check=True, env=env)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--group", default="p0")
    parser.add_argument("--case", required=True)
    parser.add_argument("--run", required=True, type=pathlib.Path)
    parser.add_argument("--nsys-bin", default=os.environ.get(
        "NSYS_BIN", "/usr/local/cuda/bin/nsys"))
    args = parser.parse_args(argv)
    result = {"case_id": args.case, "status": "FAIL"}
    status_path = args.run / "validation.json"
    try:
        _, cases = load_manifest(args.manifest, args.group)
        case = cases[args.case]
        nsys_dir = args.run / "nsys"
        report = nsys_dir / "report.nsys-rep"
        sqlite = nsys_dir / "report.sqlite"
        kernel_prefix = nsys_dir / "kernel"
        run([args.nsys_bin, "export", "--type", "sqlite",
             "--force-overwrite=true", "--output", sqlite, report])
        run([args.nsys_bin, "stats", "--force-export=true",
             "--report", "cuda_gpu_kern_sum", "--format", "csv",
             "--output", kernel_prefix, report])
        env = os.environ.copy()
        env["NSYS_HOST_BIN"] = args.nsys_bin
        run([sys.executable, HERE.parent / "tool/nsys/check_report.py", report], env)
        run([sys.executable, HERE / "validate_nsys_kernel.py", sqlite,
             "--kernel", case["target_kernel"], "--require-metrics"])
        run([sys.executable, HERE.parent / "tool/nsys/post_export.py",
             nsys_dir, "--reuse-sqlite"])

        native_text = (args.run / "native/run.log").read_text(errors="replace")
        timings = [float(value) for value in re.findall(r"TK: ([0-9.]+) ms", native_text)]
        if len(timings) != case["world_size"]:
            raise ValueError("native log has {} TK timings, expected {}".format(
                len(timings), case["world_size"]))
        kernel_csv = nsys_dir / "kernel_cuda_gpu_kern_sum.csv"
        rows = list(csv.DictReader(kernel_csv.open()))
        target = next(row for row in rows if case["target_kernel"] in row["Name"])
        dcgm_csvs = list((args.run / "dcgm").glob("*/metrics.csv"))
        if len(dcgm_csvs) != 1:
            raise ValueError("expected one DCGM metrics.csv, got {}".format(len(dcgm_csvs)))
        dcgm_rows = list(csv.DictReader(dcgm_csvs[0].open()))
        dcgm_values = [row for row in dcgm_rows
                       if row.get("value") not in (None, "", "N/A")]
        if not dcgm_values:
            raise ValueError("DCGM contains no usable values")
        required_dcgm = {"dram_active", "nvlink_rx_bytes", "nvlink_tx_bytes",
                         "pcie_rx_bytes", "pcie_tx_bytes"}
        seen_dcgm = {row["metric"] for row in dcgm_values}
        if not required_dcgm <= seen_dcgm:
            raise ValueError("DCGM is missing usable metrics: {}".format(
                ",".join(sorted(required_dcgm - seen_dcgm))))
        dcgm_gpus = {row["gpu"] for row in dcgm_values
                     if row["metric"] in required_dcgm}
        if len(dcgm_gpus) != case["world_size"]:
            raise ValueError("DCGM covers {} GPUs, expected {}".format(
                len(dcgm_gpus), case["world_size"]))
        post_rows = sum(1 for _ in csv.DictReader(
            (nsys_dir / "post_metrics.csv").open()))
        if post_rows == 0:
            raise ValueError("Nsys post_metrics.csv has no samples")
        result.update({
            "status": "PASS",
            "family": case["family"],
            "target_kernel": case["target_kernel"],
            "kernel_instances": int(target["Instances"]),
            "kernel_avg_us": float(target["Avg (ns)"]) / 1000.0,
            "native_rank_mean_ms": statistics.fmean(timings),
            "native_rank_max_ms": max(timings),
            "nsys_metric_rows": post_rows,
            "dcgm_value_rows": len(dcgm_values),
            "dcgm_gpus": len(dcgm_gpus),
        })
    except Exception as error:
        result["error"] = str(error)
    status_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[VALIDATE {}][{}] {}".format(
        result["status"], args.case, json.dumps(result, sort_keys=True)), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
