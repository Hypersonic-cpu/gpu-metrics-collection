#!/usr/bin/env python3
"""Validate one P1 native/Nsys/DCGM rotation collection."""

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
from p1_manifest import DEFAULT_CSV, DEFAULT_ROTATION_CONFIG, load_p1_suite  # noqa: E402


def run(command, env=None):
    print("[VALIDATE CMD] {}".format(" ".join(map(str, command))), flush=True)
    subprocess.run(list(map(str, command)), check=True, env=env)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=pathlib.Path, default=DEFAULT_CSV)
    parser.add_argument("--rotation-config", type=pathlib.Path,
                        default=DEFAULT_ROTATION_CONFIG)
    parser.add_argument("--case", required=True)
    parser.add_argument("--run", required=True, type=pathlib.Path)
    parser.add_argument("--nsys-bin", default=os.environ.get(
        "NSYS_BIN", "/usr/local/cuda/bin/nsys"))
    args = parser.parse_args(argv)
    result = {"case_id": args.case, "status": "FAIL"}
    status_path = args.run / "validation.json"
    try:
        _, _, cases, rotations = load_p1_suite(
            args.csv, args.rotation_config, [args.case])
        case = cases[args.case]
        expected_copies = rotations[args.case]["copies"]
        nsys_dir = args.run / "nsys"
        report = nsys_dir / "report.nsys-rep"
        sqlite = nsys_dir / "report.sqlite"
        kernel_prefix = nsys_dir / "kernel"
        run([args.nsys_bin, "export", "--type", "sqlite",
             "--force-overwrite=true", "--output", sqlite, report])
        run([args.nsys_bin, "stats", "--force-export=true", "--report",
             "cuda_gpu_kern_sum", "--format", "csv", "--output",
             kernel_prefix, report])
        env = os.environ.copy()
        env["NSYS_HOST_BIN"] = args.nsys_bin
        run([sys.executable, HERE.parent / "tool/nsys/check_report.py", report],
            env)
        run([sys.executable, HERE / "validate_nsys_kernel.py", sqlite,
             "--kernel", case["target_kernel"], "--require-metrics"])
        run([sys.executable, HERE.parent / "tool/nsys/post_export.py",
             nsys_dir, "--reuse-sqlite"])

        native_text = (args.run / "native/run.log").read_text(errors="replace")
        workload_text = (args.run / "workload.log").read_text(errors="replace")
        timings = [float(value) for value in re.findall(
            r"TK: ([0-9.]+) ms", native_text)]
        expected_timings = case["world_size"]
        if len(timings) != expected_timings:
            raise ValueError("native log has {} TK timings, expected {}".format(
                len(timings), expected_timings))
        correctness = "ROTATION_CORRECTNESS_PASS case={} copies={}".format(
            args.case, expected_copies)
        if correctness not in native_text:
            raise ValueError("native rotation correctness did not pass")

        allocation_pattern = re.compile(
            r"ROTATION_ALLOC_DONE case={} copies=(\d+) .*?"
            r"working_set_bytes_per_gpu=(\d+) .*?"
            r"fingerprint_first=([^\s]+) fingerprint_last=([^\s]+)"
            .format(re.escape(args.case)))
        native_allocation = allocation_pattern.search(native_text)
        workload_allocation = allocation_pattern.search(workload_text)
        if not native_allocation or not workload_allocation:
            raise ValueError("rotation allocation evidence is missing")
        if native_allocation.groups() != workload_allocation.groups():
            raise ValueError("native/profile rotation fingerprints differ")
        copies, working_set, first_fp, last_fp = native_allocation.groups()
        if int(copies) != expected_copies:
            raise ValueError("rotation copy count differs from JSON")
        exceeds_l2 = int(working_set) > 50 * 1024 ** 2
        warmup = "ROTATION_WARMUP_DONE case={} rounds=1 kernels={}".format(
            args.case, expected_copies)
        if warmup not in workload_text:
            raise ValueError("profile did not warm every rotating buffer")
        if "ROTATION_MEASURE_BEGIN case={}".format(args.case) not in workload_text:
            raise ValueError("profile measurement marker is missing")

        kernel_csv = nsys_dir / "kernel_cuda_gpu_kern_sum.csv"
        rows = list(csv.DictReader(kernel_csv.open()))
        target = next(row for row in rows if case["target_kernel"] in row["Name"])
        dcgm_csvs = list((args.run / "dcgm").glob("*/metrics.csv"))
        if len(dcgm_csvs) != 1:
            raise ValueError("expected one DCGM metrics.csv, got {}".format(
                len(dcgm_csvs)))
        dcgm_rows = list(csv.DictReader(dcgm_csvs[0].open()))
        dcgm_values = [row for row in dcgm_rows
                       if row.get("value") not in (None, "", "N/A")]
        required = {"dram_active", "nvlink_rx_bytes", "nvlink_tx_bytes",
                    "pcie_rx_bytes", "pcie_tx_bytes"}
        seen = {row["metric"] for row in dcgm_values}
        if not required <= seen:
            raise ValueError("DCGM is missing usable metrics: {}".format(
                ",".join(sorted(required - seen))))
        dcgm_gpus = {row["gpu"] for row in dcgm_values
                     if row["metric"] in required}
        if len(dcgm_gpus) != case["world_size"]:
            raise ValueError("DCGM covers {} GPUs, expected {}".format(
                len(dcgm_gpus), case["world_size"]))
        post_rows = sum(1 for _ in csv.DictReader(
            (nsys_dir / "post_metrics.csv").open()))
        if post_rows == 0:
            raise ValueError("Nsys post_metrics.csv has no samples")
        result.update({
            "status": "PASS", "family": case["family"],
            "world_size": case["world_size"],
            "target_kernel": case["target_kernel"],
            "kernel_instances": int(target["Instances"]),
            "kernel_avg_us": float(target["Avg (ns)"]) / 1000.0,
            "native_rank_mean_ms": statistics.fmean(timings),
            "native_rank_max_ms": max(timings),
            "nsys_metric_rows": post_rows,
            "dcgm_value_rows": len(dcgm_values), "dcgm_gpus": len(dcgm_gpus),
            "buffer_rotation": {
                "copies": int(copies),
                "working_set_bytes_per_gpu": int(working_set),
                "working_set_exceeds_50mib": exceeds_l2,
                "json_meets_2gib": rotations[args.case]["meets_2gib"],
                "fingerprint_first": first_fp, "fingerprint_last": last_fp,
                "warmup_rounds": 1, "correctness": "PASS",
                "reproducible_native_profile_fingerprints": True,
            },
        })
    except Exception as error:
        result["error"] = str(error)
    status_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[VALIDATE {}][{}] {}".format(
        result["status"], args.case, json.dumps(result, sort_keys=True)),
        flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
