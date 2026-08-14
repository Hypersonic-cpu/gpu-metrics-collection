#!/usr/bin/env python3
"""Compactly check whether every TK case's target kernel reached Nsys."""

import argparse
import csv
import json
import pathlib
import sys


REPO = pathlib.Path(__file__).resolve().parent.parent
TK_DIR = REPO / "scripts/thunderkittens"
sys.path.insert(0, str(TK_DIR))
from p0_manifest import load_manifest  # noqa: E402


def diagnostic_verdict(path):
    if not path.is_file():
        return "MISSING"
    first = path.read_text(errors="replace").splitlines()
    if not first:
        return "EMPTY"
    line = first[0]
    for verdict in ("PASS", "PARTIAL", "FAIL", "ERROR"):
        if verdict in line:
            return verdict
    return "UNKNOWN"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=pathlib.Path)
    args = parser.parse_args(argv)
    run_dir = args.run.expanduser().resolve()
    config_path = run_dir / "suite_config.json"
    if not config_path.is_file():
        parser.error("missing {}".format(config_path))
    config = json.loads(config_path.read_text())
    manifest_path = pathlib.Path(config["manifest"])
    _, cases = load_manifest(manifest_path)
    selected = config["cases"]
    delay = config.get("nsys_delay_s", "unknown")
    duration = config.get("nsys_duration_s", "unknown")
    failures = []
    print("RUN={} cases={} delay={}s duration={}s".format(
        run_dir.name, len(selected), delay, duration))
    for case_id in selected:
        target = cases[case_id]["target_kernel"]
        kernel_csv = run_dir / case_id / "nsys/kernel_cuda_gpu_kern_sum.csv"
        validation_path = run_dir / case_id / "validation.json"
        diagnostics_path = run_dir / case_id / "nsys/diagnostics.txt"
        instances = 0
        average_us = None
        if kernel_csv.is_file():
            with kernel_csv.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    if target in row.get("Name", ""):
                        instances += int(row["Instances"])
                        average_us = float(row["Avg (ns)"]) / 1000.0
        validation = "MISSING"
        if validation_path.is_file():
            validation = json.loads(validation_path.read_text()).get(
                "status", "UNKNOWN")
        diagnostic = diagnostic_verdict(diagnostics_path)
        passed = instances > 0 and validation == "PASS" and diagnostic == "PASS"
        status = "PASS" if passed else "FAIL"
        if not passed:
            failures.append(case_id)
        average = "-" if average_us is None else "{:.3f}".format(average_us)
        print("{} {} target={} instances={} avg_us={} validation={} diagnostics={}".format(
            status, case_id, target, instances, average, validation, diagnostic))
    if failures:
        print("SUMMARY FAIL {}/{} missing_or_invalid={}".format(
            len(failures), len(selected), ",".join(failures)))
        return 1
    print("SUMMARY PASS {}/{} target kernels captured".format(
        len(selected), len(selected)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
