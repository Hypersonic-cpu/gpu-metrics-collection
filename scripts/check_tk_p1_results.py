#!/usr/bin/env python3
"""Compactly validate already-produced P1 case summaries without re-exporting."""

import argparse
import json
import pathlib
import sys


def verdict(path):
    if not path.is_file():
        return "MISSING"
    lines = path.read_text(errors="replace").splitlines()
    first = lines[0] if lines else ""
    return next((value for value in ("PASS", "PARTIAL", "FAIL", "ERROR")
                 if value in first), "UNKNOWN")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=pathlib.Path)
    args = parser.parse_args(argv)
    run = args.run.expanduser().resolve()
    config_path = run / "suite_config.json"
    if not config_path.is_file():
        parser.error("missing {}".format(config_path))
    config = json.loads(config_path.read_text())
    failures = []
    print("RUN={} cases={} delay={}s duration={}s frequency={}Hz".format(
        run.name, len(config["cases"]), config.get("nsys_delay_s"),
        config.get("nsys_duration_s"), config.get("nsys_frequency_hz")))
    for case_id in config["cases"]:
        case_dir = run / case_id
        validation_path = case_dir / "validation.json"
        if not validation_path.is_file():
            failures.append(case_id)
            native = case_dir / "native/run.log"
            reason = "no validation"
            if native.is_file():
                lines = native.read_text(errors="replace").splitlines()
                if lines:
                    reason = lines[-1]
            print("FAIL {} reason={}".format(case_id, reason))
            continue
        result = json.loads(validation_path.read_text())
        diagnostics = verdict(case_dir / "nsys/diagnostics.txt")
        rotation = result.get("buffer_rotation") or {}
        passed = (
            result.get("status") == "PASS" and
            diagnostics == "PASS" and
            result.get("kernel_instances", 0) > 0 and
            result.get("nsys_metric_rows", 0) > 0 and
            result.get("dcgm_value_rows", 0) > 0 and
            result.get("dcgm_gpus") == result.get("world_size", result.get("dcgm_gpus")) and
            rotation.get("correctness") == "PASS" and
            rotation.get("reproducible_native_profile_fingerprints") is True
        )
        if not passed:
            failures.append(case_id)
        print("{} {} target={} kernels={} nsys_rows={} dcgm_rows={} "
              "dcgm_gpus={} diagnostics={} rotation={}x".format(
                  "PASS" if passed else "FAIL", case_id,
                  result.get("target_kernel", "-"),
                  result.get("kernel_instances", 0),
                  result.get("nsys_metric_rows", 0),
                  result.get("dcgm_value_rows", 0),
                  result.get("dcgm_gpus", 0), diagnostics,
                  rotation.get("copies", 0)))
    if failures:
        print("SUMMARY FAIL passed={}/{} failed={}".format(
            len(config["cases"]) - len(failures), len(config["cases"]),
            ",".join(failures)))
        return 1
    print("SUMMARY PASS {}/{} cases".format(
        len(config["cases"]), len(config["cases"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
