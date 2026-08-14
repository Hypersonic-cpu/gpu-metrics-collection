#!/usr/bin/env python3
"""Schedule the existing CSV P1 cases with JSON-sized buffer rotation."""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time


REPO = pathlib.Path(__file__).resolve().parent.parent
TK_DIR = REPO / "scripts/thunderkittens"
sys.path.insert(0, str(TK_DIR))
from p1_manifest import DEFAULT_CSV, DEFAULT_ROTATION_CONFIG, load_p1_suite  # noqa: E402


def csv_gpu_ids(value):
    items = value.split(",")
    if not items or any(not item.isdigit() for item in items):
        raise argparse.ArgumentTypeError(
            "must be comma-separated GPU IDs, e.g. 1,2,3")
    if len(items) != len(set(items)):
        raise argparse.ArgumentTypeError("GPU IDs must not repeat")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run P1 native -> Nsys -> DCGM -> kill -> validation")
    parser.add_argument("--csv", type=pathlib.Path, default=DEFAULT_CSV)
    parser.add_argument("--rotation-config", type=pathlib.Path,
                        default=DEFAULT_ROTATION_CONFIG)
    parser.add_argument("--cases", help="comma-separated case IDs; default: all P1")
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--devices", type=csv_gpu_ids,
                        default="0,1,2,3,4,5,6,7")
    parser.add_argument("--nsys-gpus", type=csv_gpu_ids, default="0")
    parser.add_argument("--nsys-duration", type=float, default=0.2)
    parser.add_argument("--nsys-frequency", type=int, default=100000)
    parser.add_argument("--nsys-delay", type=float, default=15)
    parser.add_argument("--dcgm-duration", type=float, default=5)
    parser.add_argument("--native-iters", type=int, default=20)
    parser.add_argument("--profile-iters", type=int, default=1000000000)
    parser.add_argument("--profile-timeout", type=int, default=3600)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate/print plan only; never query or use a GPU")
    args = parser.parse_args(argv)
    if len(args.devices.split(",")) != 8:
        parser.error("the P1 suite device pool must contain exactly 8 GPUs")
    if args.nsys_duration <= 0 or args.nsys_delay < 0 or args.dcgm_duration <= 0:
        parser.error("durations must be positive and delay nonnegative")
    if not 10 <= args.nsys_frequency <= 200000:
        parser.error("--nsys-frequency must be in [10, 200000]")
    if args.native_iters <= 0 or args.profile_iters <= 0:
        parser.error("iteration counts must be positive")
    if args.profile_timeout <= 0:
        parser.error("profile timeout must be positive")
    selected = args.cases.split(",") if args.cases else None
    try:
        csv_path, rotation_path, cases, rotations = load_p1_suite(
            args.csv, args.rotation_config, selected)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    selected = list(selected or cases)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = (args.out or REPO / "runs" /
                "tk_p1_rotation_{}".format(stamp)).resolve()
    extension_root = TK_DIR / "p1_extensions"
    physical_ids = args.devices.split(",")

    print("[SUITE] priority=P1 cases={} out={} devices={} nsys_gpus={} "
          "duration={}s frequency={}Hz delay={}s dcgm={}s".format(
              len(selected), out_root, args.devices, args.nsys_gpus,
              args.nsys_duration, args.nsys_frequency, args.nsys_delay,
              args.dcgm_duration), flush=True)
    for index, case_id in enumerate(selected, 1):
        case = cases[case_id]
        rotation = rotations[case_id]
        case_devices = args.devices if case["world_size"] == 8 else physical_ids[0]
        print("[PLAN {}/{}] {} family={} world={} devices={} target={} shape={} "
              "rotation={}x seed={} warmup_rounds=1 json_total_gib={}".format(
                  index, len(selected), case_id, case["family"],
                  case["world_size"], case_devices, case["target_kernel"],
                  case["execution"], rotation["copies"], case["seed_base"],
                  rotation["total_gib"]), flush=True)
    build_command = [sys.executable, TK_DIR / "build_p1_extensions.py"]
    if args.dry_run:
        subprocess.run(build_command + ["--dry-run"], check=True)
        for case_id in selected:
            case = cases[case_id]
            subprocess.run([
                sys.executable, TK_DIR / "p1_rotation_case.py",
                "--csv", csv_path, "--rotation-config", rotation_path,
                "--case", case_id, "--warmup", "1", "--iterations", "1",
                "--rotation-copies", str(rotations[case_id]["copies"]),
                "--rotation-seed-base", str(case["seed_base"]),
                "--extension-root", extension_root, "--dry-run",
            ], check=True)
        print("[DRY RUN PASS] all P1 cases/configurations are valid; GPU untouched")
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "suite_config.json").write_text(json.dumps({
        "csv": str(csv_path), "rotation_config": str(rotation_path),
        "cases": selected, "devices": args.devices,
        "nsys_gpus": args.nsys_gpus,
        "nsys_duration_s": args.nsys_duration,
        "nsys_frequency_hz": args.nsys_frequency,
        "nsys_delay_s": args.nsys_delay,
        "dcgm_duration_s": args.dcgm_duration,
        "native_iters": args.native_iters,
        "profile_iters": args.profile_iters,
        "profile_timeout_s": args.profile_timeout,
        "buffer_rotation": {case_id: rotations[case_id]
                            for case_id in selected},
    }, indent=2) + "\n")
    if not args.skip_build:
        print("[BUILD] repo-local P1 MHA/A2A/GEMM runners", flush=True)
        subprocess.run(build_command, check=True)

    results = {}
    for index, case_id in enumerate(selected, 1):
        case = cases[case_id]
        case_out = out_root / case_id
        case_devices = args.devices if case["world_size"] == 8 else physical_ids[0]
        env = os.environ.copy()
        env.update({
            "WORLD_SIZE": str(case["world_size"]),
            "CASE_DEVICES": case_devices, "NSYS_GPUS": args.nsys_gpus,
            "NSYS_DURATION": str(args.nsys_duration),
            "NSYS_FREQUENCY": str(args.nsys_frequency),
            "NSYS_DELAY": str(args.nsys_delay),
            "DCGM_DURATION": str(args.dcgm_duration),
            "NATIVE_ITERS": str(args.native_iters),
            "PROFILE_ITERS": str(args.profile_iters),
            "PROFILE_TIMEOUT": str(args.profile_timeout),
            "ROTATION_COPIES": str(rotations[case_id]["copies"]),
            "ROTATION_SEED_BASE": str(case["seed_base"]),
            "P1_EXTENSION_ROOT": str(extension_root),
        })
        print("[PROGRESS {}/{}][LAUNCH] {} family={} out={}".format(
            index, len(selected), case_id, case["family"], case_out),
            flush=True)
        rc = subprocess.run([
            REPO / "scripts/collect_tk_p1_case.sh", case_id, case_out,
            csv_path, rotation_path,
        ], env=env).returncode
        if rc:
            results[case_id] = "COLLECT_FAIL"
            print("[PROGRESS {}/{}][COLLECT FAIL] {} rc={}".format(
                index, len(selected), case_id, rc), flush=True)
            continue
        print("[PROGRESS {}/{}][COLLECT DONE] {} -> validation".format(
            index, len(selected), case_id), flush=True)
        rc = subprocess.run([
            sys.executable, REPO / "scripts/validate_tk_p1_case.py",
            "--csv", csv_path, "--rotation-config", rotation_path,
            "--case", case_id, "--run", case_out,
        ]).returncode
        results[case_id] = "PASS" if rc == 0 else "VALIDATE_FAIL"
        print("[PROGRESS {}/{}][VALIDATE {}] {}".format(
            index, len(selected), results[case_id], case_id), flush=True)

    passed = sum(status == "PASS" for status in results.values())
    status = "PASS" if passed == len(selected) else "FAIL"
    document = {"status": status, "passed": passed, "total": len(selected),
                "results": {case_id: results.get(case_id, "MISSING")
                            for case_id in selected}}
    (out_root / "suite_results.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n")
    print("[SUITE DONE] PASS={}/{} results={}".format(
        passed, len(selected), " ".join(
            "{}={}".format(case_id, results.get(case_id, "MISSING"))
            for case_id in selected)), flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
