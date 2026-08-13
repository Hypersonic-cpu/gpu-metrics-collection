#!/usr/bin/env python3
"""Schedule all official ThunderKittens P0 profiles and async validation."""

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
from p0_manifest import load_manifest, variant_name  # noqa: E402


def csv_gpu_ids(value):
    items = value.split(",")
    if not items or any(not item.isdigit() for item in items):
        raise argparse.ArgumentTypeError("must be comma-separated GPU IDs, e.g. 1,2,3")
    if len(items) != len(set(items)):
        raise argparse.ArgumentTypeError("GPU IDs must not repeat")
    return value


def reap(active, results, block=False):
    reaped = False
    while active:
        ready_items = [item for item in active if item[1].poll() is not None]
        if not ready_items and block and not reaped:
            time.sleep(0.25)
            continue
        if not ready_items:
            return
        for ready in ready_items:
            case_id, process, log_path, _ = ready
            active.remove(ready)
            text = log_path.read_text(errors="replace") if log_path.exists() else ""
            lines = [line for line in text.splitlines() if line.startswith("[VALIDATE ")]
            summary = lines[-1] if lines else "validation produced no summary line"
            status = "PASS" if process.returncode == 0 else "FAIL"
            results[case_id] = status
            print("[VALIDATE END][{}] status={} log={}\n  {}".format(
                case_id, status, log_path, summary), flush=True)
        reaped = True
        return


def main(argv=None):
    default_manifest = TK_DIR / "tk_p0_profiling.yaml"
    parser = argparse.ArgumentParser(
        description="Run official TK P0 native -> Nsys -> DCGM -> kill, then validate")
    parser.add_argument("--manifest", type=pathlib.Path, default=default_manifest)
    parser.add_argument("--group", default="p0")
    parser.add_argument("--cases", help="comma-separated case IDs; default: all")
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--devices", type=csv_gpu_ids)
    parser.add_argument("--nsys-gpus", type=csv_gpu_ids)
    parser.add_argument("--nsys-duration", type=float)
    parser.add_argument("--nsys-frequency", type=int)
    parser.add_argument("--nsys-delay", type=float)
    parser.add_argument("--dcgm-duration", type=float)
    parser.add_argument("--validator-jobs", type=int, default=1)
    parser.add_argument("--skip-moe-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and print plan; never queries or uses a GPU")
    args = parser.parse_args(argv)

    manifest, cases = load_manifest(args.manifest, args.group)
    defaults = manifest["profiling_defaults"]
    devices = args.devices or str(defaults["devices"])
    nsys_gpus = args.nsys_gpus or str(defaults["nsys_gpus"])
    nsys_duration = (args.nsys_duration if args.nsys_duration is not None
                     else float(defaults["nsys_duration_s"]))
    nsys_frequency = (args.nsys_frequency if args.nsys_frequency is not None
                      else int(defaults["nsys_frequency_hz"]))
    nsys_delay = (args.nsys_delay if args.nsys_delay is not None
                  else float(defaults["nsys_start_delay_s"]))
    dcgm_duration = (args.dcgm_duration if args.dcgm_duration is not None
                     else float(defaults["dcgm_duration_s"]))
    if len(devices.split(",")) != 8:
        parser.error("P0 cases require exactly 8 application GPUs")
    if nsys_duration <= 0 or nsys_delay < 0 or dcgm_duration <= 0:
        parser.error("durations must be positive and delay must be nonnegative")
    if not 10 <= nsys_frequency <= 200000:
        parser.error("--nsys-frequency must be in [10, 200000]")
    if args.validator_jobs <= 0:
        parser.error("--validator-jobs must be positive")
    selected = (args.cases.split(",") if args.cases else list(cases))
    unknown = [case_id for case_id in selected if case_id not in cases]
    if unknown:
        parser.error("unknown case(s): {}".format(",".join(unknown)))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = (args.out or REPO / "runs" / "tk_p0_official_{}".format(stamp)).resolve()
    variant_root = (REPO / manifest["moe_variant_root"]).resolve()
    policy = manifest["iteration_policy"]
    tk_root = pathlib.Path(os.environ.get(
        "TK_ROOT", pathlib.Path.home() / "Repos/ThunderKittens")).resolve()
    try:
        actual_commit = subprocess.check_output(
            ["git", "-C", str(tk_root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        parser.error("cannot identify ThunderKittens checkout: {}".format(error))
    expected_commit = manifest.get("thunderkittens_commit")
    if actual_commit != expected_commit:
        parser.error("ThunderKittens commit mismatch: manifest={} checkout={}".format(
            expected_commit, actual_commit))
    for family, runner in sorted({c["family"]: c["runner"] for c in cases.values()}.items()):
        if not (tk_root / runner).is_file():
            parser.error("official {} runner is missing: {}".format(family, tk_root / runner))
        dirty = subprocess.run(
            ["git", "-C", str(tk_root), "diff", "--quiet", "HEAD", "--", runner])
        if dirty.returncode != 0:
            parser.error("official benchmark has local modifications: {}".format(
                tk_root / runner))
    moe_source = "kernels/parallel/moe_dispatch_gemm/moe_dispatch_gemm_h100.cu"
    if subprocess.run(["git", "-C", str(tk_root), "diff", "--quiet", "HEAD",
                       "--", moe_source]).returncode != 0:
        parser.error("official MoE CUDA source has local modifications: {}".format(
            tk_root / moe_source))
    moe_benchmark = tk_root / "kernels/parallel/moe_dispatch_gemm/benchmark.py"
    moe_text = moe_benchmark.read_text(encoding="utf-8")
    seed_statement = "torch.random.manual_seed(42 + local_rank)"
    if moe_text.count(seed_statement) != 1:
        parser.error("official MoE benchmark no longer has the accel-sim-compatible "
                     "routing seed statement: {}".format(seed_statement))

    print("[SUITE] cases={} out={} devices={} nsys_gpus={} duration={}s "
          "frequency={}Hz dcgm={}s".format(
              len(selected), out_root, devices, nsys_gpus, nsys_duration,
              nsys_frequency, dcgm_duration), flush=True)
    for index, case_id in enumerate(selected, 1):
        case = cases[case_id]
        variant = (" variant=" + variant_name(case["execution"])
                   if case["family"] == "MoE" else "")
        print("[PLAN {}/{}] {} family={} runner={} shape={}{}".format(
            index, len(selected), case_id, case["family"], case["runner"],
            case["execution"], variant), flush=True)
    if args.dry_run:
        print("[DRY RUN PASS] manifest and all official benchmark paths are valid; GPU untouched")
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "suite_config.json").write_text(json.dumps({
        "manifest": str(args.manifest.resolve()), "cases": selected,
        "devices": devices, "nsys_gpus": nsys_gpus,
        "nsys_duration_s": nsys_duration,
        "nsys_frequency_hz": nsys_frequency,
        "nsys_delay_s": nsys_delay, "dcgm_duration_s": dcgm_duration,
    }, indent=2) + "\n")
    if not args.skip_moe_build:
        print("[BUILD] ensure three MoE variants under {}".format(variant_root), flush=True)
        subprocess.run([
            sys.executable, TK_DIR / "build_p0_moe_variants.py",
            "--manifest", args.manifest.resolve(), "--group", args.group,
            "--output-root", variant_root,
        ], check=True)

    active = []
    results = {}
    collection_failures = set()
    for index, case_id in enumerate(selected, 1):
        reap(active, results)
        case_out = out_root / case_id
        env = os.environ.copy()
        env.update({
            "GROUP": args.group, "DEVICES": devices, "NSYS_GPUS": nsys_gpus,
            "NSYS_DURATION": str(nsys_duration),
            "NSYS_FREQUENCY": str(nsys_frequency),
            "NSYS_DELAY": str(nsys_delay), "DCGM_DURATION": str(dcgm_duration),
            "WARMUP_ITERS": str(policy["warmup_iters"]),
            "NATIVE_ITERS": str(policy["native_iters"]),
            "PROFILE_ITERS": str(policy["profile_iters"]),
            "MOE_VARIANT_ROOT": str(variant_root),
        })
        print("[PROGRESS {}/{}][LAUNCH] {} family={} out={}".format(
            index, len(selected), case_id, cases[case_id]["family"], case_out),
            flush=True)
        rc = subprocess.run([
            REPO / "scripts/collect_tk_p0_case.sh", args.manifest.resolve(),
            case_id, case_out,
        ], env=env).returncode
        if rc != 0:
            collection_failures.add(case_id)
            results[case_id] = "COLLECT_FAIL"
            print("[PROGRESS {}/{}][COLLECT FAIL] {} rc={}".format(
                index, len(selected), case_id, rc), flush=True)
            continue
        print("[PROGRESS {}/{}][COLLECT DONE] {} -> async validation".format(
            index, len(selected), case_id), flush=True)
        while len(active) >= args.validator_jobs:
            reap(active, results, block=True)
        validation_log = case_out / "validation.log"
        handle = validation_log.open("w")
        process = subprocess.Popen([
            sys.executable, REPO / "scripts/validate_tk_p0_case.py",
            "--manifest", args.manifest.resolve(), "--group", args.group,
            "--case", case_id, "--run", case_out,
        ], stdout=handle, stderr=subprocess.STDOUT)
        handle.close()
        active.append((case_id, process, validation_log, case_out))
        print("[VALIDATE LAUNCH][{}] pid={} log={}".format(
            case_id, process.pid, validation_log), flush=True)
    while active:
        reap(active, results, block=True)

    passed = sum(value == "PASS" for value in results.values())
    suite_status = {
        "status": "PASS" if passed == len(selected) and not collection_failures else "FAIL",
        "passed": passed,
        "total": len(selected),
        "results": {case_id: results.get(case_id, "MISSING") for case_id in selected},
    }
    (out_root / "suite_results.json").write_text(
        json.dumps(suite_status, indent=2, sort_keys=True) + "\n")
    print("[SUITE DONE] PASS={}/{} results={}".format(
        passed, len(selected), " ".join("{}={}".format(k, results.get(k, "MISSING"))
                                         for k in selected)), flush=True)
    return 0 if suite_status["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
