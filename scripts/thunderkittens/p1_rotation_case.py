#!/usr/bin/env python3
"""Run one CSV-selected P1 case through an independent rotation runner."""

import argparse
import importlib.util
import os
import pathlib
import sys

from p1_manifest import DEFAULT_CSV, DEFAULT_ROTATION_CONFIG, load_p1_suite
from p1_rotation_benchmark import run_a2a, run_ag, run_mha, run_rs


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_EXTENSIONS = HERE / "p1_extensions"


def import_file(name, path, prepend=()):
    old_path = list(sys.path)
    try:
        for item in reversed([str(pathlib.Path(p)) for p in prepend]):
            sys.path.insert(0, item)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=pathlib.Path, default=DEFAULT_CSV)
    parser.add_argument("--rotation-config", type=pathlib.Path,
                        default=DEFAULT_ROTATION_CONFIG)
    parser.add_argument("--case", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--rotation-copies", type=int, required=True)
    parser.add_argument("--rotation-seed-base", type=int, required=True)
    parser.add_argument("--extension-root", type=pathlib.Path,
                        default=DEFAULT_EXTENSIONS)
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.warmup != 1:
        parser.error("rotation warmup must be exactly one full pass")
    if args.iterations <= 0 or args.rotation_copies <= 0:
        parser.error("iterations and rotation copies must be positive")
    _, _, cases, rotations = load_p1_suite(
        args.csv, args.rotation_config, [args.case])
    case = cases[args.case]
    expected = rotations[args.case]["copies"]
    if args.rotation_copies != expected:
        parser.error("rotation copies {} differ from JSON {}".format(
            args.rotation_copies, expected))
    if args.rotation_seed_base != case["seed_base"]:
        parser.error("seed base differs from manifest policy")
    shape = case["execution"]
    print("P1_ROTATION_CONFIG case={} family={} target={} world={} warmup={} "
          "iterations={} copies={} seed_base={} shape={}".format(
              args.case, case["family"], case["target_kernel"],
              case["world_size"], args.warmup, args.iterations,
              args.rotation_copies, args.rotation_seed_base,
              ",".join("{}={}".format(k, v) for k, v in shape.items())),
          flush=True)
    if args.dry_run:
        return 0

    tk_root = pathlib.Path(os.environ.get(
        "TK_ROOT", pathlib.Path.home() / "Repos/ThunderKittens")).resolve()
    extension_root = args.extension_root.resolve()
    family = case["family"]

    if family == "GEMM":
        executable = extension_root / "gemm/bf16_h100_rotation.out"
        if not executable.is_file():
            raise FileNotFoundError("P1 GEMM runner is not built: {}".format(
                executable))
        sys.stdout.flush()
        os.execv(str(executable), [
            str(executable), args.case, str(shape["M"]), str(shape["N"]),
            str(shape["K"]), str(args.rotation_copies), str(args.warmup),
            str(args.iterations), "1" if args.check_correctness else "0",
        ])

    if family == "MHA":
        sys.path.insert(0, str(extension_root / "mha"))
        import torch
        import _C as extension
        run_mha(extension, torch, case, 0, args.rotation_copies,
                args.rotation_seed_base, args.warmup, args.iterations,
                args.check_correctness)
        return 0

    runner_by_family = {
        "AG": tk_root / "kernels/parallel/ag_gemm/benchmark.py",
        "RS": tk_root / "kernels/parallel/gemm_rs/benchmark.py",
        "A2A": tk_root / "kernels/parallel/all_to_all/benchmark.py",
    }
    runner = runner_by_family[family]
    prepend = [runner.parent]
    if family == "A2A":
        prepend.insert(0, extension_root / "a2a")
    benchmark = import_file(
        "tk_p1_{}_benchmark".format(family.lower()), runner, prepend)
    rank, world_size = benchmark.init_distributed_environment()
    try:
        if world_size != case["world_size"]:
            raise RuntimeError("case requires {} ranks, got {}".format(
                case["world_size"], world_size))
        print("P1_ROTATION_RUN_BEGIN case={} rank={}".format(
            args.case, rank), flush=True)
        if family == "AG":
            run_ag(benchmark, args.case, shape, 8, rank, world_size,
                   args.rotation_copies, args.rotation_seed_base, args.warmup,
                   args.iterations, args.check_correctness)
        elif family == "RS":
            run_rs(benchmark, args.case, shape, rank, world_size,
                   args.rotation_copies, args.rotation_seed_base, args.warmup,
                   args.iterations, args.check_correctness)
        else:
            run_a2a(benchmark, case, rank, world_size, args.rotation_copies,
                    args.rotation_seed_base, args.warmup, args.iterations,
                    args.check_correctness)
    finally:
        benchmark.destroy_distributed_environment()
    return 0


if __name__ == "__main__":
    sys.exit(main())
