#!/usr/bin/env python3
"""Invoke one YAML-selected case through ThunderKittens' official benchmark.run."""

import argparse
import importlib.util
import os
import pathlib
import sys

from p0_manifest import load_manifest, variant_name
from rotating_p0_benchmark import run_ag, run_moe, run_rs


def import_official(case, tk_root, variant_root):
    benchmark_path = tk_root / case["runner"]
    if not benchmark_path.is_file():
        raise FileNotFoundError("official benchmark missing: {}".format(benchmark_path))
    if case["family"] == "MoE":
        variant = variant_root / variant_name(case["execution"])
        if not variant.is_dir() or not list(variant.glob("_C*.so")):
            raise FileNotFoundError("MoE variant is not built: {}".format(variant))
        sys.path.insert(0, str(variant))
    else:
        sys.path.insert(0, str(benchmark_path.parent))
    spec = importlib.util.spec_from_file_location(
        "tk_official_{}_benchmark".format(case["family"].lower()),
        benchmark_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--group", default="p0")
    parser.add_argument("--case", required=True)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--variant-root", required=True, type=pathlib.Path)
    parser.add_argument("--rotation-copies", type=int, default=1)
    parser.add_argument("--rotation-seed-base", type=int)
    parser.add_argument("--rotate-buffers", action="store_true")
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    _, cases = load_manifest(args.manifest, args.group)
    if args.case not in cases:
        parser.error("case is not in selected group: {}".format(args.case))
    if args.warmup <= 0 or args.iterations <= 0:
        parser.error("warmup and iterations must be positive")
    if args.rotation_copies <= 0:
        parser.error("rotation copies must be positive")
    if args.rotate_buffers and args.rotation_seed_base is None:
        parser.error("rotating execution requires --rotation-seed-base")
    case = cases[args.case]
    shape = case["execution"]
    print("OFFICIAL_P0_CONFIG case={} family={} runner={} target={} world={} "
          "warmup={} iterations={} shape={}".format(
              args.case, case["family"], case["runner"],
              case["target_kernel"], case["world_size"], args.warmup,
              args.iterations, ",".join("{}={}".format(k, v)
                                        for k, v in shape.items())), flush=True)
    if args.dry_run:
        return 0

    tk_root = pathlib.Path(os.environ.get(
        "TK_ROOT", pathlib.Path.home() / "Repos/ThunderKittens")).resolve()
    benchmark = import_official(case, tk_root, args.variant_root.resolve())
    local_rank, world_size = benchmark.init_distributed_environment()
    if world_size != case["world_size"]:
        raise RuntimeError("case requires {} ranks, got {}".format(
            case["world_size"], world_size))
    common = dict(local_rank=local_rank, local_world_size=world_size,
                  num_warmup_iters=args.warmup, num_iters=args.iterations,
                  check_correctness=args.check_correctness, do_profile=False)
    print("OFFICIAL_P0_RUN_BEGIN case={} rank={}".format(
        args.case, local_rank), flush=True)
    if args.rotate_buffers and case["family"] == "AG":
        run_ag(
            benchmark, args.case, shape,
            case.get("runtime", {}).get("num_comm_sms", 8),
            local_rank, world_size, args.rotation_copies,
            args.rotation_seed_base, args.warmup, args.iterations,
            args.check_correctness)
    elif args.rotate_buffers and case["family"] == "MoE":
        run_moe(
            benchmark, args.case, shape,
            case.get("runtime", {}).get("num_comm_sms", 28),
            local_rank, world_size, args.rotation_copies,
            args.rotation_seed_base, args.warmup, args.iterations,
            args.check_correctness)
    elif args.rotate_buffers and case["family"] == "RS":
        run_rs(
            benchmark, args.case, shape, local_rank, world_size,
            args.rotation_copies, args.rotation_seed_base, args.warmup,
            args.iterations, args.check_correctness)
    elif args.rotate_buffers:
        raise RuntimeError("rotation is not implemented for family {}".format(
            case["family"]))
    elif case["family"] == "AG":
        benchmark.run(shape["M"], shape["K"], shape["N"],
                      case.get("runtime", {}).get("num_comm_sms", 8), **common)
    elif case["family"] == "RS":
        benchmark.run(shape["M"], shape["K"], shape["N"], **common)
    else:
        benchmark.run(B=shape["B"], S=shape["S"], H=shape["H"], I=shape["I"],
                      num_experts=shape["experts"], top_k=shape["top_k"],
                      num_comm_sms=case.get("runtime", {}).get("num_comm_sms", 28),
                      **common)
    benchmark.destroy_distributed_environment()
    return 0


if __name__ == "__main__":
    sys.exit(main())
