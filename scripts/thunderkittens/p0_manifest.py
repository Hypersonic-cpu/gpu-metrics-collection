#!/usr/bin/env python3
"""Shared validation and lookup helpers for the official P0 profiling suite."""

import json
import pathlib

import yaml


FAMILY_RUNNERS = {
    "AG": "kernels/parallel/ag_gemm/benchmark.py",
    "RS": "kernels/parallel/gemm_rs/benchmark.py",
    "MoE": "kernels/parallel/moe_dispatch_gemm/benchmark.py",
}
FAMILY_TARGETS = {
    "AG": "main_kernel",
    "RS": "matmul_reduce_scatter::kernel",
    "MoE": "dispatch_group_gemm_kernel",
}
P0_IDS = {
    "TK002", "TK005", "TK006", "TK007", "TK022", "TK025", "TK026",
    "TK034", "TK041", "TK042", "TK043", "TK048", "TK052", "TK055",
    "TK056", "TK064",
}
ROTATION_CASES = P0_IDS
DEFAULT_ROTATION_CONFIG = pathlib.Path.home() / "Repos/ThunderKittens-RotationSize.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_manifest(path, group="p0"):
    path = pathlib.Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    require(isinstance(manifest, dict), "manifest root must be a mapping")
    require(manifest.get("schema_version") == 1, "schema_version must be 1")
    groups = manifest.get("groups")
    require(isinstance(groups, dict) and group in groups,
            "manifest has no group {!r}".format(group))
    cases = groups[group]
    require(isinstance(cases, dict) and cases, "selected group is empty")
    require(set(cases) == P0_IDS, "P0 case set differs from the CSV-derived list")
    require(manifest.get("expected_case_count") == len(cases),
            "expected_case_count mismatch")
    policy = manifest.get("iteration_policy", {})
    require(policy.get("warmup_iters", 0) > 0, "warmup_iters must be positive")
    require(policy.get("native_iters", 0) > 0, "native_iters must be positive")
    require(policy.get("profile_iters", 0) > 0, "profile_iters must be positive")
    require(manifest.get("moe_routing_seed_base") == 42,
            "MoE routing seed base must match accel-sim: 42")
    for case_id, case in cases.items():
        prefix = "{}: ".format(case_id)
        require(case.get("case_id") == case_id, prefix + "case_id mismatch")
        require(case.get("priority") == "P0", prefix + "priority is not P0")
        family = case.get("family")
        require(family in FAMILY_RUNNERS, prefix + "unknown family")
        require(case.get("runner") == FAMILY_RUNNERS[family],
                prefix + "runner must be the unmodified official benchmark.py")
        require(case.get("target_kernel") == FAMILY_TARGETS[family],
                prefix + "target kernel mismatch")
        require(case.get("world_size") == 8, prefix + "world_size must be 8")
        execution = case.get("execution")
        require(isinstance(execution, dict), prefix + "execution is missing")
        if family == "AG":
            require(execution["M"] % 2048 == 0 and
                    execution["N"] % 256 == 0 and
                    execution["K"] % 256 == 0,
                    prefix + "invalid AG alignment")
        elif family == "RS":
            require(execution["M"] % 1024 == 0 and
                    execution["N"] % 256 == 0 and
                    execution["K"] % 256 == 0,
                    prefix + "invalid RS alignment")
        else:
            require({"B", "S", "H", "I", "experts", "top_k"} <= set(execution),
                    prefix + "incomplete MoE shape")
            require(execution["top_k"] == 8,
                    prefix + "MoE top_k must match accel-sim: 8")
        rotation = case.get("buffer_rotation")
        if case_id in ROTATION_CASES:
            require(isinstance(rotation, dict),
                    prefix + "buffer_rotation is required")
            require(isinstance(rotation.get("seed_base"), int),
                    prefix + "buffer rotation seed_base must be an integer")
            require(rotation.get("warmup_rounds") == 1,
                    prefix + "buffer rotation warmup_rounds must be exactly 1")
    return manifest, cases


def load_rotation_config(path, case_ids):
    path = pathlib.Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    require(isinstance(document, dict), "rotation config root must be a mapping")
    require(document.get("rule", {}).get("gpu_count") == 8,
            "rotation config must describe 8 GPUs")
    entries = document.get("cases")
    require(isinstance(entries, dict), "rotation config cases must be a mapping")
    selected = {}
    for case_id in case_ids:
        entry = entries.get(case_id)
        require(isinstance(entry, dict), "{} missing from rotation config".format(case_id))
        copies = entry.get("rotation_buffers")
        require(isinstance(copies, int) and copies > 0,
                "{} rotation_buffers must be positive".format(case_id))
        selected[case_id] = {
            "copies": copies,
            "per_gpu_mib": entry.get("per_gpu_mib"),
            "total_gib": entry.get("total_gib"),
            "meets_2gib": entry.get("meets_2gib"),
        }
    return path, selected


def variant_name(execution):
    return "h{}_i{}_topk{}_experts{}".format(
        execution["H"], execution["I"], execution["top_k"],
        execution["experts"])
