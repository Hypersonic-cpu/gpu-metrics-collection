#!/usr/bin/env python3
"""Load the server CSV's complete P1 set without modifying TK sources."""

import csv
import pathlib

from p0_manifest import DEFAULT_ROTATION_CONFIG, load_rotation_config, require


DEFAULT_CSV = pathlib.Path.home() / "Repos/ThunderKittens-ProfilingTable-H100.csv"
FAMILY_MAP = {
    "MHA/GQA forward": "MHA",
    "Single-GPU BF16 GEMM": "GEMM",
    "TP AllGather + GEMM": "AG",
    "GEMM + ReduceScatter": "RS",
    "All-to-All proxy": "A2A",
}
TARGETS = {
    "MHA": "fwd_attend_ker",
    "GEMM": "prototype::lcf::kernel",
    "AG": "main_kernel",
    "RS": "matmul_reduce_scatter::kernel",
    "A2A": "all_to_all::kernel",
}
SEEDS = {"MHA": 0, "GEMM": 42, "AG": 0, "RS": 0, "A2A": 0}


def integer(row, name):
    value = row.get(name, "").strip()
    require(value, "{}: missing {}".format(row.get("Run ID"), name))
    number = float(value)
    require(number.is_integer(), "{}: {} is not integral".format(
        row.get("Run ID"), name))
    return int(number)


def load_p1_cases(csv_path=DEFAULT_CSV):
    csv_path = pathlib.Path(csv_path).expanduser().resolve()
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        next(stream)
        next(stream)
        next(stream)
        rows = list(csv.DictReader(stream))
    cases = {}
    for row in rows:
        if row.get("Priority") != "P1":
            continue
        case_id = row["Run ID"]
        family = FAMILY_MAP.get(row["TK kernel family"])
        require(family is not None, "{}: unsupported family".format(case_id))
        if family == "MHA":
            execution = {
                "B": integer(row, "B"),
                "q_len": integer(row, "Exec q_len"),
                "kv_len": integer(row, "Exec KV_len"),
                "logical_q_len": integer(row, "Logical q_len"),
                "logical_kv_len": integer(row, "Logical KV"),
                "q_heads": integer(row, "Local Q heads"),
                "kv_heads": integer(row, "Local KV heads"),
                "D": integer(row, "Head dim"),
                "causal": "causal" in row["Transformer stage"].lower(),
            }
        elif family == "A2A":
            execution = {
                "N": integer(row, "Exec M"),
                "H": integer(row, "Exec N"),
                "D": integer(row, "Exec K"),
                "scatter_axis": 2,
                "gather_axis": 1,
            }
        else:
            execution = {
                "M": integer(row, "Exec M"),
                "N": integer(row, "Exec N"),
                "K": integer(row, "Exec K"),
            }
        cases[case_id] = {
            "case_id": case_id,
            "priority": "P1",
            "family": family,
            "world_size": 1 if family in ("MHA", "GEMM") else 8,
            "target_kernel": TARGETS[family],
            "execution": execution,
            "seed_base": SEEDS[family],
            "warmup_rounds": 1,
            "source_path": row["Source path"],
            "exactness": row["Exactness"],
        }
    require(len(cases) == 45, "P1 case count must be 45, got {}".format(len(cases)))
    return csv_path, cases


def load_p1_suite(csv_path=DEFAULT_CSV, rotation_path=DEFAULT_ROTATION_CONFIG,
                  selected=None):
    csv_path, cases = load_p1_cases(csv_path)
    selected = list(selected or cases)
    unknown = [case_id for case_id in selected if case_id not in cases]
    require(not unknown, "unknown P1 cases: {}".format(",".join(unknown)))
    rotation_path, rotations = load_rotation_config(rotation_path, selected)
    return csv_path, rotation_path, cases, rotations
