#!/usr/bin/env python3
"""Write concise native/Nsys/DCGM statistics for one TK profiling run."""

import argparse
import csv
import json
import re
import statistics


def floats(rows, key):
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            pass
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload-log", required=True)
    parser.add_argument("--kernel-csv", required=True)
    parser.add_argument("--nsys-metrics", required=True)
    parser.add_argument("--dcgm-metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    text = open(args.workload_log, errors="replace").read()
    native = [float(x) for x in re.findall(
        r"NATIVE_SAMPLE index=\d+ latency_ms=([0-9.]+)", text)]
    if len(native) != 10:
        raise SystemExit("expected 10 native samples, got {}".format(len(native)))

    kernel_rows = list(csv.DictReader(open(args.kernel_csv)))
    target = next((row for row in kernel_rows if "main_kernel" in row.get("Name", "")
                   or "fwd_attend_ker" in row.get("Name", "")), None)
    if target is None:
        raise SystemExit("target kernel row not found")

    nsys_rows = list(csv.DictReader(open(args.nsys_metrics)))
    dcgm_rows = list(csv.DictReader(open(args.dcgm_metrics)))
    summary = {
        "native": {
            "samples": len(native),
            "mean_ms": statistics.fmean(native),
            "median_ms": statistics.median(native),
            "min_ms": min(native),
            "max_ms": max(native),
        },
        "nsys": {
            "target_kernel_instances": int(target["Instances"]),
            "target_kernel_avg_us": float(target["Avg (ns)"]) / 1000.0,
            "target_kernel_median_us": float(target["Med (ns)"]) / 1000.0,
        },
        "dcgm": {},
    }
    for metric, scale in (("dram_active", 1.0),
                          ("nvlink_tx_bytes", 1e-9),
                          ("nvlink_rx_bytes", 1e-9),
                          ("pcie_tx_bytes", 1e-9),
                          ("pcie_rx_bytes", 1e-9)):
        values = [float(row["value"]) * scale for row in dcgm_rows
                  if row.get("metric") == metric and row.get("value") not in ("", "N/A")]
        if values:
            suffix = "_mean" if metric == "dram_active" else "_mean_GBps"
            max_suffix = "_max" if metric == "dram_active" else "_max_GBps"
            summary["dcgm"][metric + suffix] = statistics.fmean(values)
            summary["dcgm"][metric + max_suffix] = max(values)

    metric_columns = [name for name in (nsys_rows[0].keys() if nsys_rows else [])
                      if name.endswith("_pct")]
    for name in metric_columns:
        values = floats(nsys_rows, name)
        if values:
            summary["nsys"][name + "_mean"] = statistics.fmean(values)

    with open(args.output, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
