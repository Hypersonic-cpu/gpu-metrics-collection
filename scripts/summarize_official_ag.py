#!/usr/bin/env python3
"""Combine official AG native output with Nsys/DCGM summaries."""

import argparse
import csv
import json
import re
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-log", required=True)
    parser.add_argument("--kernel-csv", required=True)
    parser.add_argument("--nsys-metrics", required=True)
    parser.add_argument("--dcgm-metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    native_text = open(args.native_log, errors="replace").read()
    config = re.search(r"OFFICIAL_AG_BEGIN .*warmup=(\d+) iterations=(\d+)",
                       native_text)
    if config is None:
        raise SystemExit("native configuration line not found")
    tk_ms = [float(x) for x in re.findall(r"TK: ([0-9.]+) ms", native_text)]
    nccl_ms = [float(x) for x in re.findall(r"NCCL: ([0-9.]+) ms", native_text)]
    if len(tk_ms) != 8 or len(nccl_ms) != 8:
        raise SystemExit("expected 8 TK/NCCL timings, got {}/{}".format(
            len(tk_ms), len(nccl_ms)))

    kernel_rows = list(csv.DictReader(open(args.kernel_csv)))
    main_row = next(row for row in kernel_rows if "main_kernel" in row["Name"])
    epilogue_row = next(row for row in kernel_rows if "epilogue_kernel" in row["Name"])
    nsys_rows = list(csv.DictReader(open(args.nsys_metrics)))
    dcgm_rows = list(csv.DictReader(open(args.dcgm_metrics)))

    def metric_mean(name):
        values = [float(row[name]) for row in nsys_rows if row.get(name, "") != ""]
        return statistics.fmean(values) if values else None

    def dcgm(name, scale=1.0):
        values = [float(row["value"]) * scale for row in dcgm_rows
                  if row.get("metric") == name and row.get("value") not in ("", "N/A")]
        return ({"mean": statistics.fmean(values), "max": max(values)}
                if values else None)

    nsys = {
        "main_kernel_instances": int(main_row["Instances"]),
        "main_kernel_avg_us": float(main_row["Avg (ns)"]) / 1000,
        "main_kernel_median_us": float(main_row["Med (ns)"]) / 1000,
        "epilogue_kernel_avg_us": float(epilogue_row["Avg (ns)"]) / 1000,
        "dram_total_pct_mean": metric_mean("dram_total_pct"),
        "sm_active_pct_mean": metric_mean("sm_active_pct"),
        "gr_active_pct_mean": metric_mean("gr_active_pct"),
    }
    rx_names = ["nvlink_rx_req_pct", "nvlink_rx_rsp_pct",
                "nvlink_rx_req_proto_pct", "nvlink_rx_rsp_proto_pct"]
    tx_names = ["nvlink_tx_req_pct", "nvlink_tx_rsp_pct",
                "nvlink_tx_req_proto_pct", "nvlink_tx_rsp_proto_pct"]
    nsys["nvlink_rx_total_pct_mean"] = sum(metric_mean(x) or 0 for x in rx_names)
    nsys["nvlink_tx_total_pct_mean"] = sum(metric_mean(x) or 0 for x in tx_names)

    result = {
        "native_official": {
            "warmup_iterations": int(config.group(1)),
            "measured_iterations": int(config.group(2)),
            "tk_rank_mean_ms": statistics.fmean(tk_ms),
            "tk_rank_median_ms": statistics.median(tk_ms),
            "tk_rank_max_ms": max(tk_ms),
            "nccl_rank_mean_ms": statistics.fmean(nccl_ms),
        },
        "nsys_gpu0": nsys,
        "dcgm_all_8_gpus": {
            "dram_active": dcgm("dram_active"),
            "nvlink_rx_GBps": dcgm("nvlink_rx_bytes", 1e-9),
            "nvlink_tx_GBps": dcgm("nvlink_tx_bytes", 1e-9),
            "pcie_rx_GBps": dcgm("pcie_rx_bytes", 1e-9),
            "pcie_tx_GBps": dcgm("pcie_tx_bytes", 1e-9),
        },
    }
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
