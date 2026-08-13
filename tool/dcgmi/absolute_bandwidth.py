#!/usr/bin/env python3
"""Convert a DCGM metrics.csv into absolute HBM/NVLink bandwidth columns.

DCGM's H100 ``dram_active`` is a utilization ratio, not a byte counter.  The
HBM value written here is therefore explicitly an estimate:

    hbm_est_GBps = dram_active * HBM_PEAK_GBPS

NVLink 449 is already an aggregate rate in MiB/s; 1011/1012 are byte/s.
"""

import argparse
import csv
import json
import os


DEFAULT_HBM_PEAK_GBPS = 3352.32


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("metrics_csv", help="metrics.csv produced by tool/metrics.py parse")
    ap.add_argument("--out", default=None, help="output CSV (default: absolute_bandwidth.csv)")
    ap.add_argument("--hbm-peak-gbps", type=float, default=DEFAULT_HBM_PEAK_GBPS,
                    help="HBM peak used for the dram_active estimate")
    args = ap.parse_args()

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.metrics_csv)),
                                   "absolute_bandwidth.csv")
    samples = {}
    with open(args.metrics_csv, newline="") as f:
        for row in csv.DictReader(f):
            key = (row.get("epoch", ""), row.get("t_rel", ""), row.get("gpu", ""))
            d = samples.setdefault(key, {
                "epoch": row.get("epoch", ""),
                "t_rel": row.get("t_rel", ""),
                "gpu": row.get("gpu", ""),
            })
            value = as_float(row.get("value"))
            if value is None:
                continue
            metric = row.get("metric", "")
            if metric == "dram_active":
                d["hbm_est_GBps"] = value * args.hbm_peak_gbps
            elif metric == "nvlink_bw_total_MBps":
                # DCGM 449 is binary MiB/s, not decimal MB/s.
                d["nvlink_total_GBps"] = value * 1048576.0 / 1e9
            elif metric == "nvlink_tx_bytes":
                d["nvlink_tx_GBps"] = value / 1e9
            elif metric == "nvlink_rx_bytes":
                d["nvlink_rx_GBps"] = value / 1e9

    if not samples:
        raise SystemExit(f"no numeric bandwidth samples found in {args.metrics_csv}")

    fields = ["epoch", "t_rel", "gpu", "hbm_est_GBps", "nvlink_total_GBps",
              "nvlink_tx_GBps", "nvlink_rx_GBps"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key in sorted(samples, key=lambda x: (float(x[0] or 0), int(x[2] or -1))):
            d = samples[key]
            if "nvlink_total_GBps" not in d:
                tx = d.get("nvlink_tx_GBps")
                rx = d.get("nvlink_rx_GBps")
                if tx is not None or rx is not None:
                    d["nvlink_total_GBps"] = (tx or 0.0) + (rx or 0.0)
            w.writerow({k: (f"{d[k]:.6f}" if isinstance(d.get(k), float) else d.get(k, ""))
                        for k in fields})

    meta = {
        "source": os.path.abspath(args.metrics_csv),
        "output": os.path.abspath(out),
        "hbm_peak_GBps": args.hbm_peak_gbps,
        "hbm_method": "dram_active * hbm_peak_GBps (estimate; dram_active is utilization)",
        "nvlink_449_method": "MiB/s * 1048576 / 1e9",
        "nvlink_direction_method": "bytes/s / 1e9",
    }
    meta_path = os.path.splitext(out)[0] + ".json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"saved {out} ({len(samples)} samples)")
    print(f"saved {meta_path}")


if __name__ == "__main__":
    main()
