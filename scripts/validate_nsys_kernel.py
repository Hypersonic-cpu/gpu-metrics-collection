#!/usr/bin/env python3
"""Validate that an exported nsys SQLite has CUDA kernels and GPU Metrics."""

import argparse
import sqlite3
import sys


def table_exists(db, name):
    return db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--roi-prefix")
    parser.add_argument("--require-metrics", action="store_true")
    args = parser.parse_args()

    db = sqlite3.connect(args.sqlite)
    required = ["CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"]
    missing = [name for name in required if not table_exists(db, name)]
    if missing:
        raise SystemExit("FAIL missing tables: {}".format(", ".join(missing)))

    kernel_count = db.execute(
        """
        SELECT COUNT(*)
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.demangledName
        WHERE s.value LIKE ?
        """,
        ("%{}%".format(args.kernel),),
    ).fetchone()[0]
    if kernel_count == 0:
        raise SystemExit("FAIL target kernel not found: {}".format(args.kernel))

    metrics_count = 0
    if table_exists(db, "GPU_METRICS"):
        metrics_count = db.execute("SELECT COUNT(*) FROM GPU_METRICS").fetchone()[0]
    if args.require_metrics and metrics_count == 0:
        raise SystemExit("FAIL GPU_METRICS is empty")

    roi_count = None
    roi_kernel_count = None
    if args.roi_prefix:
        if not table_exists(db, "NVTX_EVENTS"):
            raise SystemExit("FAIL NVTX_EVENTS table is missing")
        roi_count = db.execute(
            """
            SELECT COUNT(*) FROM NVTX_EVENTS n
            LEFT JOIN StringIds s ON s.id = n.textId
            WHERE COALESCE(n.text, s.value, '') LIKE ? AND n.end IS NOT NULL
            """,
            (args.roi_prefix + "%",),
        ).fetchone()[0]
        if roi_count == 0:
            raise SystemExit("FAIL ROI not found: {}".format(args.roi_prefix))
        roi_kernel_count = db.execute(
            """
            WITH roi AS (
              SELECT n.start, n.end FROM NVTX_EVENTS n
              LEFT JOIN StringIds rs ON rs.id = n.textId
              WHERE COALESCE(n.text, rs.value, '') LIKE ? AND n.end IS NOT NULL
            )
            SELECT COUNT(*)
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.demangledName, roi
            WHERE s.value LIKE ? AND k.start >= roi.start AND k.end <= roi.end
            """,
            (args.roi_prefix + "%", "%{}%".format(args.kernel)),
        ).fetchone()[0]
        if roi_kernel_count == 0:
            raise SystemExit("FAIL ROI contains no target kernels")

    print(
        "PASS kernel={!r} instances={} gpu_metric_samples={} roi_ranges={} "
        "roi_kernel_instances={}".format(
            args.kernel, kernel_count, metrics_count,
            "n/a" if roi_count is None else roi_count,
            "n/a" if roi_kernel_count is None else roi_kernel_count,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
