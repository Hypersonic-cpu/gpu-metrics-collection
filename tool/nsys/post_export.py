#!/usr/bin/env python3
"""post_export.py —— 【后处理①·导出】把 .nsys-rep 的 GPU Metrics 轨拉成 CSV。

```
① post_export.py          report.nsys-rep ─> post_metrics.csv + post_meta.json   纯标准库/宿主机
② post_plot.py overview   上面两个        ─> <iface>_overview.png                matplotlib/容器
③ dram_analyze.py         上面两个        ─> dram_stats.txt + dram_analysis.json numpy/容器
④ post_plot.py detail     ①③ 的产物      ─> <iface>_trace_*.png                 matplotlib/容器
```

**本脚本只管导出，不做统计、不画图**（那是 ②③④ 的事）。纯标准库，宿主机直接跑。
导出的**不只是 DRAM** —— 组里采到的每条序列（DRAM/SM/GR + NVLink 8 路 + PCIe 2 路）都进 CSV，
所以这一步和它的产物都叫 `post_*`（post-profiling），不叫 `dram_*`。

    post_export.py <路径…> [-o OUT] [--resample-us N] [--clip-window] [--reuse-sqlite]

`<路径>` = window 目录 / run 目录（递归找每个 window）/ report.nsys-rep 本身。

**每次都从 .nsys-rep 重新 `nsys export` 一份临时 sqlite**（用完删、不留在 run 目录），
即使目录里已经有 `report.sqlite` 也不用它 —— 那份可能是上一次采集/上一版报告留下的，
`.nsys-rep` 换了它不会跟着变，拿它导出等于拿旧数据出结论。
`--reuse-sqlite` 才用现成的（自己确定是新的、只想省几十秒时用）。

## 数据从哪来（口径，勿臆测）

`nsys export --type sqlite` 出来的库里：

| 表 | 用来干嘛 |
|---|---|
| `GPU_METRICS(timestamp, typeId, metricId, value)` | 逐点采样值，`value` = 整数百分比 0–100 |
| `TARGET_INFO_GPU_METRICS(typeId, metricId, metricName)` | metricId -> 'DRAM Read Bandwidth [Throughput %]' |
| `TARGET_INFO_GPU(id, uuid, memoryBandwidth)` | HBM 峰值（本机 3352.32 GB/s），换算 GB/s |
| `TARGET_INFO_SESSION_START_TIME(utcEpochNs)` | 采集起点的 UTC epoch，用来对齐 marks.txt |

- **`typeId` 的低 32 位 = nsys 侧 GPU 号**，join `TARGET_INFO_GPU.id`（实测：2 卡报告出
  `0x…00000000` / `0x…00000001` 两个 typeId，对上 `TARGET_INFO_GPU.id` 0/1，再按 uuid 查
  正好是 `run_meta.json` 里记的物理卡 1 和 3）。高位是 vmId，别拿整个 typeId 当卡号。
- **物理卡号优先信 `run_meta.json`** 的 `gpus`（物理）+ `gpus_nsys`（nsys 侧）配对；
  它是 nsys-tool 在**宿主机**写的。不用容器里的 `nvidia-smi` 反查 —— 容器内会重编号。
- **total = read + write**：两条都是 `pct_of_peak_sustained_elapsed`，共用同一个 HBM 峰值分母
  （nsys 自己的 `dram.config` 就把这俩标成 `type: stacked`）。本机实测 91261 个采样点里
  `read+write` 最大 99、**没有一个 >100**，佐证分母确实是同一个。
- 时间戳：`epoch_s = utcEpochNs/1e9 + timestamp/1e9`；CSV 里同时给相对时间和绝对 epoch，
  方便和 `marks.txt` / bench 日志对齐。
- `nsys stats` **没有** GPU Metrics 的内置 report（`--help-reports` 一条都没有），只能这样读表。
"""

import argparse
import csv
import json
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_report import export_sqlite, host_nsys        # noqa: E402
from post_io import CSV_NAME, META_NAME, out_dir_for, write_json   # noqa: E402

# metricName(TARGET_INFO_GPU_METRICS) -> CSV 里的短列名。
# 白名单：不在这里的 metric 即使采到了也不进 CSV。覆盖 sets/ 下所有组用得到的序列。
# ⚠️ 只有 `dram_*` 能用 CSV 里的 `hbm_peak_GBps` 换算 GB/s —— NVLink/PCIe 的百分比是
#    各自接口的 pct_of_peak_sustained_elapsed，分母不是 HBM 峰值，套上去会得出错值。
# Clock metrics are scalar MHz values, not percentages, and are written without
# a `_pct` suffix so the CSV unit is explicit.
METRIC_COLS = {
    "DRAM Read Bandwidth":  "dram_read",
    "DRAM Write Bandwidth": "dram_write",
    "SMs Active":           "sm_active",
    "GR Active":            "gr_active",
    # NVLink：请求/响应 × 用户/协议。pass1_core 只有 user 四路，nvlink 组是全 8 路。
    "NVLink RX Requests User Data":       "nvlink_rx_req",
    "NVLink RX Responses User Data":      "nvlink_rx_rsp",
    "NVLink TX Requests User Data":       "nvlink_tx_req",
    "NVLink TX Responses User Data":      "nvlink_tx_rsp",
    "NVLink RX Requests Protocol Data":   "nvlink_rx_req_proto",
    "NVLink RX Responses Protocol Data":  "nvlink_rx_rsp_proto",
    "NVLink TX Requests Protocol Data":   "nvlink_tx_req_proto",
    "NVLink TX Responses Protocol Data":  "nvlink_tx_rsp_proto",
    # PCIe：nsys 这两条含协议开销
    "PCIe RX Throughput":   "pcie_rx",
    "PCIe TX Throughput":   "pcie_tx",
    # NVIDIA GH100 clock metrics (already scaled by the metric-set multiplier).
    "GPC Clock Frequency":  "gpc_clock_MHz",
    "SYS Clock Frequency":  "sys_clock_MHz",
}

SCALAR_COLS = {"gpc_clock_MHz", "sys_clock_MHz"}
SCALAR_MULTIPLIERS = {
    # GPU_METRICS stores these counters as cycles/s; the GH100 metric-set
    # display multiplier is 1e-6, so apply it when materializing the CSV.
    "gpc_clock_MHz": 1.0e-6,
    "sys_clock_MHz": 1.0e-6,
}


# ---------------- 路径解析 ----------------

def find_reports(path):
    """<路径> -> [report.nsys-rep, ...]（目录则递归找）。"""
    if os.path.isfile(path) and path.endswith(".nsys-rep"):
        return [path]
    if os.path.isdir(path):
        direct = os.path.join(path, "report.nsys-rep")
        if os.path.isfile(direct):
            return [direct]
        out = []
        for root, _, files in os.walk(path):
            if "report.nsys-rep" in files:
                out.append(os.path.join(root, "report.nsys-rep"))
        return sorted(out)
    return []


# ---------------- 读 sqlite ----------------

def gpu_id_map(window_dir):
    """run_meta.json 的 gpus / gpus_nsys -> ({nsys_id: 物理卡号}, run_meta)。

    这份映射是 nsys-tool 在宿主机写的，是物理编号的唯一可信来源
    （容器内 nvidia-smi 会重编号，不能拿来反查）。"""
    try:
        with open(os.path.join(window_dir, "run_meta.json")) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return {}, {}
    phys = [x.strip() for x in str(meta.get("gpus", "")).split(",") if x.strip() != ""]
    nsys = [x.strip() for x in str(meta.get("gpus_nsys", "")).split(",") if x.strip() != ""]
    if len(phys) != len(nsys):
        return {}, meta
    return {int(n): int(p) for n, p in zip(nsys, phys)}, meta


def read_marks(window_dir):
    """marks.txt -> {label: epoch_s}。"""
    out = {}
    p = os.path.join(window_dir, "marks.txt")
    if not os.path.isfile(p):
        return out
    with open(p) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    out[parts[1]] = float(parts[0])
                except ValueError:
                    pass
    return out


def load_metrics(sq, id_map):
    """sqlite -> ([每卡的 dict], utcEpochNs)。

    每卡 dict: {gpu, gpu_nsys, uuid, peak_gbps, cols:[列名…], rows:[(ts_ns, {列: 值%}), …]}
    """
    c = sqlite3.connect(sq)
    tabs = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    if "GPU_METRICS" not in tabs:
        return [], None

    epoch_ns = None
    if "TARGET_INFO_SESSION_START_TIME" in tabs:
        row = c.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()
        epoch_ns = row[0] if row else None

    # nsys GPU id -> (uuid, HBM 峰值 GB/s)
    gpu_info = {}
    if "TARGET_INFO_GPU" in tabs:
        for gid, uuid, bw in c.execute("select id, uuid, memoryBandwidth from TARGET_INFO_GPU"):
            gpu_info[gid] = (uuid, (bw or 0) / 1e9)

    devs = []
    for (tid,) in c.execute("select distinct typeId from GPU_METRICS order by typeId"):
        nsys_id = tid & 0xFFFFFFFF                 # 低 32 位 = nsys 侧 GPU 号
        mid2col = {}
        for mid, name in c.execute(
                "select metricId, metricName from TARGET_INFO_GPU_METRICS where typeId=?", (tid,)):
            base = name.split("[")[0].strip()      # 'DRAM Read Bandwidth [Throughput %]'
            if base in METRIC_COLS:
                mid2col[mid] = METRIC_COLS[base]
        if not mid2col:
            continue

        by_ts = {}                                 # 按 timestamp 透视成宽表
        for ts, mid, val in c.execute(
                "select timestamp, metricId, value from GPU_METRICS where typeId=? "
                "order by timestamp", (tid,)):
            if mid in mid2col:
                by_ts.setdefault(ts, {})[mid2col[mid]] = val
        if not by_ts:
            continue

        uuid, peak = gpu_info.get(nsys_id, ("", 0.0))
        devs.append({
            "gpu": id_map.get(nsys_id, nsys_id),   # 物理号；查不到就退回 nsys 号
            "gpu_nsys": nsys_id,
            "uuid": uuid,
            "peak_gbps": peak,
            "cols": sorted(set(mid2col.values())),
            "rows": sorted(by_ts.items()),
        })
    devs.sort(key=lambda d: d["gpu_nsys"])
    return devs, epoch_ns


def resample(rows, cols, bin_ns):
    """按 bin_ns 分桶取均值（时间戳取桶内首个）。bin_ns<=0 就原样返回。"""
    if bin_ns <= 0:
        return rows
    out, cur_key, acc, n, t0 = [], None, {}, 0, None
    for ts, vals in rows:
        k = ts // bin_ns
        if cur_key is not None and k != cur_key:
            out.append((t0, {c: acc[c] / n for c in cols if c in acc}))
            acc, n, t0 = {}, 0, None
        cur_key = k
        if t0 is None:
            t0 = ts
        for c in cols:
            if c in vals:
                acc[c] = acc.get(c, 0.0) + vals[c]
        n += 1
    if n:
        out.append((t0, {c: acc[c] / n for c in cols if c in acc}))
    return out


# ---------------- 导出 ----------------

def cmd_export(paths, outdir, resample_us, clip_window, reuse_sqlite=False):
    reps = [r for p in paths for r in find_reports(p)]
    if not reps:
        print("没找到 report.nsys-rep", file=sys.stderr)
        return 1
    nsys_bin = None
    rc = 0
    for rep in reps:
        wd = os.path.dirname(rep)
        sq = os.path.join(wd, "report.sqlite")
        tmp = None
        # 默认**不认**现成的 report.sqlite：它可能比 .nsys-rep 旧（上一次采集/上一版报告留下的），
        # 而 sqlite 里读不出这件事 —— 每次重导才保证分析的是当前这份报告。
        fresh = reuse_sqlite and os.path.isfile(sq) and os.path.getsize(sq) > 4096
        if fresh:
            print(f"[{os.path.basename(wd)}] --reuse-sqlite: 用现成的 report.sqlite")
        else:
            if nsys_bin is None:
                nsys_bin = host_nsys()
            if not nsys_bin:
                print(f"[skip] {wd}: 要重导 sqlite，但找不到能读报告的 nsys", file=sys.stderr)
                rc = 1
                continue
            print(f"[{os.path.basename(wd)}] nsys export（每次重导，不用现成的 sqlite）…")
            tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False,
                                              dir=wd, prefix=".gm-")
            tmp.close()
            os.unlink(tmp.name)                    # nsys 要求目标不存在
            ok, err = export_sqlite(nsys_bin, rep, tmp.name)
            if not ok:
                print(f"[skip] {wd}: 导 sqlite 失败: {err}", file=sys.stderr)
                rc = 1
                continue
            sq = tmp.name
        try:
            if _export_one(wd, sq, outdir, resample_us, clip_window) != 0:
                rc = 1
        finally:
            if tmp and os.path.exists(tmp.name):
                os.unlink(tmp.name)
    return rc


def _export_one(wd, sq, outdir, resample_us, clip_window):
    id_map, run_meta = gpu_id_map(wd)
    devs, epoch_ns = load_metrics(sq, id_map)
    if not devs:
        print(f"[skip] {wd}: 报告里没有 GPU Metrics 轨（组里 gpu_metrics_set=none？）",
              file=sys.stderr)
        return 1

    marks = read_marks(wd)
    win = None                                     # 采集窗口（相对 ns），供 --clip-window
    if epoch_ns and "window_start" in marks and "window_end" in marks:
        win = (int(marks["window_start"] * 1e9) - epoch_ns,
               int(marks["window_end"] * 1e9) - epoch_ns)

    od = out_dir_for(wd, outdir)
    csv_path = os.path.join(od, CSV_NAME)

    # 所有卡的列取并集，保证 CSV 列固定
    allcols = sorted({c for d in devs for c in d["cols"]},
                     key=lambda c: list(METRIC_COLS.values()).index(c))
    has_dram = "dram_read" in allcols and "dram_write" in allcols
    val_cols = (["dram_total"] if has_dram else []) + allcols
    percent_cols = [c for c in val_cols if c not in SCALAR_COLS]
    scalar_cols = [c for c in val_cols if c in SCALAR_COLS]

    header = ["gpu", "gpu_nsys", "uuid", "t_ms", "epoch_s"]
    header += [f"{c}_pct" for c in percent_cols]
    header += scalar_cols
    if has_dram:
        header += ["dram_total_GBps", "dram_read_GBps", "dram_write_GBps", "hbm_peak_GBps"]

    t_ref = min(d["rows"][0][0] for d in devs)     # 全卡统一时间原点
    bin_ns = int(resample_us * 1000)
    gpu_meta, nrows = [], 0

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for d in devs:
            rows = d["rows"]
            if clip_window and win:
                rows = [(ts, v) for ts, v in rows if win[0] <= ts <= win[1]]
            rows = resample(rows, d["cols"], bin_ns)
            if not rows:
                continue
            peak = d["peak_gbps"]
            for ts, vals in rows:
                if has_dram:
                    vals = dict(vals)
                    vals["dram_total"] = vals.get("dram_read", 0) + vals.get("dram_write", 0)
                rec = [d["gpu"], d["gpu_nsys"], d["uuid"],
                       round((ts - t_ref) / 1e6, 6),
                       round(epoch_ns / 1e9 + ts / 1e9, 9) if epoch_ns else ""]
                for c in percent_cols:
                    v = vals.get(c, "")
                    rec.append(round(v, 3) if v != "" else "")
                for c in scalar_cols:
                    v = vals.get(c, "")
                    rec.append(round(v * SCALAR_MULTIPLIERS[c], 3) if v != "" else "")
                if has_dram:
                    rec += [round(vals.get(c, 0) / 100.0 * peak, 2)
                            for c in ("dram_total", "dram_read", "dram_write")]
                    rec.append(round(peak, 2))
                w.writerow(rec)
                nrows += 1

            span_ms = (rows[-1][0] - rows[0][0]) / 1e6
            gpu_meta.append({
                "gpu": d["gpu"], "gpu_nsys": d["gpu_nsys"], "uuid": d["uuid"],
                "hbm_peak_GBps": round(peak, 2), "n_samples": len(rows),
                "span_ms": round(span_ms, 3),
                "rate_kHz": round(len(rows) / span_ms, 1) if span_ms > 0 else 0.0,
                "t_start_ms": round((rows[0][0] - t_ref) / 1e6, 6),
                "t_end_ms": round((rows[-1][0] - t_ref) / 1e6, 6),
            })

    # post_meta.json：②③④ 要的上下文（note/组/频率/每卡口径），别让它们回去 grep 别人的输出
    meta = {
        "window": os.path.basename(wd.rstrip("/")),
        "note": (run_meta or {}).get("note", ""),
        "group": (run_meta or {}).get("group"),
        "gpu_metrics_freq": (run_meta or {}).get("gpu_metrics_freq"),
        "gpu_metrics_set": os.path.basename(str((run_meta or {}).get("gpu_metrics_set", ""))),
        "value_cols": val_cols,
        "percent_cols": percent_cols,
        "scalar_cols": scalar_cols,
        "scalar_units": {c: "MHz" for c in scalar_cols},
        "t_ref_epoch_s": round(epoch_ns / 1e9 + t_ref / 1e9, 9) if epoch_ns else None,
        "resample_us": resample_us,
        "clipped_to_window": bool(clip_window and win),
        "n_rows": nrows,
        "gpus": gpu_meta,
    }
    write_json(os.path.join(od, META_NAME), meta)

    print(f"saved {csv_path}  ({nrows} 行, {len(gpu_meta)} 卡)")
    print(f"saved {os.path.join(od, META_NAME)}")
    return 0


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="路径可以是 window 目录 / run 目录 / report.nsys-rep；"
               "下一步：post_plot.py overview（②）/ dram_analyze.py（③）")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--outdir", default=None,
                    help="输出根目录（默认写回各 window 目录；给了就落 <outdir>/<window名>/）")
    ap.add_argument("--resample-us", type=float, default=0,
                    help="按 N µs 分桶取均值再写 CSV（默认 0=原始 5 µs 逐点）")
    ap.add_argument("--clip-window", action="store_true",
                    help="只留 marks.txt 的 window_start..window_end 区间")
    ap.add_argument("--reuse-sqlite", action="store_true",
                    help="用目录里现成的 report.sqlite（默认每次从 .nsys-rep 重导，"
                         "免得报告换了 sqlite 还是旧的）")
    a = ap.parse_args()
    if a.outdir:
        os.makedirs(a.outdir, exist_ok=True)
    return cmd_export(a.paths, a.outdir, a.resample_us, a.clip_window, a.reuse_sqlite)


if __name__ == "__main__":
    sys.exit(main())
