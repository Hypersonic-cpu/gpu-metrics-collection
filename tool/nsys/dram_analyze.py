#!/usr/bin/env python3
"""dram_analyze.py —— 【后处理③·分析】从 CSV 算出结论，不画图。

```
① post_export.py          report.nsys-rep ─> post_metrics.csv + post_meta.json   纯标准库/宿主机
② post_plot.py overview   上面两个        ─> <iface>_overview.png                matplotlib/容器
③ dram_analyze.py         上面两个        ─> dram_stats.txt + dram_analysis.json numpy/容器
④ post_plot.py detail     ①③ 的产物      ─> <iface>_trace_*.png                 matplotlib/容器
```

    dram_analyze.py <路径…> [-o OUT] [--periods-for N] [--period-ref COL]

**四步里只有这一步是 adhoc 的**（所以名字还带 `dram_`，别的都叫 `post_*`）：
它的核心产出「主周期」是拿 **`dram_total` 当尺**量出来的 —— 那是**当前负载**（LLM 推理）
的性质，不是通用事实。换负载先读下面的 Notes。**后续加分析都往这里加**，画图那边不用动。

## 现在算什么

- **分位数**：每卡每条 metric 的 mean / p50 / p95 / max（+ 换算 GB/s）。
  **mean 和 p50 必须一起看**：kernel 之间的微小空隙会把 mean 拉得远低于 p50。
  本机 decode-hi 实测 mean 71.3% 而 p50 92%；**纯平台段（不含 step 间凹陷）里仍有 13%
  的采样点 <10%** —— 单 kernel 中位 3.4 µs，5 µs 采样正好逮得到 kernel 间的空隙。
  只报一个必然被误读成"只跑到 71%"或"一直满载"。
  接口的**方向合计与双向 total**（`nvlink_rx` = 4 路相加、`nvlink_total` = 收发均值…）
  由 `post_io.derive_family()` 派生，口径与 ④ 画出来的三条线**逐点同源**。
- **主周期**：自相关，尺 = `--period-ref`（默认 `dram_total`）。本机 decode-hi 测得
  **519.2 µs**（r=0.841），decode-mid 539.1 µs、decode-lo 524.2 µs，和 nsys-ui 上量的
  522.013 µs 对得上。④ `post_plot.py detail --periods N` 直接读这个数，不自己再测一遍。
- **建议切分数**：按 4 px/采样点（标定见 `post_io.PX_PER_POINT`）反推该劈几张图。
- **断块**：采样间隔 >1 ms 的空洞（GPU Metrics overflow，判定见 `nsys-tool diagnose`）。

## Notes for future usage（换负载/换关注点之前先读）

1. **"主周期"这个概念本身是负载给的**。当前负载 = LLM 推理，decode 一步一循环，
   而 HBM 在每一步都被读满（权重过一遍），所以 `dram_total` 是**唯一稳定的尺**
   （r=0.85–0.91）。NVLink/PCIe 太突发（实测 94% 的采样点 <10%），拿它们自相关测不出循环。
2. **换成别的负载，尺就得换**：纯 collective/AllReduce 压测里 DRAM 可能一直平（测不出周期）
   而 NVLink 才有节奏 → `--period-ref nvlink_total`；训练一个 iteration 里 forward/backward
   相位差异大，自相关会锁到 iteration 而不是层。**没有周期结构的负载**（prefill、一次性拷贝）
   本来就测不出，那时 ④ 用 `--split auto`（按 4 px/点切）而不是 `--periods`。
3. 想加相位分割（prefill vs decode）、跨 run 对比这类分析，都加在**本文件**里；
   ④ 只读 `dram_analysis.json`，不会因此改动。**别把分析原语搬进画图脚本** ——
   那正是这套流水线拆成四步之前的样子（`detect_period` 埋在画图代码里、画图靠 grep
   统计文本拿副标题）。
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from post_io import (ANALYSIS_NAME, CSV_NAME, GAP_NS, IFACE, PX_PER_POINT,   # noqa: E402
                     STATS_NAME, as_arrays, derive_all, display_cols,
                     families_in, find_csvs, load_csv, missing_parts,
                     out_dir_for, peak_for, suggest_split, title_of, write_json)

PERIOD_REF = "dram_total"       # 默认的尺，理由见上面 Notes 第 1 条


def detect_period(np, t, v, min_r=0.30):
    """自相关找主周期（返回 (ms, r)；找不到明显周期返回 None）。

    自相关在基频的整数倍上都会出峰，所以取「相关最强的那批峰里最小的那个」当基频，
    否则会把 2 倍频当周期。实测 decode-hi：基频 519.2 µs (r=0.841)，
    1043/1567/2087 µs 都是它的谐波。
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(v)
    t, v = t[ok], v[ok]
    if t.size < 100:
        return None
    dt = float(np.median(np.diff(t)))                  # ms
    if dt <= 0:
        return None
    x = v - v.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    lo = max(2, int(0.02 / dt))                        # 20 µs 起
    hi = min(len(ac) - 1, int(min(40.0, (t[-1] - t[0]) / 4) / dt))
    if hi <= lo + 2:
        return None
    peaks = [i for i in range(lo + 1, hi - 1)
             if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] >= min_r]
    if not peaks:
        return None
    best = max(ac[i] for i in peaks)
    fam = [i for i in peaks if ac[i] >= best * 0.9]    # 谐波族里取最小 = 基频
    i0 = min(fam)
    return float(i0 * dt), float(ac[i0])


def pick_period_ref(want, d):
    """尺用哪一列：`--period-ref` 优先，其次各家族的 total，最后随便一条采到的。"""
    for c in [want] + [f"{f}_total" for f in IFACE] + sorted(d):
        if c != "t" and c in d:
            return c
    return None


def gaps_of(np, t):
    """采样断块：返回 [(起 ms, 止 ms, 长度 ms), …]（间隔 > GAP_NS 算断）。"""
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return []
    d = np.diff(t)
    idx = np.nonzero(d * 1e6 > GAP_NS)[0]
    return [(float(t[i]), float(t[i + 1]), float(d[i])) for i in idx]


def analyse_one(csv_path, outdir, periods_for, period_ref):
    import numpy as np

    data = load_csv(csv_path)
    if not data["n_rows"]:
        print(f"[skip] {csv_path}: 空", file=sys.stderr)
        return 1
    src, meta, cols = data["src"], data["meta"], data["cols"]
    win, note = title_of(src, meta)

    per_gpu, used_by_fam, ref_used = [], {}, None
    for g in data["order"]:
        # 派生方向合计 / 双向 total —— 和 ④ 画的线走同一个 post_io.derive_family()
        d = as_arrays(np, data["gpus"][g], cols)
        used_by_fam = derive_all(d, cols)
        t = d["t"]
        peak = data["peak"].get(g, 0.0)
        span = float(t[-1] - t[0]) if t.size > 1 else 0.0

        ent = {"gpu": g, "n_samples": int(t.size), "span_ms": round(span, 3),
               "rate_kHz": round(t.size / span, 1) if span > 0 else 0.0,
               "hbm_peak_GBps": peak, "metrics": {}}

        for c in display_cols(cols):
            if c not in d:
                continue
            v = d[c]
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            q = np.percentile(v, [50, 95, 100])
            m = float(v.mean())
            ent["metrics"][c] = {
                "mean_pct": round(m, 3), "p50_pct": round(float(q[0]), 3),
                "p95_pct": round(float(q[1]), 3), "max_pct": round(float(q[2]), 3),
                # 低占比样本比例：mean 被拉低多少全看它（kernel 之间的空隙）
                "frac_below_10pct": round(float((v < 10).mean()), 4),
                # GB/s 用**这一列自己的**分母（post_io.peak_for）：dram_* 用报告里的 HBM 峰值，
                # pcie_* 用 63.02 GB/s/方向；nvlink_* 不定数 -> 不换算，只报百分比。
                **({"mean_GBps": round(m / 100 * pk, 1),
                    "p50_GBps": round(float(q[0]) / 100 * pk, 1),
                    "peak_GBps": pk}
                   if (pk := peak_for(c, peak)) else {}),
            }

        ref = pick_period_ref(period_ref, d)
        if ref:
            ref_used = ref
            pr = detect_period(np, t, d[ref])
            if pr:
                ent["period_us"] = round(pr[0] * 1000, 1)
                ent["period_autocorr_r"] = round(pr[1], 3)
                ent["period_ref"] = ref
        ent["gaps"] = [{"from_ms": round(a, 3), "to_ms": round(b, 3),
                        "len_ms": round(c, 3)} for a, b, c in gaps_of(np, t)]
        nsp, cap = suggest_split(t.size)
        ent["suggest_split_20in_110dpi"] = nsp
        ent["points_per_fig"] = round(cap)
        if ent.get("period_us"):
            ent[f"suggest_split_for_{periods_for:g}_periods"] = max(
                1, math.ceil(span / (ent["period_us"] / 1000 * periods_for)))
        per_gpu.append(ent)

    # 口径行：哪几路真的采到了、方向合计是怎么来的（各卡一样，报一次）
    derive_lines, warn_lines = [], []
    for f in families_in(cols):
        used = used_by_fam.get(f, {})
        dirs = list(used)
        # 只有真的相加了才写公式（dram_read/pcie_rx 本来就一条计数器，没什么可说的）
        how = "；".join(f"{f}_{dn} = " + "+".join(c[len(f) + 1:] for c in src)
                        for dn, src in used.items() if len(src) > 1)
        tot = ("+".join(dirs) if IFACE[f]["total"] == "sum"
               else "(" + "+".join(dirs) + ")/2")
        derive_lines.append(f"# 口径 {f}: {f}_total = {tot}"
                            + (f"；{how}" if how else ""))
        miss = missing_parts(f, used)
        if miss:
            warn_lines.append(f"# ⚠ {f} 这几路没采（该组没开）：{', '.join(miss)}"
                              f" -> {f}_rx/tx/total 只含采到的那几路")

    od = out_dir_for(src, outdir)
    write_json(os.path.join(od, ANALYSIS_NAME),
               {"window": win, "note": note, "group": meta.get("group"),
                "gpu_metrics_freq": meta.get("gpu_metrics_freq"),
                "period_ref": ref_used, "px_per_point": PX_PER_POINT,
                "derive": {f: used_by_fam.get(f, {}) for f in families_in(cols)},
                "gpus": per_gpu})

    # 人读的那份
    L = [f"# {win}"]
    if note:
        L.append(f"# {note}")
    if meta.get("group"):
        L.append(f"# group={meta['group']}  freq={meta.get('gpu_metrics_freq')}Hz  "
                 f"set={meta.get('gpu_metrics_set')}")
    if meta.get("resample_us"):
        L.append(f"# CSV 已按 {meta['resample_us']} µs 分桶取均值")
    L += derive_lines + warn_lines
    L.append("")
    for e in per_gpu:
        L.append(f"GPU{e['gpu']}  样本 {e['n_samples']}  跨度 {e['span_ms']:.1f} ms  "
                 f"实测采样率 {e['rate_kHz']:.1f} kHz  HBM 峰值 {e['hbm_peak_GBps']:.0f} GB/s")
        for c, m in e["metrics"].items():
            gb = f"  |  均值 {m['mean_GBps']:7.0f} GB/s" if "mean_GBps" in m else ""
            L.append(f"    {c:<20} 均值 {m['mean_pct']:6.2f}%  p50 {m['p50_pct']:6.2f}%  "
                     f"p95 {m['p95_pct']:6.2f}%  max {m['max_pct']:6.2f}%{gb}"
                     f"   (<10% 占 {m['frac_below_10pct'] * 100:.1f}%)")
        if e.get("period_us"):
            L.append(f"    主周期 {e['period_us']:.1f} µs (自相关 r={e['period_autocorr_r']}"
                     f"，尺={e['period_ref']})；一图画 {periods_for:g} 个循环 -> 劈 "
                     f"{e.get(f'suggest_split_for_{periods_for:g}_periods')} 张")
        else:
            L.append(f"    没测到明显周期（尺={ref_used}，自相关无显著峰）")
        L.append(f"    建议切分：20in@110dpi 下每图约 {e['points_per_fig']} 点 "
                 f"-> --split {e['suggest_split_20in_110dpi']}")
        if e["gaps"]:
            L.append(f"    ⚠ 采样断块 {len(e['gaps'])} 处（>1ms）："
                     + ", ".join(f"{g['from_ms']:.1f}→{g['to_ms']:.1f}ms" for g in e["gaps"][:5]))
        L.append("")

    stats_path = os.path.join(od, STATS_NAME)
    with open(stats_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"saved {stats_path}")
    print(f"saved {os.path.join(od, ANALYSIS_NAME)}")
    for w in warn_lines:
        print(w.lstrip("# "), file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="路径 = window 目录 / run 目录 / post_metrics.csv；"
               "下一步：post_plot.py detail --periods N --parts random:3")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--outdir", default=None, help="输出根目录（默认写回原目录）")
    ap.add_argument("--periods-for", type=float, default=5, metavar="N",
                    help="顺带算「一张图画 N 个循环要劈几张」（默认 5）")
    ap.add_argument("--period-ref", default=PERIOD_REF, metavar="COL",
                    help=f"拿哪一列当周期尺（默认 {PERIOD_REF}；"
                         f"没采到就退回某个 *_total，见 docstring 的 Notes）")
    a = ap.parse_args()

    csvs = [c for p in a.paths for c in find_csvs(p)]
    if not csvs:
        print(f"没找到 {CSV_NAME}（先跑 post_export.py）", file=sys.stderr)
        return 1
    if a.outdir:
        os.makedirs(a.outdir, exist_ok=True)
    rc = 0
    for c in csvs:
        if analyse_one(c, a.outdir, a.periods_for, a.period_ref) != 0:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
