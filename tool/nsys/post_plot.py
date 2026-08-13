#!/usr/bin/env python3
"""post_plot.py —— 【后处理②④·画图】CSV -> 带宽时间线 PNG。只管渲染。

```
① post_export.py          report.nsys-rep ─> post_metrics.csv + post_meta.json   纯标准库/宿主机
② post_plot.py overview   上面两个        ─> <iface>_overview.png                matplotlib/容器
③ dram_analyze.py         上面两个        ─> dram_stats.txt + dram_analysis.json numpy/容器
④ post_plot.py detail     ①③ 的产物      ─> <iface>_trace_*.png                 matplotlib/容器
```

**两个子命令 = 两步，别混着用**：

| | `overview`（②）| `detail`（④）|
|---|---|---|
| 回答什么 | 整个窗口**哪一段在忙、平台多高** | **循环内部长什么样**（单 kernel 的读写平台）|
| 时间轴 | 一张图画完整个窗口 | 劈成 N 段、每段一个 PNG（配 `--parts` 只渲染几段）|
| 每个点 | 默认压成 `--bins 1200` 个桶的均值 | 默认**逐点原样、不聚合** |
| 要不要 ③ | 不要 | `--periods N` 要（读 `dram_analysis.json` 里测好的主周期）|

⚠️ **总览图上的锯齿不是真周期**：1200 箱把 2.5 s 压成 ~2 ms/箱，比 decode 循环（~530 µs）
还长，画出来是**混叠**。要读循环结构必须 `detail --periods N`（脚注会写 `every sample`）。

**每张 GPU 一格子图**，每格三条线 **total / rx / tx**（DRAM 是 total / read / write）；
左轴一律是 `% of peak`（CSV 里 `*_pct` 列的原始单位）；分母已知的家族右轴另给 GB/s。

三条线的口径**不在本文件里**，在 `post_io.IFACE` / `derive_family()` —— 和 ③ 统计表里的数逐点同源：

| | DRAM | NVLink | PCIe |
|---|---|---|---|
| 一个方向 | read / write 各自一条 | **四路相加** `(req+rsp)×(user+proto)` | 各方向就一条计数器 |
| `total` | `read + write`（同一片 HBM、共用分母，相加有意义）| **`(rx+tx)/2`** | **`(rx+tx)/2`** |
| 右轴分母 | 报告里的 `hbm_peak_GBps` | **无**（分母不定数 → 不给 GB/s）| 63.02 GB/s/方向（Gen5×16）|

`(rx+tx)/2` 而不是相加：收发是**物理分离的双向链路**，各有各的分母，相加会出 >100% 这种
没物理意义的数（实测 nvl_write：tx 85% / rx 10%，相加 95% 会被读成"链路快满"）。

## `detail --combined`：三个接口叠一张图（interface_*.png）

上面是**逐接口各一张**；`--combined` 把 DRAM/NVLink/PCIe 叠进**同一格子图**（每卡一格）：
**颜色 = 接口，线型 = 方向**（total 实线 / 进 GPU 虚线 / 出 GPU 点线）。两个正交开关：

| 开关 | 选项 | 含义 |
|---|---|---|
| `--yaxis` | `dual`（默认）| 左轴 DRAM、右轴 NVLink+PCIe，各自 **auto-scale**（都是 %% util，不换 GB/s）|
| | `aligned` | 全部同一根 0-100%% 轴（`--ylim` 控上限）|
| `--lines` | `dir`（默认）| 每接口三条：总(实线·原生色) + 进(虚线·浅) + 出(点线·浅) |
| | `total` | 每接口只画总利用率（实线）|

画哪几个接口按 CSV 自动判（单卡纯 HBM=只 DRAM；nvlink 组=DRAM+NVLink；iface 组=三种全有）。
DRAM 缺席或只有一种 interconnect 时 dual 退成单轴。dual 的各段共用**全窗口**峰值当尺，段间可比。

## 糊不糊只取决于一件事：一个采样点分到几个横向像素

```
轴区像素 = 图宽(in) x dpi x 0.85(扣页边) x --rows
一张图放得下的点数 = 轴区像素 / 4          <- post_io.PX_PER_POINT
```

实测：52 px/点 = 一坨色块；2.05 px/点 = 能看但挤；**3.6–4.0 px/点 = 清楚**。
20in@110dpi ⇒ 每图约 470 点；200 kHz 下就是每图约 2.3 ms。
`detail --split auto` 就是按这个算劈几张；`--periods N` 按 ③ 测的主周期算。

## 两个口径，别混着读（子图标题两个都打）

| 读法 | decode-hi 实测 | 什么意思 |
|---|---|---|
| p50 / 色块顶边 | 92% = 3084 GB/s | 在搬数据时的**瞬时典型值**（nsys-ui 里肉眼读到的）|
| mean | 71.3% = 2389 GB/s | **平均带宽**（总字节/总时间），把 kernel 间空隙也算进去 |

`--style area` 可切成 read/write 堆叠填充 = nsys-ui 里 DRAM Bandwidth 那行的画法
（NVIDIA 自己的 `dram.config` 写着 `type: stacked`），颜色也用它定的那两个。
"""

import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from post_io import (ANALYSIS_NAME, AXES_FRAC, CSV_NAME, GAP_NS, IFACE,      # noqa: E402
                     PX_PER_POINT, as_arrays, ascii_only, derive_all,
                     families_in, find_csvs, load_csv, out_dir_for, peak_for,
                     read_json, title_of)

# ── 每个家族画哪三条线。**这里只管长相**，口径在 post_io.IFACE ────────────────
# 三个家族排布一致：total(黑) / 收(红) / 发(蓝)。图例文字里的 "= a+b+c" 由 _labels()
# 按**实际采到的分路**现拼（组里没开 protocol 那 4 路时会标 user only，免得图被当成含协议读）。
LINE_STYLE = [("total", "#111111", 0.9, 0.95),
              ("read",  "#d62728", 0.7, 0.75), ("rx", "#d62728", 0.7, 0.75),
              ("write", "#1f77b4", 0.7, 0.75), ("tx", "#1f77b4", 0.7, 0.75)]

FAMILIES = {
    "dram":   {"title": "HBM (DRAM) bandwidth", "ylabel": "DRAM util\n(% of HBM peak)"},
    "nvlink": {"title": "NVLink bandwidth",     "ylabel": "NVLink util\n(% of link peak)"},
    "pcie":   {"title": "PCIe bandwidth",       "ylabel": "PCIe util\n(% of link peak)"},
}

# area 模式的堆叠色：取 NVIDIA 自己 sets/dram.config 里给这两条 metric 定的颜色，
# 好让静态图和 nsys-ui 里那条 DRAM Bandwidth 行看起来是一回事。
AREA_READ, AREA_WRITE = "#FFA5A5", "#98C4DD"

# detail 不给 --parts 时最多自动渲染几个文件：--periods 常算出几百段，
# 一次全画既慢又没人看。超了就停下让人挑（真要全画：--parts all）。
MAX_AUTO_FILES = 24

# ── 合并视图 `detail --combined`：三个接口叠一张图（产出 interface_*.png）─────────
# 一格子图（每卡一格）里同时画 DRAM/NVLink/PCIe：**颜色 = 接口，线型 = 方向**。
# 口径仍走 post_io.derive_*（total/rx/tx 与 ③ 统计表逐点同源），这里只管长相。
#   颜色 = dataviz 技能验证过的分类色前 3 槽（白底 light 列，CVD-safe：三色两两 ΔE≥9.2）。
#   接口按 IFACE 固定顺序配色、不轮换（换机器/加接口时顺延，不重排）。
IFACE_COLOR = {"dram": "#2a78d6", "nvlink": "#eb6834", "pcie": "#1baf7a"}
IFACE_LABEL = {"dram": "DRAM", "nvlink": "NVLink", "pcie": "PCIe"}
# 每个接口的（进 GPU 方向, 出 GPU 方向）—— DRAM 的读=进/写=出是类比（HBM 在片内，
# 不是真过芯片边界），但视觉约定一致：进=虚线、出=点线、total=实线。
IFACE_DIRPAIR = {"dram": ("read", "write"), "nvlink": ("rx", "tx"), "pcie": ("rx", "tx")}
LS_TOTAL, LS_IN, LS_OUT = "-", "--", ":"
DIR_TINT = 0.55     # 方向线（进/出）把接口色往白里调这么多：total 用原生色最显眼，进/出更浅一档


def _tint(hex6, frac=DIR_TINT):
    """把 #rrggbb 往白色混 frac（0=原色，1=纯白）—— 方向线用它，比 total 浅一档。"""
    h = hex6.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(c + (255 - c) * frac) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _labels(fam, used):
    """图例文字：把这个家族**实际相加了哪几路**写进去（口径来自 post_io.derive_family）。"""
    dirs = list(IFACE[fam]["dirs"])
    tot = "+".join(dirs) if IFACE[fam]["total"] == "sum" else "(" + "+".join(dirs) + ")/2"
    out = {f"{fam}_total": f"total = {tot}"}
    for dname, src in used.items():
        parts = [c[len(fam) + 1 + len(dname) + 1:] for c in src]
        if len(src) == 1:                         # dram_read / pcie_rx：本来就一条计数器
            out[f"{fam}_{dname}"] = dname
            continue
        lab = f"{dname} = " + "+".join(parts)
        if not any(p.endswith("proto") for p in parts):
            lab += " (user only!)"                # 组里没采 protocol -> 少报约 18%
        out[f"{fam}_{dname}"] = lab
    return out


def _lines_for(fam, d, used):
    """-> [(列名, 图例, 颜色, 线宽, alpha)]，只留 CSV 里凑得出来的。"""
    lab = _labels(fam, used)
    return [(f"{fam}_{k}", lab[f"{fam}_{k}"], color, lw, alpha)
            for k, color, lw, alpha in LINE_STYLE
            if f"{fam}_{k}" in d and f"{fam}_{k}" in lab]


def _bin_agg(np, t, v, t0, t1, nbins, agg):
    """把 (t, v) 压到 nbins 个等宽桶做聚合；空桶 = NaN（画图自动断线）。

    为什么不画 min/max 包络：200 kHz 下 DRAM 利用率在 kernel 边界上是 0↔100 的方波，
    **实测 0.3 ms 桶内 max-min 中位数就有 96**——画包络等于把整张图涂满。
    `agg` 决定这条线回答哪个问题：mean=平均带宽，median=搬数据时的典型值。
    """
    ok = np.isfinite(v)
    t, v = t[ok], v[ok]
    centers = t0 + (np.arange(nbins) + 0.5) * (t1 - t0) / nbins
    if t.size == 0:
        return centers, np.full(nbins, np.nan)
    idx = np.clip(((t - t0) / (t1 - t0) * nbins).astype(int), 0, nbins - 1)

    if agg == "mean":                       # 快路径
        cnt = np.bincount(idx, minlength=nbins).astype(float)
        tot = np.bincount(idx, weights=v, minlength=nbins)
        with np.errstate(invalid="ignore", divide="ignore"):
            return centers, np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)

    order = np.argsort(idx, kind="stable")  # 分位数要桶内排序
    idx_s, v_s = idx[order], v[order]
    edges = np.searchsorted(idx_s, np.arange(nbins + 1))
    out = np.full(nbins, np.nan)
    fn = {"median": np.median, "p95": lambda a: np.percentile(a, 95),
          "max": np.max}[agg]
    for i in range(nbins):
        a, b = edges[i], edges[i + 1]
        if b > a:
            out[i] = fn(v_s[a:b])
    return centers, out


def period_from_analysis(src):
    """④ 的主周期**只从 ③ 的 dram_analysis.json 读**，不在这里重测。

    -> (周期 ms, 说明) 或 (None, 为什么没有)。"负载有没有节奏、尺该用哪一列"是分析结论，
    归 ③（adhoc 那一步）管；画图脚本自己再测一遍等于把同一个口径存两份。
    多卡时取自相关最强的那张卡。
    """
    a = read_json(src, ANALYSIS_NAME)
    if not a:
        return None, (f"没有 {ANALYSIS_NAME}：--periods 用的是 ③ 测好的主周期，"
                      f"先跑 dram_analyze.py")
    best = None
    for g in a.get("gpus", []):
        r = g.get("period_autocorr_r") or 0
        if g.get("period_us") and (best is None or r > best[1]):
            best = (g["period_us"], r, g.get("gpu"), g.get("period_ref") or a.get("period_ref"))
    if best is None:
        return None, (f"{ANALYSIS_NAME} 里这个窗口没测到明显周期（尺={a.get('period_ref')}）"
                      f"；改用 --split auto / --span-ms")
    us, r, gpu, ref = best
    return us / 1000.0, f"③ 测得主周期 {us:.1f} us (r={r:.3f}, GPU{gpu}, 尺={ref})"


def plot_one(csv_path, outdir, opt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = load_csv(csv_path)
    if not data["n_rows"]:
        print(f"[skip] {csv_path}: 空", file=sys.stderr)
        return 1
    src, meta, cols = data["src"], data["meta"], data["cols"]

    # list -> numpy（缺值 -> NaN），再按 post_io 的口径派生方向合计 / 双向 total
    gpus, used = {}, {}
    for g, d in data["gpus"].items():
        gpus[g] = as_arrays(np, d, cols)
        used = derive_all(gpus[g], cols)
    fams = [f for f in families_in(cols) if opt.metric in ("all", f)]
    if not fams:
        print(f"[skip] {csv_path}: 没有 {opt.metric} 列（这个 run 的 metric 组没采）",
              file=sys.stderr)
        return 1
    order, peak_of = data["order"], data["peak"]

    tmin = min(gpus[g]["t"][0] for g in order)
    tmax = max(gpus[g]["t"][-1] for g in order)
    # nsys 给每张卡起停采样的时刻不一样（实测差 0.5–1.3 s），多卡窗口两端各有一段只有一张卡
    # 有数据。劈段时随机段会落进去，画出半空的图。--overlap 把范围限死在各卡交集。
    if opt.overlap and len(order) > 1:
        olo = max(gpus[g]["t"][0] for g in order)
        ohi = min(gpus[g]["t"][-1] for g in order)
        if ohi > olo:
            print(f"  --overlap: {len(order)} 卡交集 {olo:.1f}-{ohi:.1f} ms "
                  f"({ohi - olo:.0f} ms / 全体 {tmax - tmin:.0f} ms)")
            tmin, tmax = olo, ohi
        else:
            print("  --overlap: 各卡采样区间无交集，忽略该开关", file=sys.stderr)
    X0, X1 = opt.xlim if opt.xlim else (tmin, tmax)
    span_all = X1 - X0
    n = len(order)

    # ── 劈几段（和画哪个接口无关，算一次）──
    nsamp = data["n_rows"] / max(1, n)
    axpx = opt.width * opt.dpi * AXES_FRAC * opt.rows   # 折行也是横向像素
    by_px = max(1, math.ceil(nsamp / (axpx / PX_PER_POINT)))   # 按 4 px/点该劈几张
    nsplit = 1
    if opt.mode == "detail":
        if opt.periods:
            per_ms, why = period_from_analysis(src)
            if per_ms is None:
                print(f"  [skip] {why}", file=sys.stderr)
                return 1
            nsplit = max(1, math.ceil(span_all / (per_ms * opt.periods)))
            print(f"  {why} -> 一张图画 {opt.periods:g} 个循环 = {per_ms * opt.periods:.3f} ms")
        elif opt.span_ms:
            nsplit = max(1, math.ceil(span_all / opt.span_ms))
        elif opt.nsplit == "auto":
            nsplit = by_px
        else:
            nsplit = int(opt.nsplit)

    per_fig = nsamp / nsplit
    got_px = axpx / max(per_fig, 1e-9)
    head = (f"  {nsplit} 文件 x {opt.rows} 段/卡 x {opt.width:g}in@{opt.dpi}dpi -> "
            f"每图 {span_all / nsplit:.3f} ms / 约 {per_fig:.0f} 个采样点")
    if opt.bins > 0:      # 画的是桶均值，"几 px 一个采样点"就没意义了，报桶宽
        print(head + f"，压到 {opt.bins} 桶/段 = {span_all / nsplit / opt.bins * 1000:.0f} us/桶"
                     f"（{axpx / opt.bins:.1f} px/桶）")
    else:
        print(head + f"，{got_px:.1f} px/点")
    if opt.bins <= 0 and got_px < 2.0:
        tip = (f"--periods {max(1, int(opt.periods * got_px / PX_PER_POINT))}"
               if opt.periods else f"--split auto（按 {PX_PER_POINT:g} px/点 = {by_px} 张）")
        print(f"  [提示] 只有 {got_px:.1f} px/点，线会糊；改用 {tip} 或加 --width/--dpi",
              file=sys.stderr)

    # 挑要渲染的段
    parts = opt.parts
    if isinstance(parts, tuple) and parts[0] == "random":
        # 只从**有效段**里随机抽：该段时间范围内每张卡都至少有半段的采样点
        # （避开 overflow 断块 / 多卡采样区间不齐造成的空段和半空段 —— 抽到那种图是空的）。
        edges = X0 + np.arange(nsplit + 1) * span_all / nsplit
        valid = np.ones(nsplit, dtype=bool)
        for g in order:
            t = gpus[g]["t"]
            cnt, _ = np.histogram(t[(t >= X0) & (t <= X1)], bins=edges)
            valid &= cnt >= max(10.0, 0.5 * cnt.sum() / nsplit)   # ≥ 半个满段
        pool = [i + 1 for i in range(nsplit) if valid[i]] or list(range(1, nsplit + 1))
        if len(pool) < nsplit:
            print(f"  有效段（各卡都有数据）{len(pool)}/{nsplit}")
        parts = set(random.sample(pool, min(parts[1], len(pool))))
        print(f"  随机抽中第 {sorted(parts)} 段（共 {nsplit} 段）")
    elif isinstance(parts, set):
        bad = [p for p in parts if not 1 <= p <= nsplit]
        if bad:
            print(f"  [warn] --parts 里 {bad} 超出 1..{nsplit}，已忽略", file=sys.stderr)
        print(f"  只画第 {sorted(p for p in parts if 1 <= p <= nsplit)} 段（共 {nsplit} 段）")
    elif nsplit > MAX_AUTO_FILES:
        print(f"  [skip] 会画 {nsplit} 段 x {len(fams)} 个接口；用 --parts random:3 / "
              f"3,29,47 / 10-14 挑几段（真要全画：--parts all）", file=sys.stderr)
        return 1
    want = parts or range(1, nsplit + 1)

    od = out_dir_for(src, outdir)
    win, note = title_of(src, meta)

    # ── --combined：三个接口叠一张图（interface_*.png），每段一个文件、不再逐接口分文件 ──
    if opt.combined:
        left_fams = [f for f in fams if f == "dram"]
        right_fams = [f for f in fams if f != "dram"]
        dual = opt.yaxis == "dual" and bool(left_fams) and bool(right_fams)

        def cols_of(fam):                        # 这个接口参与 auto-scale 的列（随 --lines）
            cols = [f"{fam}_total"]              # total 一直画 -> 一直参与定尺（DRAM total=读+写会超各方向）
            if opt.lines == "dir":
                cols += [f"{fam}_{p}" for p in IFACE_DIRPAIR[fam]]
            return cols

        def fam_max(famlist):                    # 全窗口 [X0,X1] 上这几个接口的峰 -> 各段共用一个尺
            m = 0.0
            for g in order:
                d = gpus[g]
                sel = (d["t"] >= X0) & (d["t"] <= X1)
                for fam in famlist:
                    for c in cols_of(fam):
                        if c in d:
                            v = d[c][sel]
                            v = v[np.isfinite(v)]
                            if v.size:
                                m = max(m, float(v.max()))
            return m

        lmax, rmax = fam_max(left_fams), fam_max(right_fams)
        scale = {"dual": dual, "right_fams": right_fams, "right_max": rmax,
                 "left_max": lmax if dual else max(lmax, rmax), "all_max": max(lmax, rmax),
                 # dual 想 auto-scale，但数据缺一侧退成单轴时也保持 auto（别被 --ylim peak 压回 0-100）
                 "single_auto": opt.yaxis == "dual" and not dual}
        for k in range(nsplit):
            if (k + 1) not in want:
                continue
            a = X0 + k * span_all / nsplit
            b = X0 + (k + 1) * span_all / nsplit
            stem = f"interface_trace_{k + 1:02d}of{nsplit:02d}" if nsplit > 1 else "interface_trace"
            if opt.xlim:
                stem += f"_{a:g}-{b:g}ms"
            _render_combined(plt, np, gpus, order, a, b, opt, (k + 1, nsplit),
                             win, note, os.path.join(od, stem + ".png"), fams, scale)
        return 0

    for fname in fams:
        fam = dict(FAMILIES[fname], name=fname, stat=f"{fname}_total",
                   lines=_lines_for(fname, gpus[order[0]], used.get(fname, {})))
        if not fam["lines"]:
            print(f"[skip] {csv_path}: {fname} 列凑不出要画的线", file=sys.stderr)
            continue
        for k in range(nsplit):
            if (k + 1) not in want:
                continue
            a = X0 + k * span_all / nsplit
            b = X0 + (k + 1) * span_all / nsplit
            if opt.mode == "overview":
                stem = f"{fname}_overview"
            elif nsplit > 1:
                stem = f"{fname}_trace_{k + 1:02d}of{nsplit:02d}"
            else:
                stem = f"{fname}_trace"
            if opt.xlim:
                stem += f"_{a:g}-{b:g}ms"           # 放大图别覆盖总览
            _render(plt, np, gpus, order, peak_of, a, b, opt, (k + 1, nsplit),
                    win, note, os.path.join(od, stem + ".png"), fam)
    return 0


def _render(plt, np, gpus, order, peak_of, x0, x1, opt, seg_of, win, note, out, fam):
    """画并存**一个文件**：版面 = 每张卡 opt.rows 条时间带（折行），x 轴覆盖 [x0, x1]。"""
    nrows = opt.rows
    xdiv, xunit = (1000.0, "s") if (x1 - x0) / nrows > 2000 else (1.0, "ms")
    n = len(order)
    fig, axes = plt.subplots(n * nrows, 1,
                             figsize=(opt.width, opt.rowh * n * nrows), squeeze=False)
    axes = axes[:, 0]
    binw_ms = None
    seg = (x1 - x0) / nrows
    have = {c for c, *_ in fam["lines"]}

    # 纵轴默认钉死 0–100（"离峰值多远"本身就是要看的，也让不同 run 之间可比）。
    # ⚠️ --ylim auto 必须按**实际画出来的那根线**取上限，不能按原始值：--bins 下画的是桶均值
    # （PCIe 原始尖峰 94% 但桶均值只有 2%，按原始值算等于没自适应）。
    # 所以先画、边画边记最大值，全部画完再统一 set_ylim —— 全卡全段共用一个上限才可比。
    plotted_max = 0.0

    for gi, g in enumerate(order):
        d = gpus[g]
        t_all = d["t"]
        peak = peak_of.get(g, 0.0)

        # 整段统计写在该卡第一条带的标题上：mean 和 p50 一起报，
        # 只报一个会被误读成"只跑到这么点"或"一直满载"。
        # 右轴/标题的 GB/s 分母按**这个家族的 total 列**查（post_io.peak_for）：
        # dram -> 报告里的 HBM 峰值；pcie -> 63.02 GB/s/方向；nvlink 不定数 -> 返回 0，不给 GB/s。
        full = (t_all >= x0) & (t_all <= x1)
        gb = peak_for(fam["stat"], peak)
        sub = ""
        if fam["stat"] in d:
            tot = d[fam["stat"]][full]
            tot = tot[np.isfinite(tot)]
            if tot.size:
                mean, p50 = float(tot.mean()), float(np.median(tot))
                sub = f"  --  mean {mean:.1f}%"
                if gb:
                    sub += f" = {mean / 100 * gb:.0f} GB/s"
                sub += f"   |   median {p50:.1f}%"
                if gb:
                    sub += f" = {p50 / 100 * gb:.0f} GB/s   (100% = {gb:.0f} GB/s)"

        for si in range(nrows):
            ax = axes[gi * nrows + si]
            a, b = x0 + si * seg, x0 + (si + 1) * seg
            sel = (t_all >= a) & (t_all <= b)
            t = t_all[sel]
            do_bin = opt.bins > 0 and t.size > opt.bins * 1.5
            if do_bin:
                binw_ms = seg / opt.bins

            def series(col, _t=t, _sel=sel, _a=a, _b=b, _bin=do_bin):
                v = d[col][_sel]
                if _bin:
                    return _bin_agg(np, _t, v, _a, _b, opt.bins, opt.agg)
                ys = v.copy()
                gap = np.diff(_t) * 1e6 > GAP_NS     # 采样断块处断开，别拉直线
                if gap.any():
                    ys[1:][gap] = np.nan
                return _t, ys

            if t.size:
                if opt.style == "area" and {"dram_read", "dram_write"} <= have:
                    # nsys-ui 的 DRAM Bandwidth 行就是 read+write 堆叠填充，色块顶边 = total。
                    # 不画 total 描边：9 万个点的轮廓线会把整块涂黑，nsys-ui 也没有它。
                    xs, rd = series("dram_read")
                    _, wr = series("dram_write")
                    x = xs / xdiv
                    ax.fill_between(x, 0, rd, color=AREA_READ, lw=0, label="read (bottom)")
                    ax.fill_between(x, rd, rd + wr, color=AREA_WRITE, lw=0,
                                    label="write (stacked on read)")
                    top = (rd + wr)[np.isfinite(rd + wr)]
                    if top.size:
                        plotted_max = max(plotted_max, float(top.max()))
                else:
                    for col, lab, color, lw, alpha in fam["lines"]:
                        xs, ys = series(col)
                        ax.plot(xs / xdiv, ys, color=color, lw=lw, alpha=alpha,
                                label=lab, solid_capstyle="butt")
                        fin = ys[np.isfinite(ys)]
                        if fin.size:
                            plotted_max = max(plotted_max, float(fin.max()))

            ax.set_xlim(a / xdiv, b / xdiv)
            ax.set_ylabel(fam["ylabel"], fontsize=9)
            ax.grid(alpha=0.3)
            if si == 0:
                ax.set_title(f"GPU {g}{sub}", fontsize=10, loc="left")
            if gb:                                   # 右轴：同一根数据换算成 GB/s
                sec = ax.secondary_yaxis(
                    "right", functions=(lambda p, kk=gb: p / 100 * kk,
                                        lambda q, kk=gb: q / kk * 100))
                sec.set_ylabel("GB/s", fontsize=9)

    # 统一纵轴（见上面 plotted_max 那段注释）
    if opt.ylim == "auto":
        ymax = max(1.0, min(100.0, plotted_max * 1.15)) if plotted_max else 100.0
    elif opt.ylim == "peak":
        ymax = 100.0
    else:
        ymax = float(opt.ylim)
    for ax in axes:
        ax.set_ylim(0, ymax)

    # 画的是原始点还是压过的，必须写在图上——不然没法判断眼睛读到的高度是什么。
    # 原始间隔按实测中位 dt 算，不能写死：同一个脚本要画 10 kHz(100 µs) 到 200 kHz(5 µs) 的报告。
    raw_us = None
    for g in order:
        t = gpus[g]["t"]
        if t.size > 1:
            raw_us = float(np.median(np.diff(t))) * 1000
            break
    xlab = f"t ({xunit}) since first GPU-metrics sample"
    xlab += (f"    [{opt.agg} per {binw_ms * 1000:.0f} us bin]" if binw_ms
             else f"    [every sample, {raw_us:.0f} us raw]" if raw_us
             else "    [every sample]")
    axes[-1].set_xlabel(xlab)

    k, ntot = seg_of
    title = win + (f"   [part {k}/{ntot}: {x0:.0f}-{x1:.0f} ms]" if ntot > 1 else "")
    if note:
        title += f"\n{note}"
    fig.suptitle(f"{fam['title']} -- {ascii_only(title)}", fontsize=11)

    # 取第一个真有内容的格子拿图例：劈段后某张卡在这一段可能一个采样点都没有
    # （多卡窗口里两张卡的采样区间本来就可能不齐），此时 axes[0] 是空的。
    h, l = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            break
    if h:
        if opt.style == "area":
            l = [x + "   [total = top of the stack]" if "write" in x else x for x in l]
        fig.legend(h, [ascii_only(x) for x in l], loc="lower center", ncol=len(l),
                   fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out, dpi=opt.dpi)
    plt.close(fig)
    print("saved", out)
    if opt.ylim == "peak" and 0 < plotted_max < 15:
        print(f"  [提示] {fam['name']} 最高只到 {plotted_max:.1f}%，线贴在轴上；"
              f"想看细节加 --ylim auto", file=sys.stderr)


def _combined_lines(fam, opt):
    """这个接口在 --combined 里画哪几条 -> [(列名, role, is_in, 线型, 线宽)]，颜色由接口定。

    --lines total：只画 `<fam>_total`（实线）；
    --lines dir  ：total（实线、原生色、粗）+ 进 GPU 方向（虚线、浅色）+ 出 GPU 方向（点线、浅色）
                   三条一起 —— total 是那条直连的基线，进/出拆分挂在它上面（方向列名来自 IFACE_DIRPAIR）。
    """
    tot = (f"{fam}_total", "total", None, LS_TOTAL, 1.6)
    if opt.lines == "total":
        return [tot]
    din, dout = IFACE_DIRPAIR[fam]
    return [tot, (f"{fam}_{din}", din, True, LS_IN, 0.8),
            (f"{fam}_{dout}", dout, False, LS_OUT, 0.8)]


def _render_combined(plt, np, gpus, order, x0, x1, opt, seg_of, win, note, out, fams, scale):
    """画并存**一个文件**：三个接口叠一张图，每张卡 opt.rows 条时间带（折行）。

    纵轴两种（scale['dual'] 决定，最终由数据是否既有 DRAM 又有 interconnect 落地）：
      dual   —— 左轴 DRAM、右轴 NVLink/PCIe，各自 auto-scale（都在 % util，不换 GB/s）；
      aligned—— 全部同一根 0-100% 轴（--ylim 控制上限）。
    只有 DRAM（单卡纯 HBM）或只有 interconnect 时没有第二根轴 -> 退成单轴。
    """
    from matplotlib.lines import Line2D
    nrows = opt.rows
    xdiv, xunit = (1000.0, "s") if (x1 - x0) / nrows > 2000 else (1.0, "ms")
    n = len(order)
    fig, axes = plt.subplots(n * nrows, 1,
                             figsize=(opt.width, opt.rowh * n * nrows), squeeze=False)
    axes = axes[:, 0]
    seg = (x1 - x0) / nrows
    binw_ms = None
    dual = scale["dual"]
    right_fams = set(scale["right_fams"])

    for gi, g in enumerate(order):
        d = gpus[g]
        t_all = d["t"]
        # 参照忙线（DRAM total）的整段统计写在该卡第一条带的标题上；没 DRAM 就不写。
        sub = ""
        full = (t_all >= x0) & (t_all <= x1)
        if "dram_total" in d:
            tot = d["dram_total"][full]
            tot = tot[np.isfinite(tot)]
            if tot.size:
                sub = f"  --  DRAM mean {tot.mean():.1f}% | median {float(np.median(tot)):.1f}%"

        for si in range(nrows):
            axL = axes[gi * nrows + si]
            axR = axL.twinx() if (dual and right_fams) else None
            a, b = x0 + si * seg, x0 + (si + 1) * seg
            sel = (t_all >= a) & (t_all <= b)
            t = t_all[sel]
            do_bin = opt.bins > 0 and t.size > opt.bins * 1.5
            if do_bin:
                binw_ms = seg / opt.bins

            def series(col, _t=t, _sel=sel, _a=a, _b=b, _bin=do_bin):
                v = d[col][_sel]
                if _bin:
                    return _bin_agg(np, _t, v, _a, _b, opt.bins, opt.agg)
                ys = v.copy()
                gap = np.diff(_t) * 1e6 > GAP_NS      # 采样断块处断开，别拉直线
                if gap.any():
                    ys[1:][gap] = np.nan
                return _t, ys

            if t.size:
                for fam in fams:
                    base, tinted = IFACE_COLOR[fam], _tint(IFACE_COLOR[fam])
                    ax = axR if (axR is not None and fam in right_fams) else axL
                    for col, role, _is_in, ls, lw in _combined_lines(fam, opt):
                        if col not in d:
                            continue
                        xs, ys = series(col)
                        ax.plot(xs / xdiv, ys, color=base if role == "total" else tinted,
                                lw=lw, ls=ls, alpha=0.95, solid_capstyle="butt")

            axL.set_xlim(a / xdiv, b / xdiv)
            axL.grid(alpha=0.3)                        # 只给左轴上格线，免得 twin 双层
            if si == 0:
                axL.set_title(f"GPU {g}{sub}", fontsize=10, loc="left")
            if dual:
                axL.set_ylim(0, max(1.0, scale["left_max"] * 1.15))
                axL.set_ylabel("DRAM util (%)", fontsize=9, color=IFACE_COLOR["dram"])
                if axR is not None:
                    axR.set_ylim(0, max(1.0, scale["right_max"] * 1.15))
                    axR.set_ylabel("NVLink / PCIe util (%)", fontsize=9)
            else:
                axL.set_ylabel("util (% of each peak)", fontsize=9)

    # aligned（单轴）才统一纵轴上限；dual 已经在上面按各自 auto-scale 钉死。
    # single_auto = 用户要 dual 但数据缺一侧退成单轴 -> 仍按数据 auto-scale。
    if not dual:
        if scale.get("single_auto") or opt.ylim == "auto":
            m = scale["all_max"]
            ymax = max(1.0, min(100.0, m * 1.15)) if m else 100.0
        elif opt.ylim == "peak":
            ymax = 100.0
        else:
            ymax = float(opt.ylim)
        for ax in axes:
            ax.set_ylim(0, ymax)

    # 图例走**两条正交通道**（用代理句柄拼；twin 会把句柄拆到两根轴上，直接取会漏）：
    # 彩色实线块 = 接口（total 的原生色），灰色线型键 = 方向（solid=total / dashed=进 / dotted=出）。
    # 9 条线全列会挤成 3 行、糊住 x 轴标签；拆两通道只占 1 行。
    present = [f for f in fams if any(f"{f}_total" in gpus[g] for g in order)]
    handles, labels = [], []
    for f in present:
        tag = ("  [L]" if f not in right_fams else "  [R]") if dual else ""
        handles.append(Line2D([0], [0], color=IFACE_COLOR[f], lw=2.4, ls=LS_TOTAL))
        labels.append(ascii_only(IFACE_LABEL[f] + tag))
    if opt.lines == "dir":                       # 方向靠线型，用中性灰键说明，避免和接口色打架
        for s, lab in ((LS_TOTAL, "total"), (LS_IN, "into GPU"), (LS_OUT, "out of GPU")):
            handles.append(Line2D([0], [0], color="#555555", lw=1.8, ls=s))
            labels.append(lab)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   fontsize=9, frameon=False)

    # 画的是原始点还是压过的，写在 x 轴（同 _render，原始间隔按实测中位 dt 算）。
    raw_us = None
    for g in order:
        t = gpus[g]["t"]
        if t.size > 1:
            raw_us = float(np.median(np.diff(t))) * 1000
            break
    xlab = f"t ({xunit}) since first GPU-metrics sample"
    xlab += (f"    [{opt.agg} per {binw_ms * 1000:.0f} us bin]" if binw_ms
             else f"    [every sample, {raw_us:.0f} us raw]" if raw_us else "")
    if dual and right_fams:
        xlab += "    (left axis: DRAM   right axis: NVLink/PCIe -- each auto-scaled)"
    axes[-1].set_xlabel(ascii_only(xlab))

    k, ntot = seg_of
    title = win + (f"   [part {k}/{ntot}: {x0:.0f}-{x1:.0f} ms]" if ntot > 1 else "")
    if note:
        title += f"\n{note}"
    fig.suptitle(f"Interfaces (DRAM/NVLink/PCIe) -- {ascii_only(title)}", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=opt.dpi)
    plt.close(fig)
    print("saved", out)


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("paths", nargs="+")
    common.add_argument("-o", "--outdir", default=None, help="输出根目录（默认写回原目录）")
    common.add_argument("--metric", default="all", choices=("all",) + tuple(FAMILIES),
                        help="画哪个接口（默认 all = CSV 里有的都画）；"
                             "产出 <metric>_*.png，互不覆盖")
    common.add_argument("--style", default="line", choices=("line", "area"),
                        help="line=三条线（默认）；area=read/write 堆叠填充"
                             "（nsys-ui 画法，只对 dram 有效）")
    common.add_argument("--rows", type=int, default=1, metavar="N",
                        help="单个文件内再折成 N 条时间带")
    common.add_argument("--width", type=float, default=20, help="图宽（英寸，默认 20）")
    common.add_argument("--row-height", type=float, default=2.4, dest="rowh",
                        help="每条带的高（英寸，默认 2.4）")
    common.add_argument("--dpi", type=int, default=110, help="输出 dpi（默认 110）")
    common.add_argument("--agg", default="mean", choices=("mean", "median", "p95", "max"),
                        help="--bins 时每桶怎么聚合（默认 mean）")
    common.add_argument("--xlim", default=None, metavar="A,B", help="只画 A–B 毫秒")
    common.add_argument("--overlap", action="store_true",
                        help="多卡时只画各卡采样区间的交集（两端只有一张卡的部分不画）")
    common.add_argument("--ylim", default="peak", metavar="peak|auto|MAX",
                        help="纵轴上限：peak=0-100（默认）/ auto=按实际画出来的线 / 数字")

    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="路径 = window 目录 / run 目录 / post_metrics.csv")
    sub = ap.add_subparsers(dest="mode", required=True)

    p2 = sub.add_parser("overview", parents=[common],
                        help="② 全局一张：哪一段在忙、平台多高（不需要 ③）")
    p2.add_argument("--bins", type=int, default=1200, metavar="N",
                    help="压到 N 个桶再画（默认 1200；0=逐点原样，几万点会糊成色块）")

    p4 = sub.add_parser("detail", parents=[common],
                        help="④ 逐点原样 + 劈段：循环内部长什么样（--periods 要先跑 ③）")
    p4.add_argument("--split", default="1", metavar="N|auto", dest="nsplit",
                    help=f"时间轴劈成 N 段，每段一个 PNG；auto=按 {PX_PER_POINT:g} px/点算")
    p4.add_argument("--periods", type=float, default=None, metavar="N",
                    help="一张图画 N 个循环（周期读 ③ 的 dram_analysis.json，不自己重测）")
    p4.add_argument("--span-ms", type=float, default=None, metavar="X",
                    help="一张图画 X 毫秒")
    p4.add_argument("--parts", default=None, metavar="LIST",
                    help=f"只渲染这几段：3,29,47 / 10-14 / random / random:4 / all"
                         f"（超过 {MAX_AUTO_FILES} 段时必给）")
    p4.add_argument("--bins", type=int, default=0, metavar="N",
                    help="压到 N 个桶再画（默认 0=逐点原样）")
    p4.add_argument("--combined", action="store_true",
                    help="三个接口叠一张图（DRAM/NVLink/PCIe），产出 interface_*.png；"
                         "颜色=接口、线型=方向；接口有哪几种按 CSV 自动判")
    p4.add_argument("--yaxis", default="dual", choices=("dual", "aligned"),
                    help="--combined 的纵轴：dual=左 DRAM/右 NVLink,PCIe 各自 auto-scale"
                         "（默认，都是 %% util 不换 GB/s）；aligned=全部同一 0-100%% 轴")
    p4.add_argument("--lines", default="dir", choices=("dir", "total"),
                    help="--combined 画什么线：dir=总(实线·原生色)+进(虚线·浅)+出(点线·浅)三条（默认）；"
                         "total=只画各接口总利用率（实线）")
    a = ap.parse_args()

    for name, dflt in (("periods", None), ("span_ms", None),      # overview 用不上的
                       ("nsplit", "1"), ("parts", None),
                       ("combined", False), ("yaxis", "dual"), ("lines", "dir")):
        if not hasattr(a, name):
            setattr(a, name, dflt)
    if a.nsplit != "auto":
        try:
            a.nsplit = max(1, int(a.nsplit))
        except ValueError:
            ap.error("--split 要么是正整数，要么是 auto")
    if a.xlim:
        try:
            a.xlim = tuple(float(x) for x in a.xlim.split(","))
            assert len(a.xlim) == 2
        except (ValueError, AssertionError):
            ap.error("--xlim 要写成 A,B（毫秒），例如 --xlim 100,130")
    if a.parts:
        if a.parts.strip() == "all":
            a.parts = None
            globals()["MAX_AUTO_FILES"] = float("inf")
        elif a.parts.strip().startswith("random"):
            tok = a.parts.split(":", 1)
            a.parts = ("random", int(tok[1]) if len(tok) > 1 and tok[1] else 1)
        else:
            try:
                s = set()
                for tok in a.parts.split(","):
                    tok = tok.strip()
                    if "-" in tok.lstrip("-"):
                        lo, hi = (int(x) for x in tok.split("-", 1))
                        s.update(range(lo, hi + 1))
                    elif tok:
                        s.add(int(tok))
                a.parts = s
            except ValueError:
                ap.error("--parts 要写成 3,29,47 / 10-14 / random / random:4 / all")

    csvs = [c for p in a.paths for c in find_csvs(p)]
    if not csvs:
        print(f"没找到 {CSV_NAME}（先跑 post_export.py）", file=sys.stderr)
        return 1
    if a.outdir:
        os.makedirs(a.outdir, exist_ok=True)
    rc = 0
    for c in csvs:
        if plot_one(c, a.outdir, a) != 0:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
