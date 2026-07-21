#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析两件事（在容器里跑，宿主机 pip 会被 OOM kill）：
  Task1  PCIe 空载底噪(PCITX/PCIRX)的均值与分布，给出"怎么减底噪"的依据。
  Task2  采样频率 >10Hz 时"空读(0/N/A)无效行"的占比，结合 dcgmi_pure(idle) 与 dcgmi_overhead(有 workload)。

用法(容器内，仓库挂到 /work)：
  python /work/experiments/dcgmi_pure/analysis/analyze_idle_and_freq.py
输出：experiments/dcgmi_pure/analysis/*.png + 控制台统计表 + stats.txt
"""
import os, sys, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.environ.get("REPO", "/work")
PURE = os.path.join(REPO, "experiments/dcgmi_pure")
OVH  = os.path.join(REPO, "experiments/dcgmi_overhead/logs")
OUT  = os.path.join(PURE, "analysis")
os.makedirs(OUT, exist_ok=True)

# ---------- header 驱动的通用解析：兼容 9 列(pure) 与 16 字段(overhead full) ----------
def parse(path):
    """返回 list[dict(field->value)]；value 为 float 或 None(=N/A)。按出现顺序。"""
    rows, fields = [], None
    with open(path) as f:
        for ln in f:
            s = ln.rstrip("\n")
            if not s.strip():
                continue
            if s.startswith("#Entity"):
                fields = s.split()[1:]            # 列名(去掉 #Entity)
                continue
            if s.startswith("ID"):                # 单位行 "ID  MB/"
                continue
            if s.startswith("GPU"):
                if fields is None:
                    continue
                tok = s.split()
                vals = tok[2:]                    # 去掉 "GPU <id>"
                if len(vals) != len(fields):
                    continue
                d = {}
                for k, v in zip(fields, vals):
                    d[k] = None if v == "N/A" else float(v)
                d["_entity"] = tok[0] + tok[1]
                rows.append(d)
    return rows

def col(rows, name):
    return [r.get(name) for r in rows]

def valid_bytes(rows, name):
    """byte 字段的有效样本：非 None 且 >0（排除 warmup N/A 和空读 0）。"""
    return [r[name] for r in rows if r.get(name) is not None and r[name] > 0]

def summ(xs):
    xs = sorted(xs)
    n = len(xs)
    q = lambda p: xs[min(n-1, int(p*(n-1)+0.5))]
    return dict(n=n, mean=sum(xs)/n, median=st.median(xs),
                std=(st.pstdev(xs) if n > 1 else 0.0),
                mn=xs[0], mx=xs[-1], p5=q(.05), p25=q(.25), p75=q(.75), p95=q(.95), p99=q(.99))

# =================== TASK 1: PCIe 空载底噪 ===================
# idle 数据源(无 workload)。interval 单位 ms。
IDLE = [
    ("1Hz",  1000, os.path.join(PURE, "idle_pcie_check_gpu45.txt")),
    ("5Hz",   200, os.path.join(PURE, "idle_pcie_check_5hz.txt")),
    ("10Hz",  100, os.path.join(PURE, "idle_pcie_check_10hz.txt")),
]
report = []
report.append("="*72)
report.append("TASK 1  PCIe 空载底噪 (idle, 无 workload)  —— PCITX/PCIRX")
report.append("="*72)

idle_data = {}     # freq -> {tx:[MB], rx:[MB], interval_ms}
for name, iv, path in IDLE:
    if not os.path.exists(path):
        report.append(f"[跳过] 缺文件 {path}"); continue
    rows = parse(path)
    tx = [v/1e6 for v in valid_bytes(rows, "PCITX")]   # MB/样本
    rx = [v/1e6 for v in valid_bytes(rows, "PCIRX")]
    idle_data[name] = dict(tx=tx, rx=rx, iv=iv)
    for lab, xs in (("PCITX", tx), ("PCIRX", rx)):
        s = summ(xs)
        rate = s["mean"]/(iv/1000.0)                    # MB/s @该间隔
        report.append(
            f"{name:>4} {lab}: n={s['n']:3d}  mean={s['mean']:.3f}MB/样本  "
            f"med={s['median']:.3f}  std={s['std']:.3f}  "
            f"p5={s['p5']:.3f} p95={s['p95']:.3f}  min={s['mn']:.3f} max={s['mx']:.3f}  "
            f"=> rate≈{rate:.1f} MB/s")

# 合并所有 idle 有效样本做"每样本底噪"分布(原始值近似与间隔无关)
pool_tx = [v for d in idle_data.values() for v in d["tx"]]
pool_rx = [v for d in idle_data.values() for v in d["rx"]]
if pool_tx:
    stx, srx = summ(pool_tx), summ(pool_rx)
    report.append("-"*72)
    report.append(f"[合并 1/5/10Hz] PCITX 每样本: mean={stx['mean']:.3f}MB med={stx['median']:.3f} "
                  f"std={stx['std']:.3f} p5={stx['p5']:.3f} p95={stx['p95']:.3f} (n={stx['n']})")
    report.append(f"[合并 1/5/10Hz] PCIRX 每样本: mean={srx['mean']:.3f}MB med={srx['median']:.3f} "
                  f"std={srx['std']:.3f} p5={srx['p5']:.3f} p95={srx['p95']:.3f} (n={srx['n']})")
    # 10Hz 运行点上折算成带宽(采集器会 bytes/interval)
    if "10Hz" in idle_data:
        r10 = summ(idle_data["10Hz"]["tx"])["mean"]/0.1
        rr10 = summ(idle_data["10Hz"]["rx"])["mean"]/0.1
        report.append(f"[10Hz 运行点] 折算 idle 带宽 ≈ TX {r10:.1f} MB/s, RX {rr10:.1f} MB/s "
                      f"(= 每样本 MB / 0.1s) —— 这就是要减掉的常数基线")

# ---- 图1: 10Hz idle PCITX/PCIRX 直方图(分布) ----
if "10Hz" in idle_data:
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.hist(idle_data["10Hz"]["tx"], bins=25, alpha=0.6, label="PCITX", color="#4C78A8")
    ax.hist(idle_data["10Hz"]["rx"], bins=25, alpha=0.6, label="PCIRX", color="#F58518")
    mtx = summ(idle_data["10Hz"]["tx"])["mean"]; mrx = summ(idle_data["10Hz"]["rx"])["mean"]
    ax.axvline(mtx, color="#4C78A8", ls="--", lw=1.5, label=f"PCITX mean={mtx:.2f}MB")
    ax.axvline(mrx, color="#F58518", ls="--", lw=1.5, label=f"PCIRX mean={mrx:.2f}MB")
    ax.set_xlabel("bytes per sample (MB/sample) @10Hz=100ms window"); ax.set_ylabel("count")
    ax.set_title("Task1: idle PCIe noise distribution (idle GPU4, 10Hz)\n"
                 f"as bandwidth ~ TX {mtx/0.1:.0f} MB/s, RX {mrx/0.1:.0f} MB/s")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "task1_idle_hist_10hz.png"), dpi=130); plt.close(fig)

# ---- 图2: idle 时序(10Hz) 看是否平稳噪声 ----
# 读法：x=第几个采样点(10Hz，相邻两点隔0.1s)，y=该点 PCIe 底噪(MB)。idle 无 workload。
# 要看的是：点只在一条**水平均值线**附近上下抖、没有上升/下降趋势 => "平稳常数+噪声"，
# 所以扣掉 mean(那条红虚线)这一个常数就能把底噪去掉(而不用随时间建模)。
if "10Hz" in idle_data:
    fig, ax = plt.subplots(figsize=(9.5,4.4))
    tx = idle_data["10Hz"]["tx"]; rx = idle_data["10Hz"]["rx"]
    mtx = summ(tx)["mean"]; stx = summ(tx)["std"]
    ax.plot(range(len(tx)), tx, lw=1, marker=".", ms=3, label="PCITX per sample", color="#4C78A8")
    ax.plot(range(len(rx)), rx, lw=1, marker=".", ms=3, label="PCIRX per sample", color="#F58518")
    ax.axhline(mtx, color="#c00", ls="--", lw=1.6, label=f"PCITX mean = {mtx:.2f} MB  (= baseline to subtract)")
    ax.axhspan(max(0,mtx-stx), mtx+stx, color="#c00", alpha=0.08, label=f"mean +/-1 sigma (sigma={stx:.2f} MB)")
    ax.set_xlabel("sample index  (10Hz  ->  each step = 0.1s of idle time)")
    ax.set_ylabel("PCIe noise (MB per sample)")
    ax.set_title("Task1: idle PCIe noise over time (no workload)\n"
                 "flat & trendless around the mean => CONSTANT baseline + jitter; just subtract the dashed line")
    ax.legend(fontsize=7.5, ncol=2, loc="upper right"); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "task1_idle_timeseries_10hz.png"), dpi=130); plt.close(fig)

# ---- 图3: 每样本原始值 跨频率箱线(证明与间隔近似无关) ----
labels = [k for k in ("1Hz","5Hz","10Hz") if k in idle_data]
if labels:
    fig, ax = plt.subplots(figsize=(7,4.5))
    data = [idle_data[k]["tx"] for k in labels]
    ax.boxplot(data, labels=[f"{k}\n({idle_data[k]['iv']}ms)" for k in labels], showfliers=True)
    ax.set_ylabel("PCITX per sample (MB/sample)")
    ax.set_title("Task1: PCITX bytes/sample ~ independent of sampling interval\n(each non-empty sample ~= one 100ms internal window)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "task1_persample_vs_freq_box.png"), dpi=130); plt.close(fig)

# ---- 图4: 底噪 vs 真实 workload 量级(log)，直观说明"几乎不用减" ----
# 参考 cumemcpy H2D: PCIRX ~5.9e10 B/样本(1Hz) ≈ 59 GB/s
fig, ax = plt.subplots(figsize=(7,4.5))
idle_mbps = summ(idle_data["10Hz"]["rx"])["mean"]/0.1 if "10Hz" in idle_data else 20.0
bars = {"PCIe idle noise\n(this experiment)": idle_mbps,
        "H2D workload\n(cumemcpy measured)": 59000.0}
ax.bar(list(bars.keys()), list(bars.values()), color=["#9ecae1","#08519c"])
ax.set_yscale("log"); ax.set_ylabel("PCIe RX (MB/s, log)")
for i,(k,v) in enumerate(bars.items()):
    ax.text(i, v*1.15, f"{v:.0f}", ha="center", fontsize=9)
ax.set_title("Task1: idle noise vs real transfer differ by ~3 orders\n-> constant subtraction or threshold removes it entirely")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "task1_idle_vs_workload_scale.png"), dpi=130); plt.close(fig)

# =================== TASK 2: >10Hz 空读无效行占比 ===================
report.append("")
report.append("="*72)
report.append("TASK 2  采样频率 vs 空读无效行占比")
report.append("="*72)
# (标签, 间隔ms, 路径, 来源)  —— byte字段空读=0, ratio字段(DRAMA)空读=N/A
FREQ_FILES = [
    ("pure idle 5Hz",   200, os.path.join(PURE,"idle_pcie_check_5hz.txt"),   "idle"),
    ("pure idle 10Hz",  100, os.path.join(PURE,"idle_pcie_check_10hz.txt"),  "idle"),
    ("pure idle 20Hz",   50, os.path.join(PURE,"idle_pcie_check_20hz.txt"),  "idle"),
    ("pure idle 100Hz",  10, os.path.join(PURE,"idle_pcie_check_100hz.txt"), "idle"),
    ("ovh mem pass1 1Hz",   1000, os.path.join(OVH,"dmon_mem_pass1_1000_rep1.txt"), "workload"),
    ("ovh mem pass1 10Hz",   100, os.path.join(OVH,"dmon_mem_pass1_100_rep1.txt"),  "workload"),
    ("ovh mem full 10Hz",    100, os.path.join(OVH,"dmon_mem_full_100_rep1.txt"),   "workload"),
    ("ovh mem full 1000Hz",    1, os.path.join(OVH,"dmon_mem_full_1_rep1.txt"),     "workload"),
    ("ovh gemm full 1000Hz",   1, os.path.join(OVH,"dmon_gemm_full_1_rep1.txt"),    "workload"),
]
report.append(f"{'文件':<22}{'间隔ms':>7}{'总行':>6}{'PCITX=0%':>10}{'DRAMA=NA%':>11}{'有效%':>8}  来源")
plot_pts = []  # (interval_ms, valid_frac, source, label)
for lab, iv, path, src in FREQ_FILES:
    if not os.path.exists(path):
        report.append(f"{lab:<22} [缺文件]"); continue
    rows = parse(path)
    tot = len(rows)
    if tot == 0:
        report.append(f"{lab:<22} [无数据行]"); continue
    tx = col(rows, "PCITX")
    dr = col(rows, "DRAMA")
    zero_tx = sum(1 for v in tx if v == 0.0)                       # byte 空读
    na_dr   = sum(1 for v in dr if v is None)                      # ratio 空读
    valid   = sum(1 for v in tx if v is not None and v > 0)        # PCITX 真值
    vf = valid/tot
    report.append(f"{lab:<22}{iv:>7}{tot:>6}{100*zero_tx/tot:>9.1f}%{100*na_dr/tot:>10.1f}%{100*vf:>7.1f}%  {src}")
    plot_pts.append((iv, vf, src, lab))

# ---- 图5: 有效占比 vs 频率(log-x)，理论线 100ms/interval ----
fig, ax = plt.subplots(figsize=(8.5,5))
import math
for src, color, mk in (("idle","#4C78A8","o"), ("workload","#E45756","s")):
    pts = [(1000.0/iv, vf) for iv,vf,s,_ in plot_pts if s==src]
    if pts:
        pts.sort()
        ax.plot([p[0] for p in pts], [100*p[1] for p in pts], mk+"-", color=color,
                label=f"measured valid% ({src})", ms=7)
# 理论: valid% = min(1, 100ms/interval) = min(1, 10Hz/freq)
fx = [1,2,5,10,20,50,100,200,1000]
ax.plot(fx, [100*min(1.0, 10.0/f) for f in fx], "k--", lw=1, label="theory = min(1, 10Hz/freq)")
ax.axvline(10, color="gray", ls=":", lw=1); ax.text(10.5, 20, "10Hz knee", fontsize=8, color="gray")
ax.set_xscale("log"); ax.set_xlabel("sampling frequency (Hz, log)"); ax.set_ylabel("valid byte-field samples (%)")
ax.set_ylim(0,105)
ax.set_title("Task2: valid rows collapse above 10Hz -- excess rows are empty reads (byte=0 / ratio=N/A)\n"
             "internal profiling update rate pinned at ~10Hz; oversampling only inserts empty rows")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(OUT, "task2_validfrac_vs_freq.png"), dpi=130); plt.close(fig)

# 落盘统计
with open(os.path.join(OUT, "stats.txt"), "w") as f:
    f.write("\n".join(report) + "\n")
print("\n".join(report))
print("\n[图已保存到]", OUT)
for p in sorted(os.listdir(OUT)):
    if p.endswith(".png"): print("  -", p)
