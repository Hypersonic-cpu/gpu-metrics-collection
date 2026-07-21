#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 Task1 PCIe 空载底噪的**原始 PCITX/PCIRX 数据**导出成 CSV(纯 stdlib，宿主机直接跑)。
产出两个文件(落 analysis/)：
  idle_pcie_raw.csv      —— 每个 idle 采样一行；含原始 bytes + 折算 MB / MB·s⁻¹ + 状态(warmup/empty/valid)
  idle_pcie_summary.csv  —— 每(频率×方向)一行的统计(mean/median/std/分位)，含 1/5/10Hz 合并行
用法: python3 make_idle_csv.py
"""
import os, csv, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
PURE = os.path.dirname(HERE)

# idle 文件(无 workload)。(freq_hz, interval_ms, 文件, 是否用于 Task1 底噪基线)
FILES = [
    (1,   1000, "idle_pcie_check_gpu45.txt", True),   # 含 GPU4+GPU5
    (5,    200, "idle_pcie_check_5hz.txt",   True),
    (10,   100, "idle_pcie_check_10hz.txt",  True),
    (20,    50, "idle_pcie_check_20hz.txt",  False),  # 过采(参考,不进底噪基线)
    (100,   10, "idle_pcie_check_100hz.txt", False),  # 过采
]

def parse(path):
    """header 驱动：返回按出现顺序的 list[(gpu, {field:value})]，value=float 或 None(N/A)。"""
    out, fields = [], None
    with open(path) as f:
        for ln in f:
            s = ln.rstrip("\n")
            if not s.strip():                continue
            if s.startswith("#Entity"):      fields = s.split()[1:]; continue
            if s.startswith("ID"):           continue
            if s.startswith("GPU") and fields:
                tok = s.split(); vals = tok[2:]
                if len(vals) != len(fields):  continue
                d = {k: (None if v == "N/A" else float(v)) for k, v in zip(fields, vals)}
                out.append((tok[0] + tok[1], d))
    return out

def status(v):
    if v is None:  return "warmup_NA"
    if v == 0.0:   return "empty_zero"
    return "valid"

# ---------- 原始逐样本 CSV ----------
raw_rows = []
# 收集底噪基线(1/5/10Hz)的有效 MB 值，供 summary
pool = {"TX": [], "RX": []}
per = {}   # (freq, dir) -> list[MB]

for freq, iv, fname, in_base in FILES:
    path = os.path.join(PURE, fname)
    if not os.path.exists(path):
        print("[skip] 缺", path); continue
    idx = {}   # 每张卡单独计数
    for gpu, d in parse(path):
        i = idx.get(gpu, 0); idx[gpu] = i + 1
        tx, rx = d.get("PCITX"), d.get("PCIRX")
        row = dict(
            source_file=fname, freq_hz=freq, interval_ms=iv, gpu=gpu, sample_idx=i,
            pcitx_bytes=("" if tx is None else int(tx)),
            pcirx_bytes=("" if rx is None else int(rx)),
            pcitx_MB=("" if tx is None else round(tx/1e6, 4)),
            pcirx_MB=("" if rx is None else round(rx/1e6, 4)),
            pcitx_MBps=("" if tx is None else round(tx/1e6/(iv/1000.0), 3)),
            pcirx_MBps=("" if rx is None else round(rx/1e6/(iv/1000.0), 3)),
            pcitx_status=status(tx), pcirx_status=status(rx),
            used_in_task1_baseline=in_base,
        )
        raw_rows.append(row)
        if in_base:
            if tx is not None and tx > 0: per.setdefault((freq,"TX"),[]).append(tx/1e6); pool["TX"].append(tx/1e6)
            if rx is not None and rx > 0: per.setdefault((freq,"RX"),[]).append(rx/1e6); pool["RX"].append(rx/1e6)

cols = ["source_file","freq_hz","interval_ms","gpu","sample_idx",
        "pcitx_bytes","pcirx_bytes","pcitx_MB","pcirx_MB","pcitx_MBps","pcirx_MBps",
        "pcitx_status","pcirx_status","used_in_task1_baseline"]
raw_csv = os.path.join(HERE, "idle_pcie_raw.csv")
with open(raw_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(raw_rows)

# ---------- 统计 summary CSV ----------
def stats_row(freq, iv, direction, xs):
    xs = sorted(xs); n = len(xs)
    q = lambda p: xs[min(n-1, int(p*(n-1)+0.5))] if n else ""
    return dict(freq_hz=freq, interval_ms=iv, direction=direction, n_valid=n,
                mean_MB=round(sum(xs)/n,4), median_MB=round(st.median(xs),4),
                std_MB=round(st.pstdev(xs),4) if n>1 else 0.0,
                p5_MB=round(q(.05),4), p95_MB=round(q(.95),4),
                min_MB=round(xs[0],4), max_MB=round(xs[-1],4),
                mean_MBps=round((sum(xs)/n)/(iv/1000.0),3))

summ = []
for freq, iv, fname, in_base in FILES:
    if not in_base: continue
    for direction in ("TX","RX"):
        xs = per.get((freq,direction), [])
        if xs: summ.append(stats_row(freq, iv, direction, xs))
# 合并 1/5/10Hz(按每样本 MB，间隔无关)；MB/s 用 10Hz 运行点折算
for direction in ("TX","RX"):
    xs = sorted(pool[direction]); n = len(xs)
    q = lambda p: xs[min(n-1,int(p*(n-1)+0.5))]
    summ.append(dict(freq_hz="1+5+10(pooled)", interval_ms=100, direction=direction, n_valid=n,
                     mean_MB=round(sum(xs)/n,4), median_MB=round(st.median(xs),4),
                     std_MB=round(st.pstdev(xs),4), p5_MB=round(q(.05),4), p95_MB=round(q(.95),4),
                     min_MB=round(xs[0],4), max_MB=round(xs[-1],4),
                     mean_MBps=round((sum(xs)/n)/0.1,3)))  # 折算到 10Hz=100ms

scols = ["freq_hz","interval_ms","direction","n_valid","mean_MB","median_MB","std_MB",
         "p5_MB","p95_MB","min_MB","max_MB","mean_MBps"]
summ_csv = os.path.join(HERE, "idle_pcie_summary.csv")
with open(summ_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=scols); w.writeheader(); w.writerows(summ)

print(f"[raw]     {len(raw_rows)} 行 -> {raw_csv}")
print(f"[summary] {len(summ)} 行 -> {summ_csv}")
print("\n--- summary 预览 ---")
for r in summ:
    print(f"{str(r['freq_hz']):>14} {r['direction']}  n={r['n_valid']:<4} "
          f"mean={r['mean_MB']}MB med={r['median_MB']} std={r['std_MB']} "
          f"p5={r['p5_MB']} p95={r['p95_MB']}  ~{r['mean_MBps']}MB/s")
