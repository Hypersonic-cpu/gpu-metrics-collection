#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task3 汇总图：各采集路径"能干净交付多少 Hz"(实测)。放 dcgmi_pure/analysis/。"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

# 实测(本机 GPU6, H2D 持续负载): 有效"非空/真刷新"样本率
paths = [
    ("DCGM prof\n-d 100 (req 10Hz)", 10.0, "#4C78A8", "clean ~10Hz\n(=internal rate)"),
    ("DCGM prof\n-d 1 (req 1000Hz)",  3.2, "#E45756", "COLLAPSES to ~3Hz\n99% rows = 0"),
    ("NVML pcie\nthroughput",        24.0, "#54A24B", "clean ~24Hz (2-read)\n~50Hz single, no empties"),
    ("Nsight/CUPTI\n(profiler)",   10000.0, "#B279A2", "kHz+ but SEIZES\ncounters (excl. DCGM)"),
]
fig, ax = plt.subplots(figsize=(9,5))
xs = range(len(paths))
bars = ax.bar(xs, [p[1] for p in paths], color=[p[2] for p in paths])
ax.set_yscale("log")
ax.set_ylabel("effective CLEAN sample rate (Hz, log)")
ax.set_xticks(list(xs)); ax.set_xticklabels([p[0] for p in paths], fontsize=9)
ax.axhline(10, color="gray", ls="--", lw=1)
ax.text(3.4, 11, "DCGM 10Hz floor", color="gray", fontsize=8, ha="right")
for i,(name,val,c,note) in enumerate(paths):
    ax.text(i, val*1.3, f"{val:g}Hz", ha="center", fontsize=9, fontweight="bold")
    ax.text(i, val*0.5 if val>5 else val*1.9, note, ha="center", fontsize=7.5, color="#333")
ax.set_title("Task3: how fast can each path deliver CLEAN PCIe throughput? (measured, GPU6 under H2D)\n"
             "DCGM profiling is pinned at 10Hz; oversampling makes it WORSE, not better")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "task3_paths_clean_rate.png"), dpi=130)
print("saved", os.path.join(OUT, "task3_paths_clean_rate.png"))
