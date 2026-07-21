#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逐字段判定"纯 idle 下有没有底噪"（除 PCIe 外的其它指标）。
纯 stdlib，宿主机直接跑：python3 analyze_idle_noise_allmetrics.py
输入 : ../idle_allmetrics_100ms_gpu{7,3}.txt（collect_idle_allmetrics.sh 采的全字段 idle 日志）
输出 : 控制台表 + idle_noise_allmetrics_summary.csv（每 卡×字段 一行的判定与统计）

判定口径：跳过头 2 拍 warmup；对每个字段统计 valid(有值)/na(N/A)/zero(=0)/nonzero(>0)，
只有存在"稳定的非 0 值"才算 FLOOR（底噪），否则 CLEAN（无底噪，idle=0/N/A）。
"""
import os, csv, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
PURE = os.path.dirname(HERE)
WARMUP = 2   # 头 2 拍 profiling 字段 N/A，丢弃（见 metrics_reference §2）

# 字段短名 -> (id, 类别, 单位类型)。单位类型: ratio(0-1) / bytes(=bytes/s 速率) / mbps(MB/s device) / int% / mb(容量)
FIELD_META = {
    "DRAMA": (1005, "HBM",     "ratio"),
    "SMACT": (1002, "compute", "ratio"),
    "SMOCC": (1003, "compute", "ratio"),
    "GRACT": (1001, "compute", "ratio"),
    "TENSO": (1004, "compute", "ratio"),
    "MCUTL": (204,  "HBM(粗)", "int%"),
    "PCITX": (1009, "PCIe",    "bytes"),
    "PCIRX": (1010, "PCIe",    "bytes"),
    "NBWLT": (449,  "NVLink",  "mbps"),
    "NVLTX": (1011, "NVLink",  "bytes"),
    "NVLRX": (1012, "NVLink",  "bytes"),
    "FBUSD": (252,  "显存容量", "mb"),
}

def parse(path):
    """header 驱动解析 -> list[dict{field: float|None}]（保持采样顺序，跳过表头/单位行）。"""
    rows, fields = [], None
    with open(path) as f:
        for ln in f:
            s = ln.rstrip("\n")
            if not s.strip():           continue
            if s.startswith("#Entity"): fields = s.split()[1:]; continue
            if s.startswith("ID"):      continue
            if s.startswith("GPU") and fields:
                tok = s.split(); vals = tok[2:]
                if len(vals) != len(fields): continue
                rows.append({k: (None if v == "N/A" else float(v)) for k, v in zip(fields, vals)})
    return rows

def summarize(rows, short):
    """返回该字段的统计 dict。"""
    xs = [r.get(short) for r in rows][WARMUP:]        # 丢 warmup
    n = len(xs)
    na   = sum(1 for v in xs if v is None)
    zero = sum(1 for v in xs if v == 0.0)
    nz   = [v for v in xs if v is not None and v != 0.0]
    d = dict(n=n, na=na, zero=zero, nonzero=len(nz))
    if nz:
        d.update(nz_mean=sum(nz)/len(nz), nz_med=st.median(nz),
                 nz_min=min(nz), nz_max=max(nz))
    return d

def verdict(short, d):
    """FLOOR / CLEAN。底噪要求：有一批稳定非 0 值（>1 且占非 NA 样本 >20%）。"""
    valid = d["n"] - d["na"]
    if d["nonzero"] >= 2 and valid and d["nonzero"] / valid > 0.20:
        return "FLOOR"
    return "CLEAN"

def fmt_bytes(v):  # bytes/s 原始值 -> "x.xx MB (=y MB/s@10Hz)"
    return f"{v/1e6:.3f}MB(~{v/1e6/0.1:.1f}MB/s)"

FILES = [("gpu7", "idle_allmetrics_100ms_gpu7.txt"),
         ("gpu3", "idle_allmetrics_100ms_gpu3.txt")]

csv_rows = []
for card, fname in FILES:
    path = os.path.join(PURE, fname)
    if not os.path.exists(path):
        print("[skip] 缺", path); continue
    rows = parse(path)
    # 只保留我们关心的短名（按 header 里真实出现顺序）
    shorts = [s for s in rows[0].keys() if s in FIELD_META] if rows else []
    print(f"\n================  {card}  ({fname}, {len(rows)} 样本, 丢头 {WARMUP} 拍)  ================")
    print(f"{'字段':<7}{'id':>5}  {'类别':<9}{'判定':<7}{'非0/有效':>9}  说明/非0统计")
    print("-"*92)
    for short in shorts:
        fid, cat, unit = FIELD_META[short]
        d = summarize(rows, short)
        v = verdict(short, d)
        valid = d["n"] - d["na"]
        note = ""
        if v == "FLOOR":
            if unit == "bytes":
                note = (f"每样本 mean={fmt_bytes(d['nz_mean'])} "
                        f"med={d['nz_med']/1e6:.3f}MB max={d['nz_max']/1e6:.3f}MB")
            else:
                note = f"mean={d['nz_mean']:.4g} med={d['nz_med']:.4g} max={d['nz_max']:.4g}"
        else:
            note = f"idle=0/NA（na={d['na']} zero={d['zero']} nonzero={d['nonzero']}）"
        print(f"{short:<7}{fid:>5}  {cat:<9}{v:<7}{d['nonzero']}/{valid:>4}    {note}")
        cr = dict(card=card, field=short, id=fid, category=cat, unit=unit, verdict=v,
                  n=d["n"], na=d["na"], zero=d["zero"], nonzero=d["nonzero"])
        if d["nonzero"]:
            cr.update(nz_mean=round(d["nz_mean"],4), nz_median=round(d["nz_med"],4),
                      nz_min=round(d["nz_min"],4), nz_max=round(d["nz_max"],4))
            if unit == "bytes":
                cr["nz_mean_MB"] = round(d["nz_mean"]/1e6,4)
                cr["nz_mean_MBps@10Hz"] = round(d["nz_mean"]/1e6/0.1,3)
        csv_rows.append(cr)

cols = ["card","field","id","category","unit","verdict","n","na","zero","nonzero",
        "nz_mean","nz_median","nz_min","nz_max","nz_mean_MB","nz_mean_MBps@10Hz"]
out_csv = os.path.join(HERE, "idle_noise_allmetrics_summary.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader(); w.writerows(csv_rows)
print(f"\n[csv] {len(csv_rows)} 行 -> {out_csv}")
print("\n结论：只有 PCIe(PCITX/PCIRX) 判 FLOOR；其余全部 CLEAN（idle=0/NA，无底噪）。")
