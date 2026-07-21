"""汇总 results/raw.tsv → 每个 workload 一张表：各 dcgmi 条件的 median kernel 时间 + 相对 baseline 的 slowdown%。

多个 rep 取 median 时间的中位数（对 GPU 时钟抖动更稳）。用法: python3 summarize.py
"""
import os, statistics, collections

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results", "raw.tsv")

def parse_kv(line):
    d = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d

# (workload, cond) -> list of dicts
rows = collections.defaultdict(list)
order = collections.OrderedDict()  # 保留出现顺序
with open(RAW) as f:
    for ln in f:
        parts = ln.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue
        wl, cond, iv, rep, result = parts[0], parts[1], parts[2], parts[3], parts[4]
        kv = parse_kv(result)
        if "ms_median" not in kv:
            continue
        rows[(wl, cond)].append(kv)
        order.setdefault(wl, [])
        if cond not in order[wl]:
            order[wl].append(cond)

PERF = {"gemm": ("tflops", "TFLOPS"), "mem": ("gbps", "GB/s"), "comm": ("gbps", "GB/s")}

for wl, conds in order.items():
    perf_key, perf_unit = PERF.get(wl, ("", ""))
    base = None
    print(f"\n=== workload: {wl} ===")
    print(f"{'condition':<14}{'ms_median':>11}{'ms_p95':>10}{'ms_std':>9}{'  '+perf_unit:>10}{'slowdown%':>11}")
    for cond in conds:
        recs = rows[(wl, cond)]
        meds = [float(r["ms_median"]) for r in recs]
        p95s = [float(r["ms_p95"]) for r in recs]
        stds = [float(r["ms_std"]) for r in recs]
        perfs = [float(r[perf_key]) for r in recs] if perf_key and perf_key in recs[0] else []
        med = statistics.median(meds)
        p95 = statistics.median(p95s)
        std = statistics.median(stds)
        perf = statistics.median(perfs) if perfs else float("nan")
        if cond == "baseline":
            base = med
        slow = (med / base - 1.0) * 100 if base else float("nan")
        print(f"{cond:<14}{med:>11.4f}{p95:>10.4f}{std:>9.4f}{perf:>10.1f}{slow:>10.2f}%")
print()
