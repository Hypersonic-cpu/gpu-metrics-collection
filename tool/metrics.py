#!/usr/bin/env python3
"""metrics.py —— profile.sh 采集后处理（parse + plot 合一，两后端自适应）。

按 run 的 backend 自动分派（读 run_meta.json 的 backend，缺则按文件名探测）：
  dcgm      解析 dcgm_raw.log；plot -> plots/trace.png
  nvswitch  解析 nvswitch.csv（本身即带 epoch 的规范数据）；plot -> plots/nvswitch_trace.png

两个子命令：
  parse  原始日志 -> metrics.csv（dcgm）+ 控制台速览。纯 stdlib，宿主机直接跑：
         python3 tool/metrics.py parse runs/<id>
  plot   -> plots/*.png（每 metric 一格、每实体一条线、叠加阶段 MARK 竖线）。
         需 matplotlib -> 在容器里跑（宿主机 pip 会 OOM），容器无中文字体故图内一律英文；
         加 --user 让输出图归当前用户（否则 png 被容器 root 占，删改要 sudo）：
         docker run --rm --user "$(id -u):$(id -g)" -e MPLCONFIGDIR=/tmp/mpl \\
             -v "$PWD":/work -w /work nvcr.io/nvidia/pytorch:25.12-py3 \\
             python tool/metrics.py plot runs/<id>
"""
import csv
import json
import os
import sys

# dmon 短名 -> 友好名（够用即可，未知短名原样保留）
SHORT2NAME = {
    "DRAMA": "dram_active", "SMACT": "sm_active", "SMOCC": "sm_occupancy",
    "GRACT": "gr_engine_active", "TENSO": "tensor_active",
    "NBWLT": "nvlink_bw_total_MBps", "NVLTX": "nvlink_tx_bytes", "NVLRX": "nvlink_rx_bytes",
    "PCITX": "pcie_tx_bytes", "PCIRX": "pcie_rx_bytes",
    "TXTPT": "pcie_tx_throughput", "RXTPT": "pcie_rx_throughput",
    "MCUTL": "mem_copy_util", "FBUSD": "fb_used_MB",
}

# plot 画哪些 metric（按此顺序，每个一格）
PLOT = ["dram_active", "nvlink_bw_total_MBps", "pcie_rx_bytes", "pcie_tx_bytes"]


def read_marks(path):
    """读 `<epoch>\\t<c1>\\t<c2>...` 文件 -> [(epoch, [c1, c2, ...])]；缺文件返回 []。"""
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                try:
                    out.append((float(p[0]), p[1:]))
                except ValueError:
                    pass
    return out


def detect_backend(rundir):
    """判断这个 run 是 dcgm 还是 nvswitch（先看 run_meta.json，缺则按文件名探测）。"""
    meta = os.path.join(rundir, "run_meta.json")
    if os.path.exists(meta):
        try:
            b = json.load(open(meta)).get("backend")
            if b:
                return b
        except (ValueError, OSError):
            pass
    if os.path.exists(os.path.join(rundir, "nvswitch.csv")):
        return "nvswitch"
    return "dcgm"


def sw_entity(level, switch, port):
    """nvswitch.csv 的 (level,switch,port) -> 实体标签：fabric / sw{N} / sw{N}p{P}。"""
    if level == "total":
        return "fabric"
    if level == "switch":
        return f"sw{switch}"
    return f"sw{switch}p{port}"


# ---------------- parse ----------------

def parse_raw(path):
    """yield (epoch, gpu_id, short, value_str)."""
    header = None
    with open(path) as f:
        for line in f:
            if "\t" not in line:
                continue
            ts_str, rest = line.split("\t", 1)
            try:
                epoch = float(ts_str)
            except ValueError:
                continue
            toks = rest.split()
            if not toks:
                continue
            if toks[0] == "#Entity":
                header = toks[1:]            # 短名列表
            elif toks[0] == "GPU" and header is not None and len(toks) >= 2:
                gpu = toks[1]
                vals = toks[2:]
                for i, short in enumerate(header):
                    if i < len(vals):
                        yield epoch, gpu, short, vals[i]
            # 其它行（ID 单位行 / 空行 / 报错）跳过


def cmd_parse(rundir):
    if detect_backend(rundir) == "nvswitch":
        return parse_nvswitch(rundir)

    raw = os.path.join(rundir, "dcgm_raw.log")
    if not os.path.exists(raw):
        print(f"ERROR: 找不到 {raw}", file=sys.stderr)
        sys.exit(1)

    marks = [(ep, cols[0]) for ep, cols in read_marks(os.path.join(rundir, "marks.txt"))]
    t0 = next((ep for ep, lab in marks if lab == "collector_start"), None)
    rows = list(parse_raw(raw))
    if t0 is None and rows:
        t0 = min(r[0] for r in rows)

    # 写 CSV
    out = os.path.join(rundir, "metrics.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "t_rel", "gpu", "short", "metric", "value"])
        for ep, gpu, short, val in rows:
            metric = SHORT2NAME.get(short, short)
            trel = round(ep - t0, 3) if t0 is not None else ""
            w.writerow([f"{ep:.3f}", trel, gpu, short, metric, val])

    # 控制台速览
    print(f"\n== {rundir} ==")
    print(f"样本行: {len(rows)}  ->  {out}")
    if marks:
        print("\nMARKS (相对 collector_start):")
        for ep, label in marks:
            trel = f"{ep - t0:+.2f}s" if t0 is not None else "?"
            print(f"  {trel:>8}  {label}")

    # 逐 (gpu,field) 统计：有效占比 + max + mean（N/A 记为缺失）
    agg = {}
    for ep, gpu, short, val in rows:
        key = (gpu, SHORT2NAME.get(short, short))
        d = agg.setdefault(key, {"n": 0, "valid": 0, "sum": 0.0, "max": None})
        d["n"] += 1
        try:
            x = float(val)
        except ValueError:
            continue
        d["valid"] += 1
        d["sum"] += x
        d["max"] = x if d["max"] is None else max(d["max"], x)

    print("\n字段速览 (valid/total, max, mean; N/A 不计入):")
    print(f"  {'gpu':>4} {'metric':<22} {'valid/total':>12} {'max':>16} {'mean':>16}")
    for (gpu, metric) in sorted(agg):
        d = agg[(gpu, metric)]
        mx = f"{d['max']:.4g}" if d["max"] is not None else "-"
        mn = f"{d['sum']/d['valid']:.4g}" if d["valid"] else "-"
        print(f"  {gpu:>4} {metric:<22} {str(d['valid'])+'/'+str(d['n']):>12} {mx:>16} {mn:>16}")


def parse_nvswitch(rundir):
    """nvswitch.csv 速览（本身即规范数据，不再另写 metrics.csv）。"""
    csvpath = os.path.join(rundir, "nvswitch.csv")
    if not os.path.exists(csvpath):
        print(f"ERROR: 找不到 {csvpath}", file=sys.stderr)
        sys.exit(1)
    rows = list(csv.DictReader(open(csvpath)))
    valcols = [c for c in (rows[0].keys() if rows else [])
               if c not in ("epoch", "dt_s", "level", "switch", "port")]

    marks = [(ep, cols[0]) for ep, cols in read_marks(os.path.join(rundir, "marks.txt"))]
    t0 = next((ep for ep, lab in marks if lab == "collector_start"), None)
    if t0 is None and rows:
        t0 = min(float(r["epoch"]) for r in rows)

    print(f"\n== {rundir} (backend=nvswitch) ==")
    print(f"样本行: {len(rows)}  ->  {csvpath}")
    if not rows:
        print("WARN: 无数据行（nvswitch_traffic 可能启动失败，见 nvswitch.err）", file=sys.stderr)
    if marks:
        print("\nMARKS (相对 collector_start):")
        for ep, label in marks:
            trel = f"{ep - t0:+.2f}s" if t0 is not None else "?"
            print(f"  {trel:>8}  {label}")

    # 逐 (entity, metric) 统计：有效占比 + max + mean
    agg = {}
    for r in rows:
        ent = sw_entity(r["level"], r["switch"], r["port"])
        for c in valcols:
            d = agg.setdefault((ent, c), {"n": 0, "valid": 0, "sum": 0.0, "max": None})
            d["n"] += 1
            try:
                x = float(r[c])
            except (ValueError, KeyError):
                continue
            d["valid"] += 1
            d["sum"] += x
            d["max"] = x if d["max"] is None else max(d["max"], x)

    print("\n速览 (valid/total, max, mean; 单位见列名):")
    print(f"  {'entity':>10} {'metric':<12} {'valid/total':>12} {'max':>14} {'mean':>14}")
    for (ent, c) in sorted(agg):
        d = agg[(ent, c)]
        mx = f"{d['max']:.4g}" if d["max"] is not None else "-"
        mn = f"{d['sum']/d['valid']:.4g}" if d["valid"] else "-"
        print(f"  {ent:>10} {c:<12} {str(d['valid'])+'/'+str(d['n']):>12} {mx:>14} {mn:>14}")
    print(f"\n提示: 出图 -> python tool/metrics.py plot {rundir}")


# ---------------- plot ----------------

def load_plot_marks(rundir):
    """marks.txt(collector/workload) + workload.log([MARK] 阶段) -> ([(t_rel,label)], t0)."""
    raw = [(ep, cols[0]) for ep, cols in read_marks(os.path.join(rundir, "marks.txt"))]
    for ep, cols in read_marks(os.path.join(rundir, "workload.log")):
        if "[MARK]" in cols and len(cols) >= 2:
            raw.append((ep, cols[1]))
    t0 = next((ep for ep, lab in raw if lab == "collector_start"), None)
    if t0 is None and raw:
        t0 = min(ep for ep, _ in raw)
    marks = [(ep - t0, lab.replace("phase_", "").replace("_begin", ""))
             for ep, lab in raw if t0 is not None and lab.startswith("phase_")]
    return marks, t0


def cmd_plot(rundir):
    if detect_backend(rundir) == "nvswitch":
        return plot_nvswitch(rundir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(os.path.join(rundir, "metrics.csv"))))
    gpus = sorted({r["gpu"] for r in rows})
    marks, _ = load_plot_marks(rundir)

    fig, axes = plt.subplots(len(PLOT), 1, figsize=(11, 2.4 * len(PLOT)), sharex=True)
    for ax, metric in zip(axes, PLOT):
        for gpu in gpus:
            xs, ys = [], []
            for r in rows:
                if r["metric"] == metric and r["gpu"] == gpu and r["t_rel"] != "":
                    try:
                        ys.append(float(r["value"]))
                        xs.append(float(r["t_rel"]))
                    except ValueError:
                        pass
            if xs:
                ax.plot(xs, ys, marker=".", ms=3, label=f"GPU{gpu}")
        for tr, lab in marks:
            ax.axvline(tr, color="grey", ls="--", lw=0.7, alpha=0.3)
            ax.text(tr, ax.get_ylim()[1], lab, rotation=90, va="top", ha="right",
                    fontsize=7, color="grey")
        ax.set_ylabel(metric, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("t_rel (s) since collector_start")
    fig.suptitle(f"GPU interface metrics trace — {os.path.basename(rundir.rstrip('/'))}")
    fig.tight_layout()
    outdir = os.path.join(rundir, "plots")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "trace.png")
    fig.savefig(out, dpi=110)
    print("saved", out)


def plot_nvswitch(rundir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(os.path.join(rundir, "nvswitch.csv"))))
    if not rows:
        print("no data to plot（nvswitch.csv 空，见 nvswitch.err）", file=sys.stderr)
        return
    valcols = [c for c in rows[0].keys()
               if c not in ("epoch", "dt_s", "level", "switch", "port")]
    marks, t0 = load_plot_marks(rundir)
    if t0 is None:
        t0 = min(float(r["epoch"]) for r in rows)
    ents = sorted({sw_entity(r["level"], r["switch"], r["port"]) for r in rows})

    fig, axes = plt.subplots(len(valcols), 1, figsize=(11, 2.4 * len(valcols)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    # 多台 switch 常跑几乎相同的流量 -> 线完全重合，单靠透明度分不开（只会糊成一团、越透越糊）。
    # 少数实体(<=8)改用「线宽递减」：先画的最粗、后画的最细，重合时呈同心色带，4 条都看得见（不用 marker，避免叠加变糊）。
    n = len(ents)
    taper = n <= 8
    for ax, c in zip(axes, valcols):
        for i, ent in enumerate(ents):
            xs, ys = [], []
            for r in rows:
                if sw_entity(r["level"], r["switch"], r["port"]) != ent:
                    continue
                try:
                    ys.append(float(r[c]))
                    xs.append(float(r["epoch"]) - t0)
                except (ValueError, KeyError):
                    pass
            if not xs:
                continue
            if taper:
                lw = 1.0 + (n - 1 - i) * 1.5     # 例 4 条: 5.5/4.0/2.5/1.0，同心可见
                ax.plot(xs, ys, lw=lw, alpha=0.9, solid_capstyle="butt", label=ent)
            else:                                # 实体太多(如 port 粒度)：细线 + 淡透明
                ax.plot(xs, ys, lw=0.7, alpha=0.5, label=ent)
        for tr, lab in marks:
            ax.axvline(tr, color="grey", ls="--", lw=0.7, alpha=0.6)
            ax.text(tr, ax.get_ylim()[1], lab, rotation=90, va="top", ha="right",
                    fontsize=7, color="grey")
        ax.set_ylabel(c, fontsize=9)
        if len(ents) <= 16:
            ax.legend(fontsize=8, loc="upper right", ncol=2)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("t_rel (s) since collector_start")
    fig.suptitle(f"NVSwitch traffic trace — {os.path.basename(rundir.rstrip('/'))}")
    fig.tight_layout()
    outdir = os.path.join(rundir, "plots")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "nvswitch_trace.png")
    fig.savefig(out, dpi=110)
    print("saved", out)


# ---------------- dispatch ----------------

def main():
    args = sys.argv[1:]
    if len(args) == 2 and args[0] in ("parse", "plot"):
        (cmd_parse if args[0] == "parse" else cmd_plot)(args[1])
        return
    print("usage:\n"
          "  python3 tool/metrics.py parse runs/<id>   # 宿主机, 纯 stdlib\n"
          "  python  tool/metrics.py plot  runs/<id>   # 容器, matplotlib",
          file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
