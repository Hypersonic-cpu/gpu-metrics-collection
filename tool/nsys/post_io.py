"""post_io.py —— 后处理四步之间的**契约**（不是 CLI，被另外三个脚本 import）。

```
① post_export.py          report.nsys-rep ─> post_metrics.csv + post_meta.json   纯标准库/宿主机
② post_plot.py overview   上面两个        ─> <iface>_overview.png                matplotlib/容器
③ dram_analyze.py         上面两个        ─> dram_stats.txt + dram_analysis.json numpy/容器
④ post_plot.py detail     ①③ 的产物      ─> <iface>_trace_*.png                 matplotlib/容器
```

`post_*` = **profile 之后的通用后处理**，与 metric 家族无关；`dram_analyze.py` 是**当前负载专属**
的那一步（周期尺认 `dram_total`，见该文件的「Notes for future usage」）。

集中在这里的三件事，就是四步能各自独立演进的原因：

1. **文件叫什么名 / CSV 长什么样 / 怎么读**（`CSV_NAME`…`load_csv`）。
   CSV 是**自描述**的：值列一律 `*_pct` 结尾，`load_csv()` 按表头动态发现 ——
   ① 换 metric 组（加 NVLink/PCIe 曲线）时 ②③④ 不用改。
2. **接口口径**（`IFACE` + `derive_family()`）：一个方向的带宽 = 哪几路相加、双向总量是和还是均值。
   ②③④ 全走这一个函数，所以统计表里的数和图上的线**永远是同一个定义**。
3. **换算 GB/s 的分母**（`IFACE_PEAK_GBPS` + `peak_for()`）和**每点像素**标定（`PX_PER_POINT`）。
"""

import csv
import json
import math
import os

CSV_NAME = "post_metrics.csv"
META_NAME = "post_meta.json"
STATS_NAME = "dram_stats.txt"           # ③ 是 adhoc 的那步，产物跟着它叫 dram_*
ANALYSIS_NAME = "dram_analysis.json"

# ① 的产物早先叫 dram_*，已归档的 runs/ 和 experiments/*/logs.tar.gz 里都是旧名。
# 读的时候两个名字都认（新名优先），写一律用新名。
LEGACY_CSV_NAME = "dram_metrics.csv"
LEGACY_META_NAME = "dram_meta.json"

GAP_NS = 1_000_000      # 相邻样本间隔 > 1ms 视为断块（正常 5–100 µs）：画图要断开、统计要排除

# ── 一张图该画多少个采样点：实测标定，见 CLAUDE.md §2.4 ──────────────────────
# 每个采样点分到 4 个横向像素时线形清晰（3.6 px/点 = 5 个 decode 循环那张，读着舒服；
# 2.05 px/点 = 100 等分那张，已经开始挤）。20in@110dpi 可用约 1870 px -> 约 470 点/图。
PX_PER_POINT = 4.0
AXES_FRAC = 0.85        # 轴区占图宽的比例（扣掉左右页边 + 右侧 GB/s 轴）

# ── 接口口径：一个方向的带宽 = 哪几路相加；双向总量 = 和 还是 均值 ──────────
# 列名 = f"{家族}_{part}"，都是 CSV 里的 `*_pct` 列（`pct_of_peak_sustained_elapsed`，整数 0–100）。
#
# `dirs` —— **同一个方向的多条分路相加**：nsys 自己的 `sets/*.config` 把同方向那几路标成
#   `type: stacked`（共用同一个分母），相加是它定义的读法。
#   NVLink 一个方向 = (request + response) × (user + protocol) **四路**：protocol 是链路上
#   真实占掉的字节（包头/ACK/credit），要算"这条链路被占了多少"必须带上它。实测
#   `nvl_write_fp8`：发送卡 tx = 69.57(user) + 15.58(proto) = **85.15%**，
#   只算 user 会少报 18%。（`user/(user+protocol)` = 链路有效率，那是
#   `experiments/nvlink_bench/nvlink_user_vs_proto.py` 的事，不在这条线里。）
#   PCIe 一个方向就一条计数器（协议开销已混在里面、拆不开），所以没有可加的。
#
# `total` —— **`mean` 而不是 `sum`**：NVLink/PCIe 的收发是**物理分离的双向链路**，
#   各自的百分比是各自峰值的占比，相加会出现 >100% 这种没有物理意义的数
#   （实测 nvl_write 的 tx 85% 而 rx 10%，相加 95% 会被读成"链路快满了"，其实是一个方向快满）。
#   取平均 = 这条链路两个方向的平均占用。DRAM 相反：读写是**同一片 HBM、共用一个分母**
#   （NVIDIA 自己的 `sets/dram.config` 把这俩标成 `type: stacked`），所以 `sum`。
#   实测 91261 个采样点里 `read+write` 最大 99、一个 >100 的都没有，佐证分母确实同一个。
IFACE = {
    "dram": {
        "dirs": {"read": ("read",), "write": ("write",)},
        "total": "sum",
    },
    "nvlink": {
        "dirs": {"rx": ("rx_req", "rx_rsp", "rx_req_proto", "rx_rsp_proto"),
                 "tx": ("tx_req", "tx_rsp", "tx_req_proto", "tx_rsp_proto")},
        "total": "mean",
    },
    "pcie": {
        "dirs": {"rx": ("rx",), "tx": ("tx",)},
        "total": "mean",
    },
}

# ── 各接口 pct_of_peak 的分母（GB/s）────────────────────────────────────────
# DRAM 的分母在报告里（`TARGET_INFO_GPU.memoryBandwidth`，本机 3352 GB/s），逐窗口读，
# 不用写死。这里只放**报告里没有**的接口分母 —— 这些是本机（zkrh-58，H100 SXM）的值，
# 换机器必须重新标定。
IFACE_PEAK_GBPS = {
    # ⚠️ **NVLink 不在这个表里**：它那组百分比的分母（"采样期内该方向最多能收/发多少字节"）
    #    本仓库不给它定数 → `peak_for()` 对 `nvlink_*` 返回 0 = **不换算 GB/s，只给 % of peak**。
    #    套别的接口的峰值会得出错值。比值（`user/(user+protocol)`、rx vs tx）与分母无关，照常可用。
    #
    # PCIe = **63.02 GB/s/方向**（Gen5 × 16，128b/130b 编码：32 GT/s × 16 / 8 × 128/130）。
    # 本机链路状态实测确认是 Gen5 × 16（`nvidia-smi --query-gpu=pcie.link.gen.current,
    # pcie.link.width.current` = 5, 16）。
    #
    # 已用已知带宽负载双向反推核对过（`experiments/cumemcpy/workloads/pcie_calib.py`，
    # 60 s 稳态 pinned memcpy，逐秒极差 0.55 GB/s）：
    #   H2D: payload 55.50 GB/s ÷ RX 91.936% -> 60.37 GB/s（占 63.02 的 95.8%）
    #   D2H: payload 39.52 GB/s ÷ TX 66.540% -> 59.39 GB/s（占 63.02 的 94.3%）
    # 这也顺带证明**两个方向各自归一化**（若共用一个双向分母 126 GB/s，55.5 GB/s 只该读到
    # 44%，不可能是 91.9%）—— 所以 total 取均值而不是相加。
    # ⚠️ **反推是一个方程两个未知量**：`payload ÷ pct = peak /(1+协议开销)`。PCIe 侧没有
    #    user/protocol 拆分（NVLink 才有），所以分不开。这里取**规格峰值 63.02**，
    #    残差就是协议开销 4.4–6.1% —— 大 TLP 下 PCIe 的头部开销正是这个量级，自洽。
    #    换个假设（开销=0）会得到 59.9，但那与 NVIDIA 对该计数器 "includes protocol"
    #    的定义矛盾。
    # ⇒ 用 63.02 换算出来的是**线上字节（含协议）**，正是这个计数器的定义；
    #    真实 payload 比它低约 5%（小 TLP 负载会更多）。
    "pcie": 63.02,
}


def peak_for(col, hbm_peak):
    """这一列换算 GB/s 该用哪个分母；返回 0 = **不换算**（分母未知，硬换会得出错值）。"""
    if col.startswith("dram_"):
        return hbm_peak
    for name, gbps in IFACE_PEAK_GBPS.items():
        if col.startswith(name + "_") or col == name:
            return gbps
    return 0.0


# ── 接口列的派生（②③④ 共用，保证统计表和图上的线是同一个定义）──────────────

def family_of(col):
    """列名 -> 它属于哪个 metric 家族（`sm_active`/`gr_active` 这种返回 None）。"""
    for fam in IFACE:
        if col.startswith(fam + "_"):
            return fam
    return None


def families_in(cols):
    """CSV 里有哪几个家族的列（按 IFACE 的顺序）。"""
    return [f for f in IFACE if any(family_of(c) == f for c in cols)]


def as_arrays(np, d, cols):
    """`load_csv()` 的一卡数据 -> {'t': 时间轴, 列名: np.array}（缺值 -> NaN）。"""
    out = {"t": np.asarray(d["t"], dtype=float)}
    for c in cols:
        out[c] = np.asarray([np.nan if x is None else x for x in d[c]], dtype=float)
    return out


def derive_family(d, fam):
    """在一卡的列字典里就地补出 `<fam>_{方向}` 和 `<fam>_total`，口径见 `IFACE`。

    **逐个采样点算**（CSV 一行一算），不是先取均值再合 —— 均值合出来的 total 会把
    "两个方向不同时忙"这件事抹平。NaN 照常传播：某一路缺了，和就是未知。

    返回 `{方向: [实际相加的列…]}`；某方向一路都没采到就不建那列（调用方跳过这条线）。
    调用方可以据此判断口径是否完整（比如 NVLink 只采了 user 4 路时，rx 里没有 protocol）。
    """
    spec = IFACE[fam]
    used = {}
    for dname, parts in spec["dirs"].items():
        src = [c for c in (f"{fam}_{p}" for p in parts) if c in d]
        if not src:
            continue
        used[dname] = src
        s = d[src[0]].copy()
        for c in src[1:]:
            s = s + d[c]
        d[f"{fam}_{dname}"] = s
    tot = f"{fam}_total"
    if tot not in d and len(used) == len(spec["dirs"]):
        cols = [f"{fam}_{x}" for x in spec["dirs"]]
        s = d[cols[0]].copy()
        for c in cols[1:]:
            s = s + d[c]
        d[tot] = s if spec["total"] == "sum" else s / len(cols)
    return used


def derive_all(d, cols):
    """把 CSV 里出现过的每个家族都派生一遍 -> {家族: {方向: [源列…]}}。"""
    return {f: derive_family(d, f) for f in families_in(cols)}


def missing_parts(fam, used):
    """`derive_family()` 的 used -> 哪几路没采到（用来提醒口径不完整）。"""
    out = []
    for dname, parts in IFACE[fam]["dirs"].items():
        got = set(used.get(dname, ()))
        out += [f"{fam}_{p}" for p in parts if f"{fam}_{p}" not in got]
    return out


def display_cols(cols):
    """报告里列的先后：按家族分组，每组先派生列（total/方向）再原始列，最后其它。"""
    out, seen = [], set()
    for fam in families_in(cols):
        raw = [c for c in cols if family_of(c) == fam]
        drv = [f"{fam}_total"] + [f"{fam}_{d}" for d in IFACE[fam]["dirs"]]
        for c in drv + raw:
            if c not in seen:
                seen.add(c)
                out.append(c)
    out += [c for c in cols if c not in seen and not family_of(c)]
    return out


def suggest_split(n_samples, width_in=20, dpi=110, nrows=1):
    """按 `PX_PER_POINT` 反推：这么多点该劈成几张图 -> (张数, 每张放得下多少点)。"""
    cap = width_in * dpi * AXES_FRAC * nrows / PX_PER_POINT
    return max(1, math.ceil(n_samples / cap)), cap


# ── 找文件 / 读写 ────────────────────────────────────────────────────────────

def find_csvs(path):
    """<路径> -> [post_metrics.csv, …]；给目录就递归找（run 目录 = 一次处理所有 window）。

    找不到新名就认旧名 `dram_metrics.csv`（已归档的 run）。
    """
    if os.path.isfile(path) and path.endswith(".csv"):
        return [path]
    if os.path.isdir(path):
        for name in (CSV_NAME, LEGACY_CSV_NAME):
            direct = os.path.join(path, name)
            if os.path.isfile(direct):
                return [direct]
        out, got = [], set()
        for root, _, files in os.walk(path):
            for name in (CSV_NAME, LEGACY_CSV_NAME):
                if name in files and root not in got:
                    got.add(root)
                    out.append(os.path.join(root, name))
        return sorted(out)
    return []


def out_dir_for(src_dir, outdir):
    """-o 给了就落 <outdir>/<window 目录名>/，没给就原地写回。"""
    if not outdir:
        return src_dir
    d = os.path.join(outdir, os.path.basename(src_dir.rstrip("/")))
    os.makedirs(d, exist_ok=True)
    return d


def read_json(src_dir, name):
    """读同目录下的 json（post_meta.json / dram_analysis.json）；没有就返回 {}。"""
    p = os.path.join(src_dir, name)
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def read_meta(src_dir):
    """① 写的 post_meta.json（旧 run 退回 dram_meta.json）。"""
    return read_json(src_dir, META_NAME) or read_json(src_dir, LEGACY_META_NAME)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_csv(csv_path):
    """post_metrics.csv -> dict（纯标准库，值是 list；调用方自己 np.asarray）。

    返回 {src, n_rows, cols, order, gpus:{g:{t,<col>…}}, peak:{g:GB/s}, meta}
    `cols` = 表头里所有 `*_pct` 列（去掉后缀），顺序照表头 —— 不写死 metric 名字。
    缺值留 None，让调用方决定当 NaN 还是跳过。
    """
    src = os.path.dirname(csv_path)
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        header = rdr.fieldnames or []
        cols = [h[:-4] for h in header if h.endswith("_pct")]
        gpus, peak, n = {}, {}, 0
        for r in rdr:
            g = int(r["gpu"])
            d = gpus.get(g)
            if d is None:
                d = gpus[g] = {"t": [], **{c: [] for c in cols}}
            d["t"].append(float(r["t_ms"]))
            if r.get("hbm_peak_GBps"):
                peak[g] = float(r["hbm_peak_GBps"])
            for c in cols:
                v = r.get(f"{c}_pct", "")
                d[c].append(float(v) if v not in ("", None) else None)
            n += 1
    return {"src": src, "n_rows": n, "cols": cols, "order": sorted(gpus),
            "gpus": gpus, "peak": peak, "meta": read_meta(src)}


def title_of(src_dir, meta):
    """图/报告的标题两行：window 目录名 + run_meta 的 note。"""
    win = meta.get("window") or os.path.basename(src_dir.rstrip("/"))
    return win, (meta.get("note") or "")


def ascii_only(s):
    """容器里没中文字体（画出来是方框 + UserWarning），图内文字一律降成 ASCII。
    中文说明留在控制台 / dram_stats.txt / README（CLAUDE.md 的约定）。换行要留住。"""
    return "\n".join(" ".join("".join(c for c in line if ord(c) < 128).split())
                     for line in s.split("\n"))
