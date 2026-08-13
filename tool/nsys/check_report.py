#!/usr/bin/env python3
"""check_report.py —— 判定一份 .nsys-rep 到底还能不能用。

用法:
    check_report.py <run目录 | report.nsys-rep> ...     # 目录会递归找所有 report.nsys-rep
    check_report.py runs/nsys_xxx/ --json               # 机器可读
    check_report.py runs/nsys_xxx/ -j 4                 # 并行导出（默认 4）

为什么需要它（不能只看有没有 Error）:
    nsys 把采集期的错误只写进【报告内部】(GUI 的 Diagnostics 页)，stdout 一个字不打、
    rc 仍是 0、报告照样生成。而其中最常见的 "GPU Metrics [N]: Sampling buffer overflow"
    **不等于整份报告作废**——实测它只砸中 [N] 那一路设备：那张卡的样本被切成两块、
    全部落到采集窗口之外(覆盖率 0%)，而同一份报告里另一张卡照样是干净的一整块、
    精确达标频率、100% 覆盖 kernel 段。所以光看 Error 有没有，会把"还能用一半"的报告
    当成全废扔掉。

判定口径:
    PASS    没有任何 Error
    PARTIAL 报了 overflow，但报告里【至少一路】GPU Metrics 仍然完整可用
    FAIL    报了 overflow，且【没有任何一路】GPU Metrics 可用
    ERROR   有 overflow 以外的 Error（一概按 ERROR 报，具体受影响的轨见明细行）

    "某一路可用" = 该设备最大连续采样块覆盖了 kernel 时间段的 >= COVER_OK。
    报告里没有 kernel 轨(trace=none 的组)时，改判"最大连续块是否占到全部样本的 >= COVER_OK"。

退出码: 0=全 PASS；1=有 PARTIAL；2=有 FAIL 或 ERROR（可直接用在 CI/编排里）
副产物: 每个报告目录写一份 diagnostics.txt（判定 + 完整 Warning/Error + 每路数据体检）
"""

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

GAP_NS = 1_000_000      # 采样间隔 > 1ms 视为断块（正常间隔是 5–100 µs）
COVER_OK = 0.80         # 最大连续块覆盖 kernel 段这个比例以上，算这一路可用

# Error 分类：overflow 单独处理，其余按"影响哪条轨"标注（但一律计为 ERROR）
RE_OVERFLOW = re.compile(r"gpu metrics\s*\[(\d+)\].*overflow", re.I)
RE_CPU_ONLY = re.compile(r"perf:|cpu ip/backtrace|context switch", re.I)


def _nsys_version(path):
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True,
                             timeout=20).stdout
    except Exception:                                       # noqa: BLE001
        return None
    m = re.search(r"(\d{4})\.(\d+)\.(\d+)", out or "")
    return tuple(int(x) for x in m.groups()) if m else None


def host_nsys():
    """找宿主机侧能读报告的 nsys —— 按【版本】挑最新的，不能只信 PATH。

    坑：本机 PATH 里的 nsys 是 /usr/local/cuda-12.8/bin/nsys (2024.6.2)，比报告的产出版本旧，
    读新报告会报 "Please update your Nsight Systems to the latest version to view this report."
    而 /usr/local/cuda/bin/nsys 是 2025.5.2。GUI/CLI 版本必须 >= 产出版本，所以这里枚举所有
    候选、比版本号取最大的。想固定用某个可以 NSYS_HOST_BIN=<路径> 覆盖。
    """
    env = os.environ.get("NSYS_HOST_BIN")
    if env and os.access(env, os.X_OK):
        return env
    import glob as _g
    cands = {shutil.which("nsys")}
    for pat in ("/usr/local/cuda/bin/nsys", "/usr/local/cuda-*/bin/nsys",
                "/opt/nvidia/nsight-systems/*/bin/nsys"):
        cands.update(_g.glob(pat))
    best, best_v = None, None
    for c in sorted(x for x in cands if x and os.access(x, os.X_OK)):
        v = _nsys_version(c)
        if v and (best_v is None or v > best_v):
            best, best_v = c, v
    return best


def blocks(ts):
    """把一路样本的时间戳切成连续块（相邻间隔 > GAP_NS 就断开）。"""
    out, cur = [], [ts[0]]
    for a, b in zip(ts, ts[1:]):
        if b - a > GAP_NS:
            out.append(cur)
            cur = []
        cur.append(b)
    out.append(cur)
    return out


def analyse(sq):
    """读 sqlite，返回 {errors, warnings, kernel:{}, devices:[...]}。"""
    c = sqlite3.connect(sq)
    tabs = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}

    errs, warns = [], []
    if "DIAGNOSTIC_EVENT" in tabs:
        for sev, text, n in c.execute(
                "select severity, text, count(*) from DIAGNOSTIC_EVENT "
                "where severity in (2,3) group by severity, text"):
            (errs if sev == 3 else warns).append({"text": text, "count": n})

    kern = {"count": 0, "span_s": 0.0, "t0": None, "t1": None}
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tabs:
        n, t0, t1 = list(c.execute(
            "select count(*), min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL"))[0]
        if n:
            kern = {"count": n, "span_s": (t1 - t0) / 1e9, "t0": t0, "t1": t1}

    devs = []
    if "GPU_METRICS" in tabs:
        tids = [r[0] for r in c.execute(
            "select distinct m.typeId from GPU_METRICS m "
            "join TARGET_INFO_GPU_METRICS t on m.metricId=t.metricId and m.typeId=t.typeId "
            "where t.metricName like 'DRAM%' or t.metricName like '%Active%' order by 1")]
        for i, tid in enumerate(tids):
            ts = [r[0] for r in c.execute(
                "select m.timestamp from GPU_METRICS m "
                "join TARGET_INFO_GPU_METRICS t on m.metricId=t.metricId and m.typeId=t.typeId "
                f"where m.typeId={tid} and t.metricName=("
                "  select t2.metricName from TARGET_INFO_GPU_METRICS t2 "
                f"  where t2.typeId={tid} limit 1) order by m.timestamp")]
            if not ts:
                continue
            bl = blocks(ts)
            big = max(bl, key=len)
            span = (big[-1] - big[0]) / 1e9
            khz = (len(big) / span / 1000) if span > 0 else 0.0
            if kern["count"]:
                ov = max(0, min(big[-1], kern["t1"]) - max(big[0], kern["t0"]))
                cover = ov / (kern["t1"] - kern["t0"])
                basis = "kernel"
            else:                       # 没有 kernel 轨时，退而看"是不是一整块"
                cover = len(big) / len(ts)
                basis = "samples"
            devs.append({"idx": i, "samples": len(ts), "n_blocks": len(bl),
                         "biggest_s": span, "khz": khz,
                         "cover": cover, "cover_basis": basis,
                         "usable": cover >= COVER_OK})
    return {"errors": errs, "warnings": warns, "kernel": kern, "devices": devs,
            "has_metrics": "GPU_METRICS" in tabs}


def verdict(a):
    """按判定口径给结论 + 一句人话理由。"""
    ovf, other = [], []
    for e in a["errors"]:
        (ovf if RE_OVERFLOW.search(e["text"] or "") else other).append(e)

    if other:
        track = "仅 CPU 轨" if all(RE_CPU_ONLY.search(e["text"] or "") for e in other) else "未分类"
        return "ERROR", f"有 overflow 以外的 Error（{track}）"

    if not ovf:
        if a["has_metrics"] and a["devices"] and not any(d["usable"] for d in a["devices"]):
            # 没报错但数据对不上窗口，同样不能用（例如窗口开早了）
            return "FAIL", "无 Error，但没有一路 GPU Metrics 覆盖到 kernel 段"
        return "PASS", "无 Error"

    idx = [RE_OVERFLOW.search(e["text"]).group(1) for e in ovf]
    ok = [d for d in a["devices"] if d["usable"]]
    bad = [d for d in a["devices"] if not d["usable"]]
    if not a["devices"]:
        return "FAIL", f"GPU Metrics [{','.join(idx)}] 溢出，且报告里没有可用的 metrics"
    if not ok:
        return "FAIL", f"GPU Metrics [{','.join(idx)}] 溢出，{len(bad)}/{len(a['devices'])} 路全部作废"
    return "PARTIAL", (f"GPU Metrics [{','.join(idx)}] 溢出；"
                       f"{len(ok)}/{len(a['devices'])} 路仍完整可用"
                       + (f"，{len(bad)} 路作废" if bad else "（数据看着完整，但 nsys 确实报了错）"))


def detail_lines(a, v, why):
    L = [f"判定: {v} —— {why}", ""]
    if a["kernel"]["count"]:
        L.append(f"kernel 轨: {a['kernel']['count']:,} 条，跨度 {a['kernel']['span_s']:.2f}s")
    else:
        L.append("kernel 轨: 无（组里 trace=none，或窗口内没有 CUDA 活动）")
    for d in a["devices"]:
        L.append(f"  metrics 设备[{d['idx']}]: {d['samples']:,} 点 / {d['n_blocks']} 块 / "
                 f"最大块 {d['biggest_s']:.2f}s @{d['khz']:.1f}kHz / "
                 f"覆盖{'kernel段' if d['cover_basis']=='kernel' else '样本'} {d['cover']*100:.0f}% "
                 f"{'✓ 可用' if d['usable'] else '✗ 作废'}")
    if a["errors"]:
        L += [""] + [f"[ERROR] x{e['count']}  {e['text']}" for e in a["errors"]]
    if a["warnings"]:
        L += [""] + [f"[WARN]  x{w['count']}  {w['text']}" for w in a["warnings"]]
    return "\n".join(L) + "\n"


EXPORT_LOCK = __import__("threading").Lock()


def export_sqlite(nsys_bin, rep, out):
    """导 sqlite。并发跑多个 nsys export 会互相踩（实测 -j4 时大报告成片失败），
    所以失败后串行重试一次；仍失败就把 nsys 自己的报错带回去，别吞掉。"""
    cmd = [nsys_bin, "export", "--type", "sqlite", "--force-overwrite", "true",
           "-o", out, rep]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and os.path.exists(out):
        return True, ""
    with EXPORT_LOCK:                       # 串行重试
        if os.path.exists(out):
            os.unlink(out)
        r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and os.path.exists(out):
        return True, ""
    tail = [l for l in (r.stderr or r.stdout or "").replace("\r", "\n").splitlines()
            if l.strip() and not re.match(r"^\[=*\d+%", l)]
    return False, (tail[-1][:160] if tail else f"rc={r.returncode}")


def check_one(rep, nsys_bin, keep_sqlite=False):
    d = os.path.dirname(rep)
    sq = os.path.join(d, "report.sqlite")
    tmp = None
    if not (os.path.isfile(sq) and os.path.getsize(sq) > 4096):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False,
                                          dir=d, prefix=".chk-")
        tmp.close()
        os.unlink(tmp.name)          # nsys 要求目标不存在 / 用 --force-overwrite
        ok, err = export_sqlite(nsys_bin, rep, tmp.name)
        if not ok:
            return {"report": rep, "dir": d, "verdict": "ERROR",
                    "why": f"读不了报告: {err}", "errors": [], "devices": []}
        sq = tmp.name
    try:
        a = analyse(sq)
    except Exception as e:                                  # noqa: BLE001
        return {"report": rep, "verdict": "ERROR", "why": f"解析失败: {e}",
                "errors": [], "devices": []}
    finally:
        if tmp and os.path.exists(tmp.name) and not keep_sqlite:
            os.unlink(tmp.name)
    v, why = verdict(a)
    with open(os.path.join(d, "diagnostics.txt"), "w") as f:
        f.write(detail_lines(a, v, why))
    return {"report": rep, "dir": d, "verdict": v, "why": why,
            "kernel": a["kernel"]["count"], "devices": a["devices"],
            "errors": [e["text"] for e in a["errors"]]}


def main():
    ap = argparse.ArgumentParser(add_help=True, description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", help="run 目录 或 report.nsys-rep")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("-j", "--jobs", type=int, default=4, help="并行导出数（默认 4）")
    ap.add_argument("--keep-sqlite", action="store_true", help="保留导出的 sqlite")
    args = ap.parse_args()

    nsys_bin = host_nsys()
    if not nsys_bin:
        sys.exit("[check] 宿主机上找不到 nsys，没法读报告")

    reps = []
    for p in args.paths:
        if os.path.isfile(p):
            reps.append(p)
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                if "report.nsys-rep" in files:
                    reps.append(os.path.join(root, "report.nsys-rep"))
    reps.sort()
    if not reps:
        sys.exit(f"[check] 没找到 .nsys-rep: {' '.join(args.paths)}")

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        res = list(ex.map(lambda r: check_one(r, nsys_bin, args.keep_sqlite), reps))

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        mark = {"PASS": "✅ PASS   ", "PARTIAL": "⚠️  PARTIAL", "FAIL": "❌ FAIL   ",
                "ERROR": "❌ ERROR  "}
        for r in res:
            name = re.sub(r"__\d{8}-\d{6}$", "", os.path.basename(r.get("dir", "")))
            print(f"  {mark[r['verdict']]} {name:<34} {r['why']}")
            for d in r["devices"]:
                if not d["usable"]:
                    print(f"              └ 设备[{d['idx']}] {d['n_blocks']} 块 / "
                          f"覆盖 {d['cover']*100:.0f}% → 这一路别用")
        tally = {}
        for r in res:
            tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
        print("  " + "  ".join(f"{k}={v}" for k, v in
                               sorted(tally.items(), key=lambda x: x[0])) +
              f"   (共 {len(res)} 份；明细见各目录 diagnostics.txt)")

    if any(r["verdict"] in ("FAIL", "ERROR") for r in res):
        sys.exit(2)
    if any(r["verdict"] == "PARTIAL" for r in res):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
