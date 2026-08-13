#!/usr/bin/env python3
"""align_windows.py —— 把 nsys 采集窗口对齐到 server.log，算出每个窗口【真正】盖住了哪一段。

用法: python3 align_windows.py runs/nsys_<...>__<ts>/            # 打 markdown 表到 stdout

为什么必须做这一步：窗口的【请求偏移】和【实际落点】不是一回事。`nsys-tool gen` 从被调用到
窗口打开有 5–9s 固定开销，stop 之后写报告还要 10–22s 且全程阻塞 —— 所以后面的窗口会被顺次推后。
判断某个窗口到底采到了 prefill 还是 decode、batch 多大，只能靠 nsys 自己打的 window_start
（各窗口 marks.txt）去对 server.log，不能看 yaml 里写的偏移量。

读：<run>/windows.tsv（bench_start / burst_anchor 的 epoch）
    <run>/windows/*/marks.txt（window_start / window_end）
    <run>/server.log（每步 Prefill/Decode batch，UTC 时间戳，秒级）
"""
import datetime
import glob
import os
import re
import sys


def load_server_log(path, since):
    """server.log -> [(epoch, 'P'|'D', #running-req, cuda graph)]，只留 since 之后的。"""
    rows = []
    for ln in open(path, errors="replace"):
        m = re.match(r"\[(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)", ln)
        if not m or ("Prefill batch" not in ln and "Decode batch" not in ln):
            continue
        t = datetime.datetime.strptime(
            m.group(1) + " " + m.group(2), "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=datetime.timezone.utc).timestamp()
        if t < since:
            continue
        rr = re.search(r"#running-req: (\d+)", ln)
        cg = re.search(r"cuda graph: (\w+)", ln)
        rows.append((t, "P" if "Prefill batch" in ln else "D",
                     int(rr.group(1)) if rr else -1, cg.group(1) if cg else "?"))
    return rows


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: align_windows.py <run_dir>")
    run = sys.argv[1].rstrip("/")
    tsv = os.path.join(run, "windows.tsv")
    log = os.path.join(run, "server.log")
    if not (os.path.exists(tsv) and os.path.exists(log)):
        sys.exit(f"缺 windows.tsv 或 server.log: {run}")

    anchors = {}
    for ln in open(tsv).read().splitlines()[1:]:
        f = ln.split("\t")
        if len(f) >= 2 and f[1] in ("bench_start", "burst_anchor"):
            anchors[f[1]] = float(f[0])
    bench = anchors.get("bench_start")
    burst = anchors.get("burst_anchor", bench)
    if burst is None:
        sys.exit("windows.tsv 里没有锚点")

    # server.log 时间戳是秒级，往前多读 3s 免得边界丢行
    rows = load_server_log(log, burst - 3)
    if not rows:
        sys.exit("server.log 里没有 Prefill/Decode batch 行")
    lastP = max((r[0] for r in rows if r[1] == "P"), default=burst)

    print(f"- 锚点 `burst`（第一条压测 Prefill batch）= bench 启动后 "
          f"{burst - bench:.1f}s" if bench else "")
    print(f"- prefill 墙：burst+0 → +{lastP - burst:.0f}s；最后一条 decode 日志 "
          f"burst+{rows[-1][0] - burst:.0f}s")
    print()
    print("| 窗口 | 窗口真正打开 | 时长 | 期间批次 | `#running-req` | cuda graph |")
    print("|---|---|---|---|---|---|")
    for d in sorted(glob.glob(os.path.join(run, "windows", "nsys_*/"))):
        mk = os.path.join(d, "marks.txt")
        if not os.path.exists(mk):
            continue
        marks = {}
        for ln in open(mk).read().splitlines():
            f = ln.split("\t")
            if len(f) == 2:
                marks[f[1]] = float(f[0])
        ws, we = marks.get("window_start"), marks.get("window_end")
        if ws is None or we is None:
            continue
        inw = [r for r in rows if ws - 1 <= r[0] <= we + 1]
        p = sum(1 for r in inw if r[1] == "P")
        dd = sum(1 for r in inw if r[1] == "D")
        rrs = [r[2] for r in inw if r[2] > 0]
        cgs = sorted({r[3] for r in inw})
        name = os.path.basename(d.rstrip("/"))
        name = re.sub(r"^nsys_\w+?_(\d+_)", r"\1", name)
        name = re.sub(r"__\d{8}-\d{6}$", "", name)
        kinds = " ".join(x for x in (f"Prefill×{p}" if p else "", f"Decode×{dd}" if dd else "") if x)
        print(f"| {name} | burst{ws - burst:+.1f}s | {we - ws:.1f}s | {kinds or '—'} | "
              f"{f'{min(rrs)}–{max(rrs)}' if rrs else '—'} | {','.join(cgs) or '—'} |")


if __name__ == "__main__":
    main()
