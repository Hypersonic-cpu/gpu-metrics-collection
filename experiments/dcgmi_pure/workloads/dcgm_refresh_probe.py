#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task3 探针：DCGM profiling 字段的"真实内部更新率"到底多少？能不能 >10Hz？
思路：用 dcgmi dmon -d 1(=1ms 请求) 高频拉一个/多个 profiling 字段，给每行打**墙钟时间戳**，
     统计两件事：
       (1) dmon 实际出行速率 (lines/s)  —— host 端能拉多快
       (2) 字节字段真正"刷新"(出现新非空值)的速率 (updates/s) —— 这才是内部更新率
     再对比 1 字段 vs 16 字段，看 multiplexing 是否拖慢内部更新。
只用标准库(subprocess/time)，在 host 上直接跑(dcgmi 在 host)。原始带时间戳日志落 dcgmi_pure/。

用法: python3 dcgm_refresh_probe.py <gpu> <count> <outdir>
"""
import subprocess, time, sys, os

GPU   = sys.argv[1] if len(sys.argv) > 1 else "6"
COUNT = sys.argv[2] if len(sys.argv) > 2 else "3000"
OUTD  = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACK = os.environ.get("TRACK", "1009")   # 要监视其"刷新"的字段 id(默认 PCITX;加载 H2D 测时用 1010=PCIRX)
TAG   = os.environ.get("TAG", "idle")     # 日志文件名标签(idle / h2dload)

# 两个条件：单 profiling 字段(=TRACK) vs full 16 字段。间隔都请求 1ms(=远超内部率)。
CONDS = [
    (f"1field_1ms_{TAG}",  TRACK),
    (f"16field_1ms_{TAG}", "1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,449,204,252,250"),
]

def run(name, fields):
    log = os.path.join(OUTD, f"freqtest_{name}_gpu{GPU}.txt")
    cmd = ["dcgmi", "dmon", "-e", fields, "-i", GPU, "-d", "1", "-c", COUNT]
    ts, vals = [], []            # 每个数据行的(时间戳, 第一个profiling字段值或None)
    hdr_pcitx_idx = None         # PCITX 在数据行里的位置(去掉 GPU <id> 后)
    fld = fields.split(",")
    # TRACK 字段在字段串里的序号 => 数据值里的序号(数据行前2列是 GPU/<id>)
    pcitx_pos = fld.index(TRACK)
    t0 = time.time()
    with open(log, "w") as fh:
        fh.write(f"# probe {name} fields={fields} gpu={GPU} -d 1 -c {COUNT}\n")
        fh.write("# 每行前缀 = 相对墙钟时间(s)\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        for line in p.stdout:
            now = time.time() - t0
            fh.write(f"{now:9.4f}  {line}")
            s = line.rstrip("\n")
            if s.startswith("GPU"):
                tok = s.split()
                data = tok[2:]                 # 去掉 GPU <id>
                v = None
                if pcitx_pos < len(data):
                    raw = data[pcitx_pos]
                    v = None if raw == "N/A" else float(raw)
                ts.append(now); vals.append(v)
        p.wait()
    return log, ts, vals

def analyze(name, ts, vals):
    n = len(ts)
    if n < 2:
        return f"{name}: 数据行不足({n})"
    dur = ts[-1] - ts[0]
    line_rate = (n - 1) / dur if dur > 0 else 0
    # 非空(=有真值)样本数：byte 字段空读=0 或 warmup=N/A 都算空
    nonempty = [v for v in vals if v is not None and v > 0]
    ne = len(nonempty)
    update_rate = ne / dur if dur > 0 else 0
    # "刷新事件"：相邻样本值发生变化(且新值非空)——另一种数内部更新的方式
    changes = 0
    prev = None
    for v in vals:
        if v is not None and v > 0 and v != prev:
            changes += 1
        prev = v
    change_rate = changes / dur if dur > 0 else 0
    empty_pct = 100 * (n - ne) / n
    return (f"{name}: 数据行={n}  时长={dur:.2f}s\n"
            f"    dmon 实际出行率   = {line_rate:7.1f} lines/s   (请求 -d 1 = 1000Hz)\n"
            f"    非空样本占比      = {100-empty_pct:6.1f}%  (空读 {empty_pct:.1f}%)\n"
            f"    ★内部更新率(非空) = {update_rate:7.1f} updates/s\n"
            f"    ★内部更新率(变化) = {change_rate:7.1f} changes/s")

if __name__ == "__main__":
    out = [f"# DCGM 内部更新率探针  gpu={GPU} count={COUNT}  time0(rel)", ""]
    for name, fields in CONDS:
        log, ts, vals = run(name, fields)
        res = analyze(name, ts, vals)
        out.append(res); out.append(f"    raw log -> {log}"); out.append("")
        print(res); print("    raw log ->", log); print()
        time.sleep(1)
    summ = os.path.join(OUTD, f"freqtest_summary_gpu{GPU}.txt")
    with open(summ, "w") as f:
        f.write("\n".join(out) + "\n")
    print("[summary] ->", summ)
