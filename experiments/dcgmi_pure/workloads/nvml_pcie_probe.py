#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task3 补充：DCGM 卡死在 10Hz，那"能不能高点"？测 NVML 这条不经过 DCP 的路。
NVML nvmlDeviceGetPcieThroughput 官方是"20ms 窗口的字节计数"(≈50Hz 上限, 只有 PCIe)。
本脚本在容器里(需 pynvml)高频(~1ms)采 RX/TX 吞吐，统计**值真正变化**的速率 = 有效刷新率。
需在有持续 PCIe 流量时跑(另起 h2d_loop 容器)，否则值恒 0 看不出刷新。

用法(容器内): python nvml_pcie_probe.py <seconds>
"""
import sys, time
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
import pynvml as N
N.nvmlInit()
h = N.nvmlDeviceGetHandleByIndex(0)   # 容器内重编号后 = 暴露的那张卡
RX = N.NVML_PCIE_UTIL_RX_BYTES
TX = N.NVML_PCIE_UTIL_TX_BYTES

samples = []   # (t, rx, tx)
t0 = time.time()
while time.time() - t0 < secs:
    t = time.time() - t0
    try:
        rx = N.nvmlDeviceGetPcieThroughput(h, RX)  # KB/s over ~20ms window
        tx = N.nvmlDeviceGetPcieThroughput(h, TX)
    except N.NVMLError as e:
        rx = tx = -1
    samples.append((t, rx, tx))
    # 尽量快地打，靠调用本身的开销自然间隔;不 sleep(0) 也行
N.nvmlShutdown()

n = len(samples)
dur = samples[-1][0] - samples[0][0]
call_rate = (n - 1) / dur
# RX 值发生变化的次数 = 有效刷新次数
chg = 0; prev = None
for _, rx, _ in samples:
    if rx != prev:
        chg += 1
    prev = rx
refresh_rate = chg / dur
rxvals = [rx for _, rx, _ in samples if rx > 0]
print(f"NVML pcie throughput 探针  时长={dur:.2f}s  样本={n}")
print(f"    调用速率(采多快)     = {call_rate:8.1f} calls/s")
print(f"    ★值刷新速率(有效)   = {refresh_rate:8.1f} refreshes/s   (官方: ~20ms 窗 => ~50Hz 上限)")
if rxvals:
    import statistics as st
    print(f"    RX 吞吐样例          = {st.median(rxvals)/1e6:.2f} GB/s (中位, KB/s->GB/s), 非零样本={len(rxvals)}/{n}")
