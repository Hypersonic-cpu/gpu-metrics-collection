#!/usr/bin/env python3
"""给 dcgmi dmon 的每行打时间戳。两种用法：

  stamp.py <rawfile>   # 边采边做两件事：
                       #   - 写文件 <rawfile>：每行前缀 `<epoch>\t`（供 metrics.py parse 解析）
                       #   - 打到 stdout：每行前缀人类可读 `HH:MM:SS `（终端实时看进度）
  stamp.py             # 无参数：仅把 `<epoch>\t` 写到 stdout（兼容老用法）

dcgmi dmon 的输出不带时间，管道过本脚本即可给每个采样打上真实墙钟时间。
纯 stdlib，逐行 flush；配合 `stdbuf -oL dcgmi dmon` 近实时。
"""
import sys
import time

raw = open(sys.argv[1], "w") if len(sys.argv) > 1 else None

for line in sys.stdin:
    now = time.time()
    if raw is None:
        sys.stdout.write(f"{now:.3f}\t{line}")
    else:
        raw.write(f"{now:.3f}\t{line}")
        raw.flush()
        clk = time.strftime("%H:%M:%S", time.localtime(now))
        sys.stdout.write(f"{clk} {line}")
    sys.stdout.flush()
