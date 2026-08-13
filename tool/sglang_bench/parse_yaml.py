#!/usr/bin/env python3
"""parse_yaml.py —— 读一份实验配置（experiments/<名>.yaml），合并 defaults，逐个实验输出一行给 run.sh 消费。

用法: python3 parse_yaml.py experiments/default.yaml

每行一个实验，字段用 \\x1f(US) 分隔（非空白分隔符，空字段不会被 bash 折叠），列序 = 下面 COLS：
  name gpus tp model server bench note image models_dir sglang_repo port fields
  interval_ms server_timeout extra_docker model_tag label exclusive_process

规则：
- defaults 里的键给所有实验兜底；实验自身的键覆盖 defaults。
- tp 不填则 = gpus 的张数（逗号切分）；填了就用填的（EP/特殊场景可覆盖）。
- name / gpus / model 必填。
纯 stdlib + pyyaml（本机宿主机已装，见 README「前置」）。
"""
import sys

try:
    import yaml
except ImportError:
    sys.exit("需要 pyyaml：本机宿主机应已装（python3 -c 'import yaml'）。")

SEP = "\x1f"  # 列分隔符：US 控制符，正常参数里不会出现，且非空白（bash 读时不折叠空字段）

DEFAULTS = {
    "image": "lmsysorg/sglang:v0.5.15.post1-cu129",
    "models_dir": "/ssd/yjdiao/models",
    "sglang_repo": "/home/yjdiao/Repos/sglang",
    "port": 30000,
    "fields": "pass1_core",
    "interval_ms": 100,
    "server": "",
    "bench": "--num-prompts 1000 --random-input-len 1024 --random-output-len 512 --request-rate inf",
    "note": "",
    "server_timeout": 600,
    "extra_docker": "",
    "model_tag": "",   # 运行目录名里的短模型名；空则 run.sh 从 model 目录名推导
    "label": "",       # 运行目录名尾部的自定义描述（如 dramhigh）；空则不加
    "exclusive_process": True,  # true=可以独占→nvidia-smi -c 3；false=不能独占→-c 0(DEFAULT)
    # ── profiler：采什么后端 ──
    #   dcgm = 宿主机旁路采样，10Hz，startup + bench 两个全程窗口，出 metrics.csv/trace.png
    #   nsys = 容器内 Nsight Systems，server 起在 nsys 下，bench 期间开几个短窗口，出 .nsys-rep
    "profiler": "dcgm",
    "nsys_group": "cuda",          # nsys metric 组 -> tool/nsys/groups/<name>.conf
    # 采集窗口表，逗号分隔：<标签>@[bench|burst]<±偏移>:<窗口秒数>；不写锚点前缀 = burst。
    #   burst = server.log 里第一条真正的压测 Prefill batch（不是 bench_serving 的 warmup 单请求），
    #           这样窗口位置不受 warmup 时长漂移影响；bench = bench 进程启动时刻。
    #   prefill 必须挂 bench 锚点：prefill 墙只有 ~5s，而 burst 要等日志出现才探得到，
    #   等探到再发命令（还有 gen 自身的前置开销）窗口就已经错过它了。
    "nsys_windows": "prefill@bench+0:8,decode-hi@burst+17:2,decode-mid@burst+42:2,decode-lo@burst+67:2",
    # gen 从被调用到窗口真正打开的固定开销（docker exec + nsys start），实测 5–9s，提前这么多发命令
    "nsys_lead": 7,
    # 覆盖组文件的 GPU Metrics 采样频率（Hz）；空=用组文件的值。
    # ⚠️ 太高会 "Sampling buffer overflow"（Error 记在报告内部，stdout 看不到）→ 该卡数据错位作废。
    "nsys_freq": "",
    # 只在这几张卡上采 GPU Metrics（空=与 gpus 相同）。kernel 甘特图是 Application scope，
    # 不受这里影响、始终覆盖全部进程/卡。TP 对称负载下采一张卡就够，且能把采样缓冲压力减半。
    "nsys_gpus": "",
    # 把宿主机的 nsight-systems 挂进容器并用它——容器自带的 nsys 更新，产出的报告宿主机 nsys-ui 打不开
    "nsys_host_dir": "/usr/local/cuda-13.1/nsight-systems-2025.5.2",
}
COLS = ["name", "gpus", "tp", "model", "server", "bench", "note", "image",
        "models_dir", "sglang_repo", "port", "fields", "interval_ms",
        "server_timeout", "extra_docker", "model_tag", "label",
        "exclusive_process", "profiler", "nsys_group", "nsys_windows",
        "nsys_host_dir", "nsys_lead", "nsys_freq", "nsys_gpus"]


def truthy(v):
    """把 yaml 里的布尔/字符串统一判成 True/False（run.sh 侧只认 1/0）。"""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on", "y", "t")


def clean(v):
    if isinstance(v, list):
        v = ",".join(str(x) for x in v)
    s = str(v)
    for bad in ("\x1f", "\t", "\n", "\r"):
        s = s.replace(bad, " ")
    return s.strip()


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: parse_yaml.py <配置.yaml>")
    with open(sys.argv[1]) as f:
        doc = yaml.safe_load(f) or {}
    defaults = dict(DEFAULTS)
    defaults.update(doc.get("defaults") or {})
    exps = doc.get("experiments") or []
    if not exps:
        sys.exit(f"{sys.argv[1]} 里没有 experiments")
    for i, exp in enumerate(exps):
        m = dict(defaults)
        m.update(exp or {})
        for req in ("name", "gpus", "model"):
            if not m.get(req):
                sys.exit(f"experiment #{i} 缺少必填字段: {req}")
        gpus = clean(m["gpus"])
        if not m.get("tp"):
            m["tp"] = len([g for g in gpus.split(",") if g.strip()])
        m["exclusive_process"] = "1" if truthy(m.get("exclusive_process", True)) else "0"
        prof = clean(m.get("profiler", "dcgm")).lower()
        if prof not in ("dcgm", "nsys"):
            sys.exit(f"experiment #{i} 的 profiler 只能是 dcgm 或 nsys，给的是 {prof!r}")
        m["profiler"] = prof
        row = [gpus if c == "gpus" else clean(m.get(c, "")) for c in COLS]
        sys.stdout.write(SEP.join(row) + "\n")


if __name__ == "__main__":
    main()
