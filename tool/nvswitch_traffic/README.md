# tool/nvswitch_traffic —— 实时打印过 NVSwitch 的 NVLink 流量

一个小 CLI：直连 `libnvidia-nscq` 读 NVSwitch 每端口的 `throughput_counters`，差分成速率，像 `dcgmi dmon` 那样按间隔刷。
**这是本机唯一能拿到 NVSwitch 芯片流量的路**——DCGM 的 `nvswitch_throughput` 字段在本机 H100 恒 0
（NVSDM 后端 / [DCGM issue #236](https://github.com/NVIDIA/DCGM/issues/236)），本工具绕过 DCGM 直读 NSCQ。

> 为什么 DCGM 拿不到、NSCQ 能拿到，负载实测依据 → `experiments/nvswitch/`；数据源/口径/采样机制 → `docs/metrics_reference.md` 第二部分。

## 依赖 / 构建
```bash
cd tool/nvswitch_traffic && make            # 产出 ./nvswitch_traffic
make install PREFIX=/usr/local              # 可选：装进 PATH，脚本里直接 nvswitch_traffic ...
```
- 运行时只依赖系统的 `libnvidia-nscq.so`（随 nvidia 驱动 / Fabric Manager 安装）；头文件自包含（`nscq_min.h`）。
- **本机免 sudo** 即可读（FM 在跑也能并发只读）。

## 参数（风格照 `dcgmi dmon`；默认打印全部）
```
nvswitch_traffic [-e fields] [-i switches] [-l level] [-d ms] [-c count] [-r] [--csv] [-H]
  -e  字段(逗号): rx,tx,total          默认全部
  -i  switch 索引(逗号) 或 all          默认 all
  -l  粒度: total | switch | port       默认 switch
  -d  采样间隔毫秒                       默认 1000（有效地板 ~100ms，见文末）
  -c  采几拍后退出 (0 = 直到 Ctrl-C)     默认 0
  -r  打原始累计增量(Mibits) 而非速率(GB/s)
  --csv  CSV 输出(带 epoch，给脚本解析)     -H  不打表头
```

## 快速上手
```bash
# 实时看
nvswitch_traffic                     # 每 1s 每台物理 switch 的 rx/tx/total GB/s
nvswitch_traffic -d 200 -l total     # 每 0.2s 只看全 fabric 总带宽一行
nvswitch_traffic -l port -i 1        # 看 sw1 每个端口(每根 NVLink)的流量热点

# 脚本里采一段窗口存 CSV
nvswitch_traffic -d 1000 -c 30 --csv > runs/xxx/nvswitch.csv          # 采 30 拍
nvswitch_traffic -d 500 --csv > sw.csv & P=$!; run_workload; kill $P  # 后台起、跑完 kill
```
输出示例（P2P 流量打进来那一刻，4 台 switch 全亮）：
```
#time(s)   switch   RX(GB/s)  TX(GB/s)  TOT(GB/s)
3.31       sw0-3      0.000     0.000     0.000     ← idle
4.41       sw1      110.5     110.5     221.0       ← 负载进来
```

## 配置组 `<name>.conf`（`profile.sh --backend nvswitch` 用）
上层 `tool/profile.sh` 可把本工具当**独立后端**编排（`--backend nvswitch`，起停 + marks + wrap/attach + 出图，输出统一进 `runs/<id>/`）——
**集成用法/命令/输出见项目 [`README.md`](../../README.md) ② 段**；这里只讲本工具专属的**配置组**。

**配置组 = `tool/nvswitch_traffic/<name>.conf`**（`key=value`，`#` 注释；`profile.sh --sw-config <name>` 读它）——
角色对应 DCGM 的字段组 `tool/dcgmi/<name>.txt`。只放**采什么**（`-l/-e/-i/-r`）；**采多快**（`-d`）与起停由 `profile.sh` 的
`--interval-ms` + wrap/attach 统一控制，不在 conf 里。加新组 = 丢个 `.conf`，无需改代码。

| key | 对应参数 | 取值 | 默认 |
|---|---|---|---|
| `level` | `-l` | `total`(全 fabric 一行) / `switch`(每台物理 switch) / `port`(每端口, 256 行/拍) | `switch` |
| `fields` | `-e` | `rx,tx,total` 任意组合 | `rx,tx,total` |
| `switches` | `-i` | `all` 或 `0,1,2,3` | `all` |
| `raw` | `-r` | `0`=速率 GB/s，`1`=每拍原始累计增量 Mibits | `0` |

现成组：`sw_switch`(默认，每台 switch) · `sw_total`(全 fabric 一行，最省) · `sw_port`(每端口，看热点，数据量大)。

> **CSV 里 `switch`/`port` 列在聚合粒度填 `-1`（= N/A，不是出错）**：`total` 两列都 -1（整机一行）、`switch` 的 `port` 为 -1、
> 只有 `port` 粒度两列才都有真值。想看每台 switch 的编号用默认 `sw_switch`；`sw_total` 本就是把 4 台汇成一行。

## 输出口径（重要）
- **实体**：`switch` = 一颗**物理 NVSwitch 芯片**（本机 **4 颗** = PCIe 设备 05/06/07/08）；`port` = 该芯片上**一根 NVLink 端口**
  （每颗 64 端口，共 256）。`0..3` 按 UUID 排序稳定编号（注：`dcgmi discovery -l` 报"12 NvSwitch"是逻辑口径）。
- **数值**：源计数器是**累计值、单位 Mibits**，工具自动差分成 **GB/s（十进制）**；`-r` 打每拍原始增量。`total` = rx+tx。
- **视角 = fabric/端口**：一条跨卡流量在源侧 switch 计 rx、目的侧计 tx → **整机总带宽 ≈ 2× 单向应用带宽**。
  量"我的程序用了多少 NVLink"用 GPU 侧 `nvlink_tx/rx_bytes` 更直观；这工具适合看**每台 switch / 每端口的流量分布与热点**。
- **`port` 粒度**：终端下是**宽表**（每 switch 一行、端口做列，配 `-i` 选单台看更清爽）；CSV 下仍每端口一行便于解析。

## 采样地板 ~100ms
后台 reader 线程背靠背读 NSCQ（每次读全 256 端口 ~100ms = 一次采集），主线程按 `-d` 取最新快照差分打印——读写解耦，
`-d 100` 负载下实测稳定 ~100ms。地板来自**读延迟**（读 256 端口就要 ~100ms），非计数器刷新；`-d<100` 不会更快。
机制与 GPM 的 100ms 不同（详见 `docs/metrics_reference.md` §10）。

## 排障 / 局限
- **`no NVSwitch ports found`**：多为**偶发**（首拍与 DCGM 的 `nv-hostengine` 抢 NSCQ 返回空）。已内置重试，**重跑一次**通常就好。
  仍报错按提示查：`systemctl status nvidia-fabricmanager`、`ls /proc/driver/nvidia-nvswitch/devices`、`ls /usr/lib/x86_64-linux-gnu/libnvidia-nscq.so*`。
- `0..3` 是按 UUID 排序的稳定编号，**不保证**等于 `dcgmi` 的 `nvswitch:N` 物理 id（需对齐再加 physical_id 映射）。
- 读延迟 ~100ms 来自读全部 256 端口；只关心少数端口也省不掉（该 path 一次返回所有端口）。
