# tool/dcgmi —— 用 `dcgmi dmon` 采 GPU 侧 metric

这个目录 = **怎么用 `dcgmi dmon` 命令本身**采 GPU 侧 metric（HBM / SM / PCIe / GPU 侧 NVLink / 显存占用），
外加本项目的**字段组文件**（`<name>.txt`）。`dcgmi` 是**宿主机**二进制（本机 DCGM 装在 host）。

> - 整条采集流水线（起停 + 打墙钟戳 + wrap/attach + 解析出图）由上层 `tool/profile.sh` 编排 → 见项目 `README.md`。
> - 选哪个字段 / 字段语义 / 权限 / 坑的完整依据 → `docs/metrics_reference.md`（§1 选字段、§2 命令、§7 可用性）。

## 依赖
- 宿主机装了 DCGM：`dcgmi`、`nv-hostengine` 在 `/usr/bin`。确认：`dcgmi discovery -l`。
- `-i` 用 GPU **物理卡号**；选空卡（GPU0 常被占）：`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`。

## 快速上手
```bash
# 直接采（字段 id 手写；GPU 4,5；1000ms；不给 -c 则持续到 Ctrl-C）
dcgmi dmon -e 1005,449,1011,1012,1009,1010,252 -i 4,5 -d 1000

# 从本目录字段组文件取 id（-c 30 = 采 30 拍后退出）
dcgmi dmon -e "$(grep -oE '^[0-9]+' tool/dcgmi/pass1_core.txt | paste -sd, -)" -i 4 -d 100 -c 30
```

## 参数（`dcgmi dmon -e <fields> -i <entities> -d <ms> [-c <count>]`）
| 参数 | 含义 | 取值 / 说明 |
|---|---|---|
| `-e` | 要采的**字段 id 列表**（逗号）| 如 `1005,1011,1012,1009,1010,449,252`；id 见 `metrics_reference` §1 速查表 / §7 可用性 |
| `-i` | 采**哪些实体** | GPU：`-i 4,5`（**物理 id**，非容器内重编号）；NVSwitch：`-i nvswitch:0,nvswitch:1,…`（**wildcard 不支持**，须显式列）|
| `-d` | **采样间隔（ms）** | 默认 1000；**有效地板 ~100ms**（NVML GPM 内部刷新率，非 DCGM 策略），再快只产空行 |
| `-c` | 采**几拍后退出** | 不给 = 持续到 Ctrl-C。⚠️ 短测 `-c 3` 会被 warmup 坑（见下）|

## 字段组文件（`tool/dcgmi/<name>.txt`）
- 一行一个 dcgmi field id，`#` 开头为注释；取 id：`grep -oE '^[0-9]+' <file> | paste -sd, -`。
- **`pass1_core.txt`（默认，核心接口 utilization）**：
  `1005 dram_active`(HBM 利用率 0–1, profiling) · `449 nvlink_bandwidth_total`(NVLink 总带宽 MB/s, device 字段) ·
  `1011/1012 nvlink_tx/rx_bytes`(profiling) · `1009/1010 pcie_tx/rx_bytes`(profiling) · `252 fb_used`(显存占用 MB, device)。
- 其余遍（pass2 每链路 / pass3 SM / extra_compute / extra_base / fallback_noadmin / full）的规划见 `PLAN.md`「待做」。

## 输出格式（读日志要知道）
- 一行一个"**实体 × 拍**"；表头是**字段短名**：`DRAMA/SMACT/NVLTX/NVLRX/PCITX/PCIRX/NBWLT/MCUTL/FBUSD…`
  —— 需按 **短名 ↔ id ↔ 全名** 映射（`metrics.py parse` 已做）。列头单位标注可能被截断（PCIe 列头显示 `MB/`，实际是 **bytes/秒**）。
- ★ **warmup**：byte/profiling 字段头 ~2 拍返回 `N/A`，第 3 拍起才出真值 → **至少跑 5 拍**再信；短测 `-c 3` 易误判"字段坏"。
- ★ **byte 字段已是速率、不用差分**：`pcie_*_bytes`/`nvlink_*_bytes` 是 bytes/秒（1Hz 与 10Hz 原始值一致）→ `GB/s = 值/1e9`。

## 权限
- **profiling 字段**（id≥1000：`dram_active`/`sm_active`/`pcie_*_bytes`/`nvlink_*_bytes`/`tensor_active`…）：容器内需 `--cap-add SYS_ADMIN`。
- **device 字段**（id<1000：`449 nvlink_bandwidth_total`/`252 fb_used`/`204 mem_copy_util`…）：免额外权限。
- **本宿主机**：profiling 直接跑宿主机**无需额外 sudo**（idle 返回合法 `0` 而非权限错误）。

## 坑速记（详见 `docs/metrics_reference.md`）
- **别设 >10Hz**：profiling/byte 有效采样地板 100ms，更快只产空行（byte 空读=`0` 与真无流量二义、ratio 空读=`N/A`）。§3.1
- **`pcie_*_bytes` idle 非 0**：~17 MB/s 底噪（DCGM 轮询走 PCIe），分析扣掉；其它接口 idle 即 0。§5.3/§7.4
- **不可用**：`pcie_*_throughput`(200/201) Deprecated 全程 N/A → 用 `pcie_*_bytes`。§7.1
- **别用 `sm_active` 判"有没有搬数据"**：跨边界拷贝走 Copy Engine、`sm_active=0`；接口带宽用 `dram_active`/`*_bytes`。§5.6
- **NVSwitch 侧吞吐字段(780/781/861/862…)本机恒 0** → 过交换机流量用 `tool/nvswitch_traffic`（NSCQ 直读）。§5.2
