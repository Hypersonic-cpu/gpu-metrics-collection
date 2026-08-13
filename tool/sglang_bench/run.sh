#!/usr/bin/env bash
# run.sh —— SGLang dummy-weight 性能实验一键编排（在【宿主机】跑，驱动 docker + 宿主机 dcgmi 采集）
#
# 一条命令跑完一份配置（experiments/<短名>.yaml）里的一串实验。每个实验自动走完整条链路：
#   ⓪ 先查目标卡是否空闲（nvidia-smi 看显存占用）：忙则等几秒重试三次，仍忙就跳过本实验
#   ① docker run -d 起一个具名容器（sleep infinity 常驻）→ 容器内按 yaml 的 exclusive_process 设计算模式
#      （true=可以独占→nvidia-smi -c 3 EXCLUSIVE_PROCESS；false=不能独占→-c 0 DEFAULT 共享）
#   ② docker exec -d 后台起 server（日志落到挂载的 sglang/logs/）
#   ③ startup 窗口：宿主机 profile.sh wrap 住"轮询 server 就绪"→ server 一 ready 自动停采
#   ④ bench 窗口：  profile.sh wrap 住 docker exec 里的 bench_serving → bench 一结束自动停采
#   ⑤ 两个窗口各自 metrics.py parse(宿主机) + plot(容器)
#   ⑥ 按 name 把 runs/<时间戳>/ 改名成 server_<name>__<ts> / bench_<name>__<ts>，并把复现命令写进各自 README
#   ⑦ docker rm -f 关容器，跑下一个实验
# server 起不来（OOM/超时）→ 该窗口目录标 _FAIL、跳过 bench、继续下一个实验（不卡死）。
#
# yaml 里 profiler: nsys 时走另一条链路（③④⑤ 换成 run_nsys_phase）：server 起在 nsys 下（kernel
# 甘特图必须启动时注入）→ 等就绪但不采 → bench 后台跑 → 按窗口表在 prefill/decode 各开几个 2-3s
# 短窗口 → shutdown。产出 runs/nsys_<stem>__<ts>/（每个窗口一个子目录）。详见 README「两种 profiler」。
#
# 用法：
#   tool/sglang_bench/run.sh <配置> [选项]
# <配置> 必给（没有隐式默认）：experiments/ 下的短名（如 2gpu.nsys-timeline →
#        experiments/2gpu.nsys-timeline.yaml），或任意 yaml 路径。不给参数 = 列出可选配置后退出。
# 选项：
#   --list            列出 experiments/ 下的配置及其实验名，不跑
#   --only a,b        只跑指定 name 的实验（逗号分隔）
#   --interactive     半自动：只开好容器 + 打印 server/bench/profile 命令给你手动粘，容器留着不删
#   --keep-container  实验结束不删容器（默认删）
#   --dry-run         只打印每个实验会执行的命令，不真跑（不碰 docker/GPU）
#   -h|--help
#
# 前置：本脚本必须在【宿主机】跑（要用宿主机 dcgmi + docker）；不要在 sglang 容器里跑。详见同目录 README.md。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCROOT="$(cd "$HERE/../.." && pwd)"
PROFILE="$MCROOT/tool/profile.sh"
METRICS="$MCROOT/tool/metrics.py"
NSYSTOOL="$MCROOT/tool/nsys/nsys-tool"
NSYS_CTR_MNT="/opt/nsys-host"                      # 宿主机 nsight-systems 在容器里的挂载点
NSYS_CTR_BIN="$NSYS_CTR_MNT/target-linux-x64/nsys" # 用它而不是容器自带的（版本见 README）
PLOT_IMAGE="${PLOT_IMAGE:-nvcr.io/nvidia/pytorch:25.12-py3}"
SEP=$'\x1f'

EXPDIR="$HERE/experiments"           # 配置清单目录（短名 <n> -> experiments/<n>.yaml）
YAML=""                              # 必须显式给配置：没有隐式默认，免得手滑跑起一整份大模型清单
YAML_REF=""                          # 写进 run README 复现命令里的配置引用（短名或原路径）
ONLY=""; INTERACTIVE=0; KEEP=0; DRYRUN=0

usage(){ grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# 列出 experiments/ 下的配置：短名 + 首行注释 + 各实验 name
list_configs(){
  local f b
  for f in "$EXPDIR"/*.yaml; do
    [[ -e "$f" ]] || continue
    b="$(basename "$f" .yaml)"
    printf '%-28s %s\n' "$b" "$(head -1 "$f" | sed 's/^#\s*//; s/^[^—]*—— //')"
    grep -oE '^\s*-\s*name:\s*\S+' "$f" | sed 's/.*name:\s*//' | tr '\n' ' ' | sed 's/^/      /; s/\s*$/\n/'
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)          list_configs; exit 0;;
    --only)          ONLY="$2"; shift 2;;
    --interactive)   INTERACTIVE=1; shift;;
    --keep-container) KEEP=1; shift;;
    --dry-run)       DRYRUN=1; shift;;
    -h|--help)       usage 0;;
    -*)              echo "未知选项: $1" >&2; usage 1;;
    # 配置：短名（experiments/<n>.yaml）优先，其次当成路径
    *)               if   [[ -f "$EXPDIR/$1.yaml" ]]; then YAML="$EXPDIR/$1.yaml"; YAML_REF="$1"
                     elif [[ -f "$EXPDIR/$1" ]];      then YAML="$EXPDIR/$1";      YAML_REF="${1%.yaml}"
                     else                                  YAML="$1";              YAML_REF="$1"; fi
                     shift;;
  esac
done

[[ -n "$YAML" ]] || {
  echo "要跑哪份配置？（没有默认值——不给就不跑，免得手滑起一整份大模型清单）" >&2
  echo "" >&2
  list_configs >&2
  echo "" >&2
  echo "用法: tool/sglang_bench/run.sh <短名|yaml路径> [--only a,b] [--dry-run]" >&2
  exit 1
}
[[ -f "$YAML" ]] || {
  echo "找不到配置: $YAML" >&2
  echo "experiments/ 下可用的短名：" >&2
  ls "$EXPDIR"/*.yaml 2>/dev/null | xargs -rn1 basename | sed 's/\.yaml$//; s/^/  /' >&2
  exit 1
}
[[ -x "$PROFILE" ]]  || { echo "找不到/不可执行 profile.sh: $PROFILE" >&2; exit 1; }
command -v docker >/dev/null || { echo "需要 docker" >&2; exit 1; }
if [[ "$DRYRUN" != "1" ]]; then
  command -v dcgmi >/dev/null || echo "警告: 宿主机找不到 dcgmi，采集会失败（本脚本要在宿主机跑）" >&2
fi

# ---- 卡空闲检测 + 独占（EXCLUSIVE_PROCESS）参数（可用环境变量覆盖）----
HAVE_NVSMI=1; command -v nvidia-smi >/dev/null || HAVE_NVSMI=0
[[ "$DRYRUN" != "1" && "$HAVE_NVSMI" == "0" ]] && \
  echo "警告: 宿主机找不到 nvidia-smi，跳过卡空闲检测与独占设置" >&2
GPU_FREE_RETRIES="${GPU_FREE_RETRIES:-3}"    # 卡忙时最多检测几次
GPU_FREE_WAIT="${GPU_FREE_WAIT:-5}"          # 每次检测之间等几秒
GPU_FREE_MEM_MIB="${GPU_FREE_MEM_MIB:-1024}" # 显存占用超过此值(MiB)判为“忙”

# ---- 容器清理（Ctrl-C / 正常退出都收尾）----
CUR_CONTAINER=""
teardown(){
  [[ -n "$CUR_CONTAINER" && "$KEEP" != "1" ]] && {
    # 关容器前把独占计算模式复位成 Default（-c 0），别把卡留在独占态
    docker exec "$CUR_CONTAINER" nvidia-smi -c 0 >/dev/null 2>&1 || true
    echo "[run] 关容器 $CUR_CONTAINER"
    docker rm -f "$CUR_CONTAINER" >/dev/null 2>&1 || true
  }
  CUR_CONTAINER=""
}
trap 'teardown' EXIT INT TERM

sanitize(){ printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'; }

# 找 profile.sh 刚建的原始时间戳 run 目录（basename 形如 20260722-202132；按 mtime 取最新）
latest_raw_run(){
  local d b
  for d in $(ls -dt "$MCROOT"/runs/*/ 2>/dev/null); do
    b="$(basename "$d")"
    [[ "$b" =~ ^[0-9]{8}-[0-9]{6}$ ]] && { printf '%s' "${d%/}"; return 0; }
  done
  return 1
}

# parse(宿主机) + 改名 + plot(容器)；结果最终目录写进全局 LAST_FINAL
LAST_FINAL=""
process_run(){  # <raw_rundir> <final_basename>
  local raw="$1" finalb="$2" final="$MCROOT/runs/$2"
  local ts; ts="$(basename "$raw")"
  echo "[run] parse $raw" >&2
  python3 "$METRICS" parse "$raw" >&2 || echo "[run] parse 失败（继续）" >&2
  mv "$raw" "$final"
  # profile.sh 用裸时间戳往 runs/README.md 写了索引行，改名后把链接指向新目录名
  [[ -f "$MCROOT/runs/README.md" ]] && \
    sed -i "s#($ts/README.md)#($finalb/README.md)#g; s#\[$ts\]#[$finalb]#g" "$MCROOT/runs/README.md"
  echo "[run] plot $finalb（容器出图）" >&2
  ( cd "$MCROOT" && docker run --rm --user "$(id -u):$(id -g)" -e MPLCONFIGDIR=/tmp/mpl \
      -v "$MCROOT":/work -w /work "$PLOT_IMAGE" \
      python tool/metrics.py plot "runs/$finalb" ) >&2 || echo "[run] plot 失败（继续）" >&2
  LAST_FINAL="$final"
}

# 往 run 的 README.md 追加"复现命令"段（profile.sh 已建好含 meta+note 的 README）
append_readme(){  # <dir> <phase> <gpus> <image> <models_dir> <port> <cmd>
  local dir="$1" phase="$2" gpus="$3" image="$4" models_dir="$5" port="$6" cmd="$7"
  {
    echo
    echo "## reproduce ($phase)"
    echo '```bash'
    echo "# ① 开容器（宿主机）"
    echo "docker run --gpus '\"device=$gpus\"' --rm -it \\"
    echo "    --shm-size 32g --ipc=host --cap-add=SYS_ADMIN -p $port:$port \\"
    echo "    -v $models_dir:$models_dir \\"
    echo "    -v ~/.cache/huggingface:/root/.cache/huggingface \\"
    echo "    -v <sglang_repo>:/sgl-workspace/sglang-src \\"
    echo "    $image /bin/bash"
    echo
    echo "# ② 容器内 $phase"
    echo "$cmd"
    echo '```'
  } >> "$dir/README.md"
}

# ---- 卡空闲检测 + 独占 ----
# 判断这些物理卡是否空闲：有 compute 进程 或 显存占用超阈值 → 忙（返回1）；否则空（返回0）。
# 宿主机 nvidia-smi 能看到容器内的 GPU 进程（实测确认），所以主判据就是“有没有进程”；
# 显存阈值只作兜底（进程已退但显存没释放/泄漏的情况）。都在宿主机看，不进容器。
gpus_free(){  # <gpus_csv>
  [[ "$HAVE_NVSMI" == "1" ]] || return 0        # 没 nvidia-smi 就不拦
  local gpus="$1" apps mem
  # 主判据：有没有 compute 进程在用这些卡
  apps="$(nvidia-smi -i "$gpus" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  [[ -n "$apps" ]] && return 1
  # 兜底：没进程但显存仍被占（残留/泄漏）也算忙
  while IFS= read -r mem; do
    mem="${mem//[[:space:]]/}"
    [[ -z "$mem" ]] && continue
    (( mem > GPU_FREE_MEM_MIB )) && return 1
  done < <(nvidia-smi -i "$gpus" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  return 0
}

# 等卡空：最多 GPU_FREE_RETRIES 次、每次间隔 GPU_FREE_WAIT 秒；空→0，一直忙→1
wait_gpus_free(){  # <gpus_csv>
  local gpus="$1" i
  for (( i=1; i<=GPU_FREE_RETRIES; i++ )); do
    if gpus_free "$gpus"; then
      echo "[run] 卡 $gpus 空闲 ✓（第 $i/$GPU_FREE_RETRIES 次检测）"
      return 0
    fi
    echo "[run] 卡 $gpus 忙（第 $i/$GPU_FREE_RETRIES 次），当前占用：" >&2
    nvidia-smi -i "$gpus" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/    /' >&2
    nvidia-smi -i "$gpus" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null | sed 's/^/    proc: /' >&2
    (( i < GPU_FREE_RETRIES )) && sleep "$GPU_FREE_WAIT"
  done
  return 1
}

# 设计算模式：容器内是 root 且带 SYS_ADMIN，nvidia-smi -c <mode>；
#   -c 3 = EXCLUSIVE_PROCESS（独占），-c 0 = DEFAULT（共享，允许多进程共用一张卡）。
# 容器只见被分配的那几张卡（重编号 0..n），故无需 -i。失败不致命。
gpu_set_compute_mode(){  # <container> <mode 0|3> <人类可读标签>
  [[ "$HAVE_NVSMI" == "1" ]] || return 0
  if docker exec "$1" nvidia-smi -c "$2" >/dev/null 2>&1; then
    echo "[run] 计算模式已设为 $3"
  else
    echo "[run] ⚠ 设置计算模式（$3）失败（继续，非致命）" >&2
  fi
}

# ---- nsys 模式的采集编排（由 run_one 调用，容器已起好）----
# 和 dcgm 模式的三点差别：
#   ① server 必须【起在 nsys 下】—— CUDA kernel 甘特图是 Application scope，事后 attach 拿不到；
#   ② 不采 startup 窗口 —— 启动期时间线不是这个模式要的东西，全程 trace 报告也会大到没法看；
#   ③ bench 放后台跑，前台按窗口表在 prefill / decode 各阶段开几个【短】窗口。
# 产出：一个实验一个 runs/nsys_<stem>__<ts>/，每个窗口是它 windows/ 下的一个子目录。
# 依赖 run_one 的局部变量（bash 动态作用域）：cname stem gpus name note server_cmd bench_cmd
#   server_log_host server_log_ctr server_timeout nsys_group nsys_windows image models_dir port。
run_nsys_phase(){
  local ts parent wdir
  ts="$(date +%Y%m%d-%H%M%S)"
  parent="$MCROOT/runs/nsys_${stem}__${ts}"
  wdir="$parent/windows"
  mkdir -p "$wdir"
  export NSYS_BIN="$NSYS_CTR_BIN"   # 容器里用宿主机那份 nsys，产出的报告宿主机 GUI 才打得开
  # GPU Metrics 采哪几张卡：默认全部；nsys_gpus 可以只挑一张（TP 对称时够用，且减半缓冲压力）。
  # kernel 甘特图不受影响 —— 它是 Application scope，覆盖所有被 launch 的进程。
  local mgpus="${nsys_gpus:-}"; [[ -z "$mgpus" ]] && mgpus="$gpus"

  # ① server 起在 nsys 下（launch = 把 trace 注进去，但先不开采）
  echo "[run] nsys launch server（组 $nsys_group，metrics 采卡 $mgpus / 模型用卡 $gpus）: $server_cmd"
  if ! "$NSYSTOOL" launch --in-container "$cname" -g "$nsys_group" -i "$mgpus" -o "$wdir" \
        -- bash -lc "$server_cmd > $server_log_ctr 2>&1"; then
    echo "[run] ⚠ nsys launch 失败，跳过实验 $name" >&2
    mv "$parent" "$MCROOT/runs/nsys_${stem}_FAIL__${ts}" 2>/dev/null || true
    return 1
  fi

  # ② 等就绪（不采集）
  bash "$HERE/wait_ready.sh" "$server_log_host" "$server_timeout"
  local prc=$?
  if [[ $prc -ne 0 ]]; then
    echo "[run] ⚠ server 未就绪 (wait_ready rc=$prc)，跳过 bench。server.log 末尾："
    tail -n 20 "$server_log_host" 2>/dev/null | sed 's/^/    /'
    "$NSYSTOOL" shutdown -o "$wdir" >/dev/null 2>&1 || true
    cp "$server_log_host" "$parent/server.log" 2>/dev/null || true
    mv "$parent" "$MCROOT/runs/nsys_${stem}_FAIL__${ts}" 2>/dev/null || true
    return 0
  fi

  # ③ bench 放后台（前台留给窗口调度）
  local from_line; from_line="$(wc -l < "$server_log_host")"
  echo "[run] 发 bench（后台）: $bench_cmd"
  mkdir -p "$wdir"   # 兜底：server 启动那几分钟里目录被清掉过，重定向会 ENOENT 直接吞掉 bench
  docker exec "$cname" bash -lc "$bench_cmd" > "$parent/bench.log" 2>&1 &
  local bench_pid=$!
  local bench_epoch; bench_epoch="$(date +%s.%3N)"

  # ④ 锚点探测放【后台】：bench 锚点的窗口（prefill）必须在 burst 之前就发出去，等不了探测结果。
  local burst_f="$parent/.burst_epoch"
  ( bash "$HERE/wait_burst.sh" "$server_log_host" "$from_line" 300 > "$burst_f" ) &
  local burst_pid=$!

  # ⑤ 按窗口表逐个开窗口
  { printf 'epoch\tlabel\tanchor\toffset_s\tdur_s\n'
    printf '%s\t%s\t%s\t%s\t%s\n' "$bench_epoch" "bench_start" "bench" "0" "-"; } > "$parent/windows.tsv"
  local -a WINS; IFS=',' read -ra WINS <<< "$nsys_windows"
  local i=0 spec rest wlabel anchor off dur base now sleep_s t0=""
  for spec in "${WINS[@]}"; do
    i=$((i+1))
    wlabel="${spec%%@*}"; rest="${spec#*@}"
    # <标签>@[bench|burst]<±偏移>:<时长>；不写锚点前缀 = burst
    anchor="burst"
    case "$rest" in
      bench[+-]*) anchor="bench"; rest="${rest#bench}";;
      burst[+-]*) anchor="burst"; rest="${rest#burst}";;
    esac
    off="${rest%%:*}"; dur="${rest##*:}"
    if [[ "$anchor" == "bench" ]]; then
      base="$bench_epoch"
    else
      if [[ -z "$t0" ]]; then      # 第一个 burst 锚点的窗口才需要等探测结果
        wait "$burst_pid"; t0="$(cat "$burst_f" 2>/dev/null)"
        if [[ -z "$t0" ]]; then
          echo "[run] ⚠ 没等到压测流量，burst 锚点退回 bench 启动时刻（窗口会偏 warmup 那几秒）" >&2
          t0="$bench_epoch"
        fi
        printf '%s\t%s\t%s\t%s\t%s\n' "$t0" "burst_anchor" "burst" "0" "-" >> "$parent/windows.tsv"
      fi
      base="$t0"
    fi
    # 提前 nsys_lead 秒发命令：gen 从被调用到窗口真正打开有固定开销（docker exec + nsys start，
    # 实测 5–9s）。不补这一下，2–3s 的短窗口会整体后移、错过 prefill 墙那种短阶段。
    now="$(date +%s.%3N)"
    sleep_s="$(awk -v b="$base" -v o="$off" -v l="$nsys_lead" -v n="$now" \
                   'BEGIN{d=b+o-l-n; printf "%.3f", (d>0?d:0)}')"
    echo "[run] 窗口 $i/${#WINS[@]} '$wlabel'：${anchor}${off}s 起、采 ${dur}s（提前量 ${nsys_lead}s，还要等 ${sleep_s}s）"
    sleep "$sleep_s"
    if ! kill -0 "$bench_pid" 2>/dev/null; then
      echo "[run] ⚠ bench 已经结束，剩下的窗口（从 $wlabel 起）不采了" >&2
      break
    fi
    mkdir -p "$wdir"
    if "$NSYSTOOL" gen -t "$dur" -o "$wdir" ${nsys_freq:+--freq "$nsys_freq"} --name "$(printf '%02d' "$i")_${wlabel}" \
         --note "$name | $wlabel | ${anchor}${off}s 采 ${dur}s"; then
      # 记窗口【真正打开】的时刻（nsys 自己打的 window_start），不是 gen 返回的时刻
      local ws; ws="$(awk -F'\t' '$2=="window_start"{print $1}' \
            "$(ls -dt "$wdir"/nsys_*"_${wlabel}"__*/ 2>/dev/null | head -1)marks.txt" 2>/dev/null)"
      printf '%s\t%s\t%s\t%s\t%s\n' "${ws:-$(date +%s.%3N)}" "$wlabel" "$anchor" "$off" "$dur" \
          >> "$parent/windows.tsv"
    else
      echo "[run] ⚠ 窗口 $wlabel 采集失败（继续下一个）" >&2
    fi
  done
  kill "$burst_pid" 2>/dev/null; wait "$burst_pid" 2>/dev/null

  # ⑥ 等 bench 跑完 + 收尾（shutdown 连 server 一起结束）
  echo "[run] 等 bench 结束..."
  wait "$bench_pid"; local brc=$?
  # --in-container 显式给：shutdown 平时从状态文件继承容器名，状态文件没了就会去宿主机找 nsys
  "$NSYSTOOL" shutdown --in-container "$cname" -o "$wdir" 2>&1 | sed 's/^/    /' || true
  # 采集期的告警/错误 nsys 只写进报告内部（stdout 一个字不打），最狠的是采样缓冲溢出——
  # 溢出那张卡的 GPU Metrics 会整段错位、数据作废，但 rc 仍是 0。所以跑完统一查一次。
  # 放在这里而不是每个窗口：查一次要 export sqlite（~1s/份），塞进窗口路径会把窗口起点推后。
  echo "[run] 查报告内部诊断..."
  # 退出码有语义：0=全 PASS / 1=有 PARTIAL / 2=有 FAIL 或 ERROR。存一份进 run 目录，
  # write_nsys_readme 会把它贴进 README，事后不用重跑就知道这批数据能不能用。
  "$NSYSTOOL" diagnose "$wdir" > "$parent/diagnose.txt" 2>&1
  DIAG_RC=$?
  sed 's/^/    /' "$parent/diagnose.txt"
  case $DIAG_RC in
    1) echo "[run] ⚠ 有窗口是 PARTIAL：报告里【部分】GPU Metrics 可用，见 README / diagnose.txt" >&2;;
    2) echo "[run] ⚠ 有窗口是 FAIL/ERROR：那些窗口的数据别直接拿来分析，见 README / diagnose.txt" >&2;;
  esac
  cp "$server_log_host" "$parent/server.log" 2>/dev/null || true

  local final="$parent"
  if [[ $brc -ne 0 ]]; then
    final="$MCROOT/runs/nsys_${stem}_FAIL__${ts}"
    mv "$parent" "$final" 2>/dev/null && parent="$final"
    echo "[run] ⚠ bench rc=$brc，见 $parent/bench.log" >&2
  fi
  write_nsys_readme "$parent" "$ts" "$brc"
  echo "[run] 实验 $name 完成 → $parent"
  return 0
}

# nsys run 目录的 README + 索引行（dcgm 模式那套由 profile.sh 写，这里自己写）
write_nsys_readme(){  # <parent> <ts> <bench_rc>
  local dir="$1" ts="$2" brc="$3"
  {
    echo "# run $(basename "$dir")"
    echo
    echo "nsys kernel 时间线 | $name | ${note:-—}"
    echo
    echo "## meta"
    echo "- **host**: $(uname -n)"
    echo "- **backend**: nsys / 组 \`$nsys_group\`（见 \`tool/nsys/$nsys_group.conf\`）"
    echo "- **gpus**: $gpus (tp$tp)  **model**: $model"
    echo "- **image**: $image"
    echo "- **nsys**: 用宿主机挂进容器的 \`$nsys_host_dir\`（容器自带版本更新、宿主机 GUI 打不开）"
    echo "- **windows**: \`$nsys_windows\`（\`<标签>@<burst 后第几秒>:<窗口秒数>\`）"
    echo "- **bench_rc**: $brc"
    echo "- **诊断**: $(case ${DIAG_RC:-0} in 0) echo '全部 PASS ✓';; 1) echo '⚠ 有 PARTIAL（部分 GPU Metrics 可用）';; 2) echo '⚠ 有 FAIL/ERROR';; esac)"
    echo
    if [[ -f "$dir/diagnose.txt" ]]; then
      echo "## 报告可用性判定（nsys-tool diagnose）"
      echo
      echo "nsys 把采集期的 Error 只写进【报告内部】，stdout 一个字不打、rc 仍是 0。"
      echo "下面是逐份报告的判定；**PARTIAL 表示报了 overflow 但仍有整路 GPU Metrics 可用**，"
      echo "明细（含每路设备的连续块 / 频率 / 覆盖率）见各窗口目录的 \`diagnostics.txt\`。"
      echo
      echo '```'
      cat "$dir/diagnose.txt"
      echo '```'
      echo
    fi
    echo "## 采到了哪几段（窗口 × server.log 对齐）"
    echo
    echo "**看这张表，别看 yaml 里写的偏移量**：\`gen\` 有 5–9s 前置开销、写报告还要 10–22s 且阻塞，"
    echo "所以窗口实际落点会比请求的晚，且越靠后越晚。下面用各窗口 \`marks.txt\` 的 \`window_start\` 算的。"
    echo
    python3 "$HERE/align_windows.py" "$dir" 2>/dev/null || echo "（对齐失败，原始时刻见 \`windows.tsv\`）"
    echo
    echo "## 怎么看"
    echo '```bash'
    echo "nsys-ui $dir/windows/<窗口>/report.nsys-rep"
    echo "nsys stats --report cuda_gpu_kern_sum $dir/windows/<窗口>/report.nsys-rep"
    echo '```'
    echo
    echo "## reproduce"
    echo '```bash'
    echo "tool/sglang_bench/run.sh $YAML_REF --only $name"
    echo '```'
    echo
    echo "- server: \`$server_cmd\`"
    echo "- bench:  \`$bench_cmd\`"
  } > "$dir/README.md"

  local idx="$MCROOT/runs/README.md" n
  n="${note//$'\n'/ }"; n="${n//|/\\|}"; [[ -z "$n" ]] && n="—"
  [[ -f "$idx" ]] && printf '| [%s](%s/README.md) | %s | %s |\n' \
      "$(basename "$dir")" "$(basename "$dir")" "nsys/$nsys_group" "sglang kernel 时间线 | $n" >> "$idx"
  return 0
}

# ---- 单个实验 ----
run_one(){
  local name="$1" gpus="$2" tp="$3" model="$4" server="$5" bench="$6" note="$7" \
        image="$8" models_dir="$9" sglang_repo="${10}" port="${11}" fields="${12}" \
        interval="${13}" server_timeout="${14}" extra_docker="${15}" \
        model_tag="${16}" label="${17}" excl="${18}" \
        profiler="${19}" nsys_group="${20}" nsys_windows="${21}" nsys_host_dir="${22}" \
        nsys_lead="${23}" nsys_freq="${24}" nsys_gpus="${25}"
  local sname; sname="$(sanitize "$name")"
  local cname="sglperf_${sname}_$$_${EXP_IDX}"
  local model_path="$models_dir/$model"
  local host_logdir="$sglang_repo/logs"
  local server_log_host="$host_logdir/${sname}.server.log"
  local server_log_ctr="/sgl-workspace/sglang-src/logs/${sname}.server.log"

  local server_cmd="python3 -m sglang.launch_server --model-path $model_path --load-format dummy --tp-size $tp --host 0.0.0.0 --port $port --enable-metrics $server"
  local bench_cmd="python3 -m sglang.bench_serving --backend sglang --dataset-name random-ids $bench --host 127.0.0.1 --port $port"

  # ---- 组装统一命名的运行目录名 stem（字段固定顺序）----
  #   <ngpu>gpu_<model>_tp<N>_ep<M>_in<X>_out<Y>_<cudagraph>[_<label>]
  #   phase(server/bench) 与 _FAIL / __<ts> 由调用处补。
  local ngpu; ngpu="$(awk -F, '{print NF}' <<<"$gpus")"      # gpus 张数
  # 短模型名：优先 YAML model_tag，否则从 model 目录名去掉 -config/-instruct 噪声
  local mtag="$model_tag"
  if [[ -z "$mtag" ]]; then mtag="$model"; mtag="${mtag%-config}"; mtag="${mtag%-instruct}"; fi
  mtag="$(sanitize "$mtag")"
  # 并行度分组：tp 后面紧跟 ep（专家并行），把所有 parallelism 字段排在一起。
  # ep 从 server 的 --ep-size 推；没写(dense/未启用专家并行)→ ep0，保证字段恒在、可对齐横比。
  local ep; ep="$(grep -oE -- '--ep-size[ =]+[0-9]+' <<<"$server" | grep -oE '[0-9]+' | head -1)"
  [[ -z "$ep" ]] && ep=0
  local par="tp${tp}_ep${ep}"
  # in/out seq len（从 bench 抠）
  local io="" ilen olen
  ilen="$(grep -oE -- '--random-input-len[ =]+[0-9]+'  <<<"$bench" | grep -oE '[0-9]+' | head -1)"
  olen="$(grep -oE -- '--random-output-len[ =]+[0-9]+' <<<"$bench" | grep -oE '[0-9]+' | head -1)"
  [[ -n "$ilen" && -n "$olen" ]] && io="_in${ilen}_out${olen}"
  # cuda graph 模式（从 server 参数推导，始终有值，保持字段统一）
  local cg="graph"
  if   grep -q  -- '--disable-cuda-graph'                          <<<"$server"; then cg="eager"
  elif grep -qE -- '--cuda-graph-backend-prefill[ =]+tc_piecewise' <<<"$server"; then cg="tcpiece"
  elif grep -qE -- '--cuda-graph-backend-prefill[ =]+disabled'     <<<"$server"; then cg="nopfgraph"
  fi
  # 自定义 label（可空）
  local lab=""; [[ -n "$label" ]] && lab="_$(sanitize "$label")"
  local stem="${ngpu}gpu_${mtag}_${par}${io}_${cg}${lab}"

  # nsys 模式多挂一份宿主机的 nsight-systems：容器自带的 nsys 版本更新，
  # 它产出的 .nsys-rep 宿主机 nsys-ui 打不开（GUI 版本必须 >= 产出版本）。
  local nsys_mount=()
  [[ "$profiler" == "nsys" ]] && nsys_mount=(-v "$nsys_host_dir:$NSYS_CTR_MNT:ro")

  local docker_run=(docker run -d --name "$cname"
      --gpus "\"device=$gpus\""
      --shm-size 32g --ipc=host --cap-add=SYS_ADMIN
      -p "$port:$port"
      -v "$models_dir:$models_dir"
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
      -v "$sglang_repo:/sgl-workspace/sglang-src"
      "${nsys_mount[@]}"
      $extra_docker "$image" sleep infinity)

  echo ""
  echo "==================================================================="
  echo "[run] 实验 $name  (gpus=$gpus  tp=$tp  model=$model)"
  echo "==================================================================="

  if [[ "$DRYRUN" == "1" ]]; then
    if [[ "$profiler" == "nsys" ]]; then echo "# 运行目录名: nsys_${stem}__<ts>"
    else echo "# 运行目录名: server_${stem}__<ts> , bench_${stem}__<ts>"; fi
    if [[ "$excl" == "1" ]]; then echo "# 计算模式: EXCLUSIVE_PROCESS (nvidia-smi -c 3)"
    else echo "# 计算模式: DEFAULT (nvidia-smi -c 0，exclusive_process=false)"; fi
    echo "# docker run:"; printf '  %q' "${docker_run[@]}"; echo
    if [[ "$profiler" == "nsys" ]]; then
      echo "# 每个窗口一个子目录: nsys_${stem}__<ts>/windows/nsys_${nsys_group}_<NN>_<标签>__<ts>/"
      echo "# server (容器内, 起在 nsys 下): $NSYSTOOL launch --in-container $cname -g $nsys_group -i ${nsys_gpus:-$gpus} -- bash -lc '$server_cmd > $server_log_ctr'"
      echo "# 等就绪(不采集): bash $HERE/wait_ready.sh $server_log_host $server_timeout"
      echo "# bench (容器内, 后台): $bench_cmd"
      echo "# 采集窗口表: $nsys_windows   (锚点 burst=第一条压测 Prefill batch / bench=bench 启动；提前量 ${nsys_lead}s)"
      local _s _at
      for _s in ${nsys_windows//,/ }; do
        _at="$(echo "${_s#*@}" | cut -d: -f1)"
        [[ "$_at" == bench* || "$_at" == burst* ]] || _at="burst+$_at"
        echo "#   $NSYSTOOL gen -t ${_s##*:} --name ${_s%%@*}   # 窗口开在 ${_at}s"
      done
      echo "# 收尾: $NSYSTOOL shutdown（连 server 一起结束）"
    else
      echo "# server (容器内, 后台): $server_cmd  > $server_log_ctr"
      echo "# startup 采集: $PROFILE --gpus $gpus --fields $fields --interval-ms $interval --lead 1 --lag 1 -- bash wait_ready.sh $server_log_host $server_timeout"
      echo "# bench (容器内): $bench_cmd"
      echo "# bench 采集:   $PROFILE --gpus $gpus --fields $fields --interval-ms $interval -- docker exec $cname bash -lc '<bench>'"
    fi
    return 0
  fi

  # ---- 独占前置：先确认目标卡空闲（忙则等几秒重试，三次仍忙就跳过本实验）----
  if ! wait_gpus_free "$gpus"; then
    echo "[run] ⚠ 卡 $gpus 重试 ${GPU_FREE_RETRIES} 次仍非空闲，跳过实验 $name" >&2
    return 0
  fi

  mkdir -p "$host_logdir"
  : > "$server_log_host"   # 清空旧 log，免得 wait_ready 读到上一次的 ready 行

  echo "[run] 起容器 $cname"
  "${docker_run[@]}" >/dev/null || { echo "[run] docker run 失败，跳过本实验" >&2; return 1; }
  CUR_CONTAINER="$cname"

  # ---- 计算模式（容器内 root + SYS_ADMIN）：按 yaml 的 exclusive_process ----
  #   1=可以独占 → -c 3 (EXCLUSIVE_PROCESS)；0=不能独占 → -c 0 (DEFAULT，允许多进程共卡)
  if [[ "$excl" == "1" ]]; then
    gpu_set_compute_mode "$cname" 3 "EXCLUSIVE_PROCESS（独占）"
  else
    gpu_set_compute_mode "$cname" 0 "DEFAULT（共享，exclusive_process=false）"
  fi

  # ---- 半自动：开好容器 + 打印命令，交给你手动 ----
  if [[ "$INTERACTIVE" == "1" ]]; then
    cat <<EOF

  --- interactive：容器已就绪，下面命令自己粘（宿主机 & 容器混合）---
  # ② 起 server（容器内后台，日志宿主机 $server_log_host 可看）
  docker exec -d $cname bash -lc '$server_cmd > $server_log_ctr 2>&1'
  # ③ startup 窗口采集（宿主机；server ready 自动停）
  $PROFILE --gpus $gpus --fields $fields --interval-ms $interval --lead 1 --lag 1 --note 'server | $name' -- bash $HERE/wait_ready.sh $server_log_host $server_timeout
  # ④ bench 窗口采集（宿主机；bench 结束自动停）
  $PROFILE --gpus $gpus --fields $fields --interval-ms $interval --note 'bench | $name' -- docker exec $cname bash -lc '$bench_cmd'
  # ⑤ 处理：python3 $METRICS parse runs/<id>  然后容器出图（见 README）
  # 完事复位独占： docker exec $cname nvidia-smi -c 0
  # 完事收容器：   docker rm -f $cname
EOF
    CUR_CONTAINER=""   # 交给用户，别在 trap 里删
    return 0
  fi

  # ---- nsys 模式在这里分岔：不采 startup、bench 期间开几个短窗口（详见 run_nsys_phase）----
  if [[ "$profiler" == "nsys" ]]; then
    run_nsys_phase
    return $?
  fi

  # ---- ③ startup 窗口：起 server + wrap 住"等就绪" ----
  echo "[run] 起 server（后台）: $server_cmd"
  docker exec -d "$cname" bash -lc "$server_cmd > $server_log_ctr 2>&1"
  local snote="server | $name | ${note:-—} | $model tp$tp | ${server:-<sglang默认>}"
  "$PROFILE" --gpus "$gpus" --fields "$fields" --interval-ms "$interval" --lead 1 --lag 1 \
      --note "$snote" -- bash "$HERE/wait_ready.sh" "$server_log_host" "$server_timeout"
  local prc=$?

  local raw ts
  raw="$(latest_raw_run)" || { echo "[run] 找不到 startup run 目录，跳过" >&2; return 1; }
  ts="$(basename "$raw")"
  local sfx=""; [[ $prc -ne 0 ]] && sfx="_FAIL"
  process_run "$raw" "server_${stem}${sfx}__${ts}"
  local sfinal="$LAST_FINAL"
  cp "$server_log_host" "$sfinal/server.log" 2>/dev/null || true
  append_readme "$sfinal" "server" "$gpus" "$image" "$models_dir" "$port" "$server_cmd > logs/${sname}.server.log 2>&1"

  if [[ $prc -ne 0 ]]; then
    echo "[run] ⚠ server 未就绪 (wait_ready rc=$prc)，跳过 bench。server.log 末尾："
    tail -n 20 "$server_log_host" 2>/dev/null | sed 's/^/    /'
    return 0
  fi

  # ---- ④ bench 窗口：wrap 住容器内 bench ----
  echo "[run] 发 bench: $bench_cmd"
  local bnote="bench | $name | ${note:-—} | $bench"
  "$PROFILE" --gpus "$gpus" --fields "$fields" --interval-ms "$interval" \
      --note "$bnote" -- docker exec "$cname" bash -lc "$bench_cmd"
  local brc=$?

  raw="$(latest_raw_run)" || { echo "[run] 找不到 bench run 目录" >&2; return 1; }
  ts="$(basename "$raw")"
  local bsfx=""; [[ $brc -ne 0 ]] && bsfx="_FAIL"
  process_run "$raw" "bench_${stem}${bsfx}__${ts}"
  local bfinal="$LAST_FINAL"
  # 把 sglang 自己的两份 log 都收进 bench 目录，方便在一个文件夹里看全：
  #   server.log = server 完整运行期 log（含 bench 窗口的 gen throughput；此刻 host log 已累积到 bench 结束）
  #   bench.log  = bench_serving 客户端输出（workload.log 的副本，命名更直观；workload.log 被 metrics.py 依赖，保留不动）
  cp "$server_log_host"     "$bfinal/server.log" 2>/dev/null || true
  cp "$bfinal/workload.log" "$bfinal/bench.log"  2>/dev/null || true
  append_readme "$bfinal" "bench" "$gpus" "$image" "$models_dir" "$port" "$bench_cmd"
  [[ $brc -ne 0 ]] && echo "[run] ⚠ bench rc=$brc，见 $bfinal/workload.log"
  echo "[run] 实验 $name 完成 → $sfinal , $bfinal"
  return 0
}

# ---- 读 YAML → 逐个跑 ----
TMP="$(mktemp)"
if ! python3 "$HERE/parse_yaml.py" "$YAML" > "$TMP"; then
  echo "解析 YAML 失败：" >&2; cat "$TMP" >&2; rm -f "$TMP"; exit 1
fi
mapfile -t LINES < "$TMP"; rm -f "$TMP"
[[ ${#LINES[@]} -gt 0 ]] || { echo "YAML 里没有可跑的 experiments" >&2; exit 1; }

EXP_IDX=0
for line in "${LINES[@]}"; do
  IFS="$SEP" read -r name gpus tp model server bench note image models_dir sglang_repo port fields interval server_timeout extra_docker model_tag label excl profiler nsys_group nsys_windows nsys_host_dir nsys_lead nsys_freq nsys_gpus <<<"$line"
  if [[ -n "$ONLY" ]]; then
    case ",$ONLY," in *",$name,"*) ;; *) continue;; esac
  fi
  EXP_IDX=$((EXP_IDX+1))
  run_one "$name" "$gpus" "$tp" "$model" "$server" "$bench" "$note" "$image" \
          "$models_dir" "$sglang_repo" "$port" "$fields" "$interval" "$server_timeout" "$extra_docker" \
          "$model_tag" "$label" "$excl" "$profiler" "$nsys_group" "$nsys_windows" "$nsys_host_dir" "$nsys_lead" "$nsys_freq" "$nsys_gpus"
  teardown   # 关掉本实验容器（interactive 下 CUR_CONTAINER 已置空，不会误删）
done

echo ""
echo "[run] 全部完成。运行目录见 $MCROOT/runs/（server_* / bench_* 已按 name 改名）。"
