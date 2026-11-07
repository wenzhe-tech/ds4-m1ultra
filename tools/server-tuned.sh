#!/bin/sh
# ds4-server 启动器（M1 Ultra 调优版 + 官方采样默认值）
#
# 用法：  ./tools/server-tuned.sh                  前台启动（默认 0.0.0.0:8000，ctx 262144）
#         DS4_TRACE=1 ./tools/server-tuned.sh      额外打开 --trace（每次请求的采样参数会写进 trace）
#         CTX=32768 PORT=8123 ./tools/server-tuned.sh   临时改上下文/端口
#         SAMP_TOP_P=0.9 ./tools/server-tuned.sh   临时改采样默认值
#
# 采样默认值语义：ds4-server 把它们用作「客户端没传该字段时」的默认值，
# 客户端显式传的值永远优先（见 ds4-server --help sampling）。
#
# 关于下面的环境变量：上游把一批调优默认值门控在 M3 Ultra / M5 / M6 上，
# M1 Ultra 不在名单里。本仓库已用 m1_ultra_defaults.patch 在
# ds4_gpu_detect_metal4_features() 里给 M1 Ultra 注入同样的默认值，
# 所以这里再 export 一遍是冗余的（但显式 env 优先，值相同，无副作用）。
set -e

DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
LOG=${DS4_LOG:-$HOME/Library/Logs/ds4-server.log}

# ---- 可覆盖的启动参数（默认值就是你手上那条命令）----
MODEL=${MODEL:-Qwen3.8-Flash-Next-Q4.gguf}
VISION=${VISION:-$DIR/gguf/mmproj-Qwen3.8-Flash-Next-F16.gguf}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
CTX=${CTX:-262144}
KV=${KV:-/tmp/ds4-kv}
KV_MB=${KV_MB:-262144}

# ---- 官方（Qwen3.8 Flash 思考模式）采样默认值 ----
SAMP_TEMP=${SAMP_TEMP:-1.0}
SAMP_TOP_P=${SAMP_TOP_P:-0.95}
SAMP_TOP_K=${SAMP_TOP_K:-20}
SAMP_MIN_P=${SAMP_MIN_P:-0.0}
# 温度 > 0 时，MTP 默认只在 draft == target argmax 时接受，等于局部偷用贪心。
# 打开下面这项让投机保持原温度分布（单用户没有额外代价；多用户 --batched-session
# 时温度>0 的请求会退回普通批量解码，不再走投机）。SAMP_EXACT=0 可关。
if [ "${SAMP_EXACT:-1}" = "1" ]; then MTP_EXACT="--mtp-exact-sampling"; else MTP_EXACT=""; fi

cd "$DIR"
mkdir -p "$KV" "$(dirname "$LOG")"

if [ ! -f "$MODEL" ]; then echo "模型不存在: $DIR/$MODEL"; exit 1; fi
if [ ! -f "$VISION" ]; then echo "视觉编码器不存在: $VISION"; exit 1; fi

if nc -z -G 1 127.0.0.1 "$PORT" >/dev/null 2>&1; then
  echo "端口 $PORT 已有实例在跑；先停掉旧实例再启动。"
  pgrep -fl ds4-server || true
  exit 1
fi

# ---- M1 Ultra 调优开关（与代码内注入的默认值等价，保留以便显式覆盖）----
export DS4_QWEN4_MOE_MM_ORDER=1          # 专家优先排布，权重跨 tile 驻留 L2
export DS4_QWEN4_MOE_MM_SPECIALIZE=1     # MoE GEMM 专用内核
export DS4_QWEN4_MOE_MV_SPECIALIZE=1     # MoE GEMV 专用内核
export DS4_QWEN4_MOE_MV_NSG=16           # Q2_K / MXFP4 十六组
export DS4_QWEN4_MOE_MV_NR=2             # 每次两行
export DS4_QWEN4_Q8_PREFILL_UNPACK=1     # Q8 预填解包成 F16，走快 GEMM
export DS4_QWEN4_Q4K_MID_NR=1
export DS4_QWEN4_MOE_MID_TILES=24
export DS4_QWEN4_MOE_DOWN_TILES=8
export DS4_QWEN4_MOE_DOWN_NT=2
export DS4_QWEN4_MOE_TAILS=1
export DS4_QWEN4_MOE_HALF_OPERAND=1      # resident 预填半精度操作数（需本地补丁）
export DS4_QWEN4_HC_MIX_PREFETCH=1
export DS4_QWEN4_HC_NORM_REUSE=1
export DS4_QWEN4_GDN_PREFILL_NSG=16      # 实际被夹到 8，写 16 等价
export DS4_QWEN4_GDN_FRONT_THREADS=512   # 不要设 1024：输出会乱码
export DS4_QWEN4_ATTN_MERGE_WIDE=1
export DS4_QWEN4_IDX_SCORE_VEC=1
export DS4_QWEN4_IDX_PREFILTER=1
export DS4_QWEN4_PREFILL_REUSE=1
export DS4_QWEN4_FLUSH_LAYER=6
# ---------------------------------------------------------------------

EXTRA=""
if [ "${DS4_TRACE:-0}" = "1" ]; then EXTRA="--trace $HOME/Library/Logs/ds4-trace.log"; fi
if [ "${DS4_NO_MTP_SPEC:-0}" = "1" ]; then export DS4_MTP_SPEC_DISABLE=1; fi

echo "日志: $LOG"
echo "模型: $MODEL"
echo "监听: $HOST:$PORT   ctx=$CTX"
echo "KV:   $KV (${KV_MB} MB)"
echo "采样默认: temp=$SAMP_TEMP top_p=$SAMP_TOP_P top_k=$SAMP_TOP_K min_p=$SAMP_MIN_P exact=${SAMP_EXACT:-1}"

exec ./ds4-server \
  -m "$MODEL" \
  --vision "$VISION" \
  --host "$HOST" \
  --port "$PORT" \
  --ctx "$CTX" \
  --kv-disk-dir "$KV" \
  --kv-disk-space-mb "$KV_MB" \
  -t 16 \
  --warm-weights \
  --mtp \
  $MTP_EXACT \
  --temp "$SAMP_TEMP" \
  --top-p "$SAMP_TOP_P" \
  --top-k "$SAMP_TOP_K" \
  --min-p "$SAMP_MIN_P" \
  $EXTRA \
  >>"$LOG" 2>&1
