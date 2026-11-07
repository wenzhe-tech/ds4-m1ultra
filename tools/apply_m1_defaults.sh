#!/bin/sh
# 把 M1 Ultra 调优默认值打进源码（幂等：已应用则直接退出）
#
# 背景：上游把一批 Qwen3.8 调优默认值门控在 M3 Ultra / M5 / M6 上，
# M1 Ultra 不在名单里 → decode 36.4 掉到 29.6（256K 时 28.3 掉到 20.3）。
# 补丁在 ds4_gpu_detect_metal4_features() 里注入 21 个环境变量默认值，
# 用户显式设置的变量仍然优先。详见 README.md。
#
# 用法： ./tools/apply_m1_defaults.sh
set -e

DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PATCH=$DIR/patches/m1_ultra_defaults.patch
cd "$DIR"

if [ ! -f "$PATCH" ]; then
    echo "找不到补丁：$PATCH"
    exit 1
fi

if grep -q "ds4_gpu_apply_m1_ultra_defaults" ds4_metal.m 2>/dev/null; then
    echo "已应用（ds4_metal.m 里存在 ds4_gpu_apply_m1_ultra_defaults），无需重打。"
    exit 0
fi

if git apply --check "$PATCH" 2>/dev/null; then
    git apply "$PATCH"
    echo "补丁已应用。"
else
    echo "补丁与当前源码冲突（上游改动了附近代码），尝试 3-way 合并…"
    if git apply --3way "$PATCH" 2>/dev/null; then
        echo "3-way 合并成功，请用 git diff 复核。"
    else
        echo "自动合并失败。请手工处理："
        echo "  1) 在 ds4_metal.m 的 ds4_gpu_detect_metal4_features() 之前加入"
        echo "     ds4_gpu_apply_m1_ultra_defaults() 函数（见补丁文件）"
        echo "  2) 在 snprintf(g_metal_device_name…) 之后调用它"
        exit 1
    fi
fi

echo "编译中…"
make -j4
echo
echo "完成。验证方式：不设任何 DS4_* 环境变量，"
echo "  ./ds4-bench -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt \\"
echo "    --metal --warm-weights -t 16 --ctx-start 2048 --ctx-max 2048 \\"
echo "    --step-incr 2048 --gen-tokens 64"
echo "decode 应约为 36 t/s（未打补丁时约 29.6）。"
