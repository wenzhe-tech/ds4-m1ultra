# ds4 · M1 Ultra 上的 Qwen3.8 Flash 优化记录

本仓库是 [antirez/ds4](https://github.com/antirez/ds4) 的个人分支，基线提交 `b3f5018`。
只做一件事：**把 Qwen3.8 Flash Next Q4 在 Apple M1 Ultra / Metal 上的推理速度调上去**。
上游代码、提交历史与 MIT 许可（`LICENSE`）均保留，`origin` 仍指向上游以便继续合并更新。

> 仓库里**不含任何模型权重**（文本模型、视觉模型都没有），需要自行准备 GGUF。

---

## 1. 优化前 / 优化后怎么定义

| | 含义 |
|---|---|
| **优化前（stock）** | 干净的 `b3f5018`，不带任何本地补丁，不设任何 `DS4_QWEN4_*` 环境变量 |
| **优化后** | 本分支 `main`，含 M1 Ultra 默认值注入 + Metal 内核改动 + 服务端采样默认值 |

测试环境：Mac Studio（Mac13,2）、Apple M1 Ultra 48 核 GPU / 128 GiB 统一内存、macOS arm64、
`Qwen3.8-Flash-Next-Q4.gguf`（约 165 GiB，常驻权重 69.73 GiB）。
统一用 `--metal -t 16 --warm-weights`、`--temp 0`（贪心），A/B 同场次交错重复取中位数。

**噪声地板**：同配置重复差 0.07 t/s，跨场次漂移约 0.5 t/s，**小于 1% 的差异一律不算数**。
任何"提速"先过输出逐字节校验（与 stock 比对），再看 t/s —— 这条纪律下面会解释为什么重要。

## 2. 核心指标（ctx 2048，同场次 A/B）

| 指标 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| prefill | 443.5 t/s | **529.3 t/s** | **+19.3%** |
| decode | 31.0 t/s | **36.4 t/s** | **+17.4%** |
| MTP 投机解码 | 38.7 t/s | **46.0 t/s** | **+18.9%** |

后续把路由层的 F32 矩阵乘改道到 `kernel_mul_mm_f32_f32_full`，prefill 中位数进一步提升到
**549.9 t/s**（另一场次测得 +5.6%，不与上表直接并列）。prefill 与 MTP 已达标，
**decode 目标 40 t/s 未达成**，原因见第 5 节。

## 3. 长上下文：真正的问题不是"新版变慢了"

新目录在 256K 上下文下 decode 掉到 20 t/s 出头，旧目录却能保持 28 t/s 以上，一度以为是新版本退化。
实测 ctx 曲线（`ds4-bench` 稳态口径）：

| ctx | 优化前 | 优化后 | 变化 |
|---:|---:|---:|---:|
| 2 048 | 29.56 t/s | **36.40 t/s** | +23.1% |
| 65 536 | 26.45 t/s | **34.41 t/s** | +30.1% |
| 262 144 | 20.34 t/s | **28.29 t/s** | +39.1% |

**结论**：新旧版本的衰减绝对量几乎相同（约 -8~9 t/s），差的是基数。旧版本地补丁
`qwen4_tuned_like_m3_ultra()` 把 M3 Ultra 专属的 7 处默认值也给了 M1 Ultra；新目录是纯净上游，
回落到通用路径，所以整体更慢。本分支在 `ds4_gpu_detect_metal4_features()` 里把 21 个调优默认值
重新注入给 M1 Ultra，问题消失。

之前一直没发现，是因为基准测试始终套着 `env_best.txt` 里的 21 个开关，把差异掩盖了。

## 4. 256K 上下文下的 MTP（端到端 A/B）

`ds4-bench` 不支持 `--mtp`，因此两条腿都走 `ds4 --prompt-file`（261 978 token 提示词，
`--ctx 262144 -n 192 --temp 0`）：

| 配置 | prefill | decode | 接受率 |
|---|---:|---:|---:|
| 无 MTP | 508.0 t/s | **31.07 t/s** | — |
| `--mtp` | 544.3 t/s | **36.63 t/s** | 81.3%（91 cycles / 74） |

**+17.9%**，与短上下文的 +18.7%（30.61 → 36.33 t/s，接受率 75.2%）几乎一致：
收益与上下文长度基本无关 —— 2-row verify 省下的权重读取，被第二行 attention/indexer 的开销抵掉了。

质量方面：`--temp 0` 时逐字节无损（6 次运行 MD5 一致），因为只在
`draft == target argmax` 时才接受；`temp > 0` 时必须加 `--mtp-exact-sampling`
才能严格保持温度分布，否则约 80% 的位置会偷偷变成贪心。

## 5. 做了哪些改动

1. **M1 Ultra 默认值注入**（收益最大的一项）：在设备探测阶段对 M1 Ultra 设置 21 个
   `DS4_QWEN4_*` 开关；用户显式设的环境变量优先，`DS4_QWEN4_NO_M1_ULTRA_DEFAULTS=1` 可整体关闭。
2. **路由 GEMM 改道**：`qwen4_gemv_rows()` 中 F32 权重、大 batch、维度对齐时走
   `kernel_mul_mm_f32_f32_full`，prefill +5.6%；`DS4_QWEN4_NO_ROUTER_MM=1` 可回退。
3. **Metal 内核微调**：`hc_norm`/`hc_combine_norm` 线程组 128→512；resident MoE 半精度操作数
   接线（`DS4_QWEN4_MOE_HALF_OPERAND=1`，输出逐位一致）；两个 decode 融合可独立关闭以便 A/B；
   `DS4_QWEN4_MM_STATS=1` 输出 GEMM 形状与 MoE FLOPs 直方图。
4. **服务端采样默认值**：`ds4-server` 新增 `--temp/--top-p/--top-k/--min-p`，只作用于
   客户端未传的字段（显式请求值永远优先），启动时打印生效值。
5. **KV 缓存精度**（实测，非改动）：K/V 为 **FP16**，indexer key 为 FP32，block key 为 FP16；
   13 层带 KV，262144 ctx 实测 `KV 8.33 GiB (raw 6.50 + compressed 1.83)`。没有 q8/q4 量化。

`DS4_QWEN4_ICB` 回放层代码保留但**默认关闭**：`MTLIndirectCommandBuffer` 把绑定偏移截断到
32 位（5 GiB 实际读到 1 GiB），与单块 69.73 GiB 模型映射根本冲突，启用会输出乱码，不要开。

## 6. 定因与「别再试」清单

- **GEMM 已到硬件天花板**：最小程序实测 M1 Ultra 的 `simdgroup_multiply_accumulate`
  峰值仅 **13.8 TFLOP/s**（M1 无矩阵单元，FMA 模拟），`kernel_mul_mm_f16_f32` 已跑
  12.9 TFLOP/s = **93%**。64×64 tile 变体理论上限 7%，实测为 0，不用做。
- **decode 40 t/s 不是带宽问题**：把 MoE 权重读取和反量化**整个删掉**，decode 也只有
  40.08 t/s（破坏性实验）。实测 27.5 ms/步 = GPU 19.0 + CPU 编码 3.7（自回归无法重叠）
  \+ commit→启动 4.0。f16 权重（3.56 倍数据量、165 GiB 装不下）是死路。
  剩下的空间只有 ICB 预录制（上限约 41）或层内融合（40~42），都是数小时级改造且会破坏逐字节一致。
- **`DS4_QWEN4_GDN_FRONT_THREADS=1024` 让 decode +17%，但输出乱码**；
  384/512/640/768 逐位一致，取 512。这是"先校验再提速"纪律的由来。
- **MoE mid 的 tile 几何别动**：NT=1/2/8 分别 -12/-4/-3%；运行时跳过尾部 MMA -43%
  （破坏展开致累加器 spill）；64 行 tile / 256 线程 -4.5%。44% 效率是 Q4_K 反量化
  + barrier 的结构成本，不是 tile 参数问题。
- 其余证伪项：`MTP_DEPTH=3` 手工锁死、`F16_MM_DECODE=1`、`FLUSH_LAYER=1/96`、
  ICB 分块绑定的三种绕法（全部 OOM，因上游刻意用极少数重叠大视图 + 一次性 residency）。

## 7. 复现

```sh
make clean && make -j4        # Makefile 无头文件依赖追踪，更新后必须 clean
./ds4 -m ds4flash.gguf --metal -t 16 --warm-weights --ctx 262144 \
      --mtp --temp 0

# 服务端：官方思考模式采样默认值 + 严格保分布的投机解码
./ds4-server -m ds4flash.gguf --metal -t 16 --ctx 262144 \
      --mtp --mtp-exact-sampling \
      --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
```

从上游更新后，需要重新应用本地默认值补丁，否则 decode 会从 36.4 掉回 29.6，
262K 时从 28.3 掉到 20.3。补丁与幂等脚本保存在本机，未纳入本仓库。

## 8. 许可

上游 [antirez/ds4](https://github.com/antirez/ds4) 为 MIT，`LICENSE` 原样保留。
本分支的本地改动同样按 MIT 提供。
