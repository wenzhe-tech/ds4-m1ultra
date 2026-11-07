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

## 7. 参数配置建议

以下都是本机实测 / 查证过源码的结论，不是通用默认值。

### 7.1 采样参数（官方 Qwen3.8 Flash 思考模式）

| 参数 | 官方建议 | ds4 内置默认 | 怎么设置 |
|---|---:|---:|---|
| `temperature` | 1.0 | 1.0 | `--temp 1.0`，或请求里传 |
| `top_p` | 0.95 | **1.0** | `--top-p 0.95` |
| `top_k` | 20 | **0** | `--top-k 20`（**仅 `ds4-server` 有此开关**，CLI 没有） |
| `min_p` | 0.0 | **0.05** | `--min-p 0.0` |
| `presence_penalty` | 0.0 | 不支持 | 传了会被静默忽略，官方值本来就是中性 |
| `repetition_penalty` | 1.0 | 不支持 | 同上 |

`--temp` / `--top-p` / `--top-k` / `--min-p` 在 `ds4-server` 上是**请求默认值**：
客户端没传该字段才生效，显式传的值永远优先；启动时打印生效值，`--trace` 会记录每次请求的实际值。
`--temp 0` 走贪心，以上四个参数全部作废；服务端在工具调用语法内部也强制贪心。
不同 API 入口支持度不同：`/v1/chat/completions` 和 `/v1/completions` 四个都认，
`/v1/messages` 认 `top_k` 不认 `min_p`，`/v1/responses` 两个都不认 —— 统一用 chat/completions 最稳。

### 7.2 MTP 投机解码

| 场景 | 怎么开 | 质量 |
|---|---|---|
| 任意 ctx + `--temp 0` | `--mtp` | **逐字节无损**（实测 6 次运行 MD5 一致） |
| `temp > 0`，要严格保持分布 | `--mtp --mtp-exact-sampling` | 分布级无损 |
| `temp > 0`，只要快 / 更确定 | `--mtp` | 约 80% 位置偷偷变贪心，多样性下降 |
| 多用户 `--batched-session` + `temp > 0` | 加 exact-sampling 会让投机退回普通批量解码 | 需权衡 |

**建议任何上下文长度都开**：2K 实测 +18.7%，256K 实测 +17.9%，收益与 ctx 基本无关，
也不增加 KV 占用。

不要加的参数：`--mtp-draft 7` 对 Qwen3.8 **完全无效**（该模型深度由
`qwen4_spec_depth()` 决定，默认 2、自适应升 3；实测 1 与 7 的 cycle 数、接受率完全相同）；
`--mtp-timing` 在常驻服务里只在**退出时**打印一行累计值，日常没用；
`--trace FILE` 每请求写完整原始 JSON + 渲染 prompt + 生成文本（256K 时约 1.8 MB/请求，
无轮转），只在排障时开；`DS4_QWEN4_MTP_DEPTH=3` 手工锁死在一般文本上反而更慢。

### 7.3 上下文、运行与网络

- `--ctx 262144`：模型 `context_length=262144`，不设 YARN 不外推，写 262144 即可
  （不用刻意写 260000）。`--ctx` 是**总容量**，prompt 上限恒为 `ctx-1`；
  改 ctx 会让 KV 磁盘检查点失效（ctx 是查找键的一部分）。
- `--metal` 在 macOS 上本来就是默认后端，写不写没区别。
- `-t 16 --warm-weights`：前者给 host 侧辅助线程，后者预热权重减少首个请求卡顿。
- `--kv-disk-dir DIR --kv-disk-space-mb N`：磁盘 KV 检查点；N 是预算不是预分配。
- `--host 0.0.0.0 --port 8000` 可让局域网访问；服务端**无认证、无 TLS**，
  公网请走 Tailscale、cloudflared 或反代 + basic auth，不要直接端口映射。
- 仓库不含任何模型权重，需自备文本 GGUF；视觉功能另需 `--vision FILE` 指向 mmproj。

### 7.4 环境变量

M1 Ultra 的 21 个 `DS4_QWEN4_*` 调优默认值在 `ds4_gpu_detect_metal4_features()` 里
**按设备名自动注入**，无需手动设置；显式设置的变量优先，
`DS4_QWEN4_NO_M1_ULTRA_DEFAULTS=1` 可整体关闭。完整开关清单见 `tools/env_best.txt`
（非 M1 Ultra 机型可参考，但未验证）。

两个禁忌：`DS4_QWEN4_GDN_FRONT_THREADS=1024` 会输出乱码（用 512）；
`DS4_QWEN4_ICB=1` 会输出乱码（32 位绑定偏移截断），默认关闭，别开。

## 8. 仓库里有什么 / 别人怎么用

`clone` 默认分支 `main` 拿到的**就是优化后的源码**，但要满足几个条件：

1. **必须自己编译**：仓库不含二进制，`make clean && make -j4`
   （Makefile 无头文件依赖追踪，更新后只跑 `make` 会留下按旧头文件编译的 `.o`）。
2. **自动优化只对 M1 Ultra 生效**：默认值注入按 GPU 设备名判断，其他机型走上游路径，
   不会自动拿到这些调优值。
3. **模型自备**：文本 GGUF 与视觉 mmproj 均不在仓库内。
4. **服务端采样默认值不是自动的**：`ds4-server` 若不传那四个 flag，仍是上游的
   `1.0 / 1.0 / 0 / 0.05`；按 7.1 显式传，或直接跑 `./tools/server-tuned.sh`。

目录说明：

| 路径 | 内容 |
|---|---|
| `patches/` | 4 个 format-patch：`m1_ultra_defaults`（21 个默认值）、`ds4_local_optim_20261105`（Metal 内核）、`server_sampling_defaults`（服务端四个参数）、`local_all_b3f5018`（全量快照） |
| `tools/` | `apply_m1_defaults.sh`（幂等补默认值）、`update_ds4.sh`（幂等快进上游 + 全量重编译）、`server-tuned.sh`（一键启动器）、`env_best.txt`（21 个开关）、`icb/`（ICB 驱动探测器） |

从上游更新后的固定流程：

```sh
./tools/update_ds4.sh          # 快进 upstream，自动备份本地改动、make clean && make -j4
./tools/apply_m1_defaults.sh   # 重新注入 M1 Ultra 默认值（幂等，已应用则跳过）
```

漏掉第二步会静默变慢：decode 36.4 → 29.6，256K 时 28.3 → 20.3。

一键启动（等价 7.1 + 7.2 + 7.3 的推荐组合）：

```sh
./tools/server-tuned.sh        # 可用 CTX/PORT/SAMP_TOP_P/SAMP_EXACT 等环境变量覆盖
```

单次运行的验证命令：

```sh
./ds4 -m ds4flash.gguf --metal -t 16 --warm-weights --ctx 262144 --mtp --temp 0
```

## 9. 许可

上游 [antirez/ds4](https://github.com/antirez/ds4) 为 MIT，`LICENSE` 原样保留。
本分支的本地改动同样按 MIT 提供。
