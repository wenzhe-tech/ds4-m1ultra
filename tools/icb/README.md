# ICB 探测：这台机器能不能用间接命令缓冲

## 一句话结论（2026-11-05 修订）

**ICB 本身能用，但绑定偏移超过 4 GiB 会被截断成 32 位，所以 ds4 的 decode 回放仍然不可用。**

三条事实，按重要性排：

1. **偏移 ≥ 4 GiB 会折回**：5 GiB 的绑定实际读到 1 GiB 处（5 − 4）。2 GiB 正常。
   普通分配和 mmap 无拷贝缓冲都一样，与缓冲类型无关。
   ds4 把 69.73 GiB 的模型映射成**一个**缓冲，权重偏移从 1.8 GiB 一路到 69 GiB，
   **大部分 decode 命令都落在 4 GiB 以上** —— 这就是回放输出乱码的根因。
2. **ICB 内命令默认并发**，依赖链必须每条 `setBarrier`，否则后一条不执行。
3. 之前记录的「读输入缓冲恒为 0」是**错的**：那是我探测程序少写 `setBarrier`。

真实的驱动行为是：

| 场景 | 结果 |
|---|---|
| 单条间接命令（读 + 写） | ✅ 正确，10 种配置全过（serial/concurrent、inherit、private、bind 2/4/8、无 useResource） |
| 参数放在缓冲区偏移处（index 0 / offset 256） | ✅ 正确 |
| 两条有依赖的命令**不加**屏障 | ❌ 第二条**根本不执行**（不是读错，是没跑） |
| 两条有依赖的命令**加** `setBarrier` | ✅ 正确 |

也就是说：ICB 内的命令默认并发，依赖链必须每条显式 `setBarrier`。

## 怎么用

```sh
./tools/icb/run_probe.sh
```

退出码：`0` = ICB 可用，`1` = 坏了，`2` = 探测程序自己失败（与 ICB 无关）。

换 macOS 大版本、换机器、或怀疑驱动升级后行为变了，都跑一次这个。

## 探测覆盖什么

- 先跑**直连对照**（同样的内核和缓冲区，宿主编码）：失败说明是探测程序的问题，
  与 ICB 无关，直接判 `2`，避免再次得出错误结论。
- 再跑 10 种 ICB 配置，逐个 PASS/FAIL。
- 单独覆盖「参数经缓冲区偏移传入」——回放层就是这么喂参数的。
- 最后跑依赖链：无屏障（预期失败，单独计数不计入判决）与有屏障（必须通过）。

## 定位过程（可复现）

`DS4_QWEN4_ICB_ONLY=<内核> DS4_QWEN4_ICB_CAP=<n>` 逐条二分：

| 内核 | CAP=1..4 | CAP=5 | 说明 |
|---|---|---|---|
| `kernel_qwen4_hc_norm_f16` | ✅ 正确 | ❌ 乱码 | 第 5 条对应权重偏移 **4.97 GiB**，第一条越过 4 GiB |
| `kernel_qwen4_attn_prep` | ✅ 正确（测到 CAP=6） | ✅ | 权重在低偏移区 |
| `kernel_qwen4_gdn_out` | ❌ 任何条数都乱码 | ❌ | 专家权重从一开始就在 4 GiB 以上 |

录制/回放两侧转储的命令参数（pso、grid、threadgroup、5 个绑定、偏移、参数长度）
**完全一致**，所以不是参数问题；同一模式在探测程序里 2 GiB 正常、5 GiB 折回，
这就把范围钉死到偏移位宽上。

## 绕法：试过，被常驻内存挡住

要让偏移小于 4 GiB，只能让每一处的 `inner_offset` 变小，也就是**把 70 GiB 的单块映射
拆成多个缓冲**。三种方式都试了，全部因 `Insufficient Memory` 失败：

| 尝试 | 结果 |
|---|---|
| `ds4_gpu_wrap_model_range` 对大偏移改走 `ds4_gpu_wrap_model_exact_range`（每个权重范围一个视图） | prefill 直接 OOM |
| 同上但按固定 **1 GiB 分块**缓存（至多 70 个视图） | prefill OOM；把 `DS4_METAL_EXACT_VIEW_CACHE_GIB` 提到 128 仍 OOM |
| 不改代码，直接 `DS4_METAL_MODEL_VIEW_MAX_GIB=3` / `=2` | **启动阶段** residency 就 OOM |

原因在 `ds4_gpu_create_model_views()`：它刻意把模型包成**极少数的重叠大视图**
（注释写明「tiny number of page-aligned views created once」），并对每个视图
`addAllocation` 进 `g_model_residency_set`。视图一多，常驻申请就超过预算。
这不是调参能解决的，是「单块常驻映射」这个设计本身与 4 GiB 偏移上限冲突。

**结论：ICB 回放在本机被映射设计挡住。** 要解锁只能改加载路径，把模型一次性映射成
多个 < 4 GiB 的常驻视图 —— 那会重写 `ds4_gpu_create_model_views()` 与 residency 逻辑，
影响 --warm-weights、streaming 和所有既有路径，风险远高于收益，暂不做。

## ds4 回放层的状态（重要）

`ds4_metal.m` 里的回放层（`DS4_QWEN4_ICB=1` 才启用，默认关闭）**目前仍然输出错误**。
本轮在它里面修掉了三个真实缺陷：

1. 录制时没有 `setBarrier` → 依赖链被打乱（就是上面那条驱动行为）。
2. 命令无法录制时（绑定数超限等）录制继续 → 索引整体错位，回放可能执行到别的命令。
   现在一旦遇到不可录制就 `capped`，之后一律正常编码。
3. `QWEN4_ICB_MAX_BINDS` 8 → 16（`attn_prep` 要 15、`gdn_front` 12、`multi_gemv` 9）。

但即使只回放 `kernel_qwen4_hc_norm_f16` 一个内核、每条命令独立一个 pass，
输出仍然是乱码（`DS4_QWEN4_ICB_ALSO=1` 双跑同样乱，说明是回放**写坏了数据**）。
而同样的模式在探测程序里是对的，所以剩下的 bug 在回放层的集成，
不在驱动。已知排查方向：ICB pass 与相邻 pass 的次序/可见性。

**结论：`DS4_QWEN4_ICB` 保持不设。** 回放层代码留着，等这条查清再启用。
它是唯一能削掉每 token 7.7ms dispatch 固定开销的路径（目标 decode 40+）。
