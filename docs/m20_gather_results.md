# M20：decode KV gather 与 K/V copy pipeline

**状态：实现与正确性验证完成。用户声明空闲后的五次模型复测已完成；ABBA 与两组 fresh pair 均未显示 pipeline 加速。pipeline 保持 opt-in，默认关闭。**

工作区：`NanoVLLM-Ubuntu:/opt/nano-vllm`，分支 `codex/m19-selector-cuda-graph`。测量时以 `adce2a5` 为基线，M19/M20 源码尚未提交；实现、测试和报告现一同纳入版本管理。实验 JSON 的 `git_head` 只标识测量基线，不能单凭它重建完整实验源码。

> 先前一轮测量期间有其他 GPU 或高负载任务，负载强度未知；其结果保留为历史观察。最新五次模型运行是在用户声明电脑空闲后完成，但这不等于实验室级隔离，差异仍不能证明因果或泛化加速。pipeline 默认保持关闭。

## 实现与验证

最终候选是可选的 K/V copy pipeline：发起 K 的 H2D 后，在 CPU 上 gather V，再发起 V 的 H2D；复用原有 pinned staging 和 stream，不增加 staging buffer。CLI 使用 `--kv-pipeline`，要求 `--index-select 1`；`sparse_decode_kv_pipeline` 默认值为 `false`。此前整块 `decode_block_gather` 路径已撤回，18 项 block-path 测试不再代表当前实现。

五次模型运行均使用缓存中的 Qwen3-4B snapshot、32K 输入、Top-32、36 层、8 个 CPU 线程、`index_select=1`，关闭 selector CUDA graph。ABBA 运行生成 48 tokens；两组 fresh pair 各生成 32 tokens。所有运行完成，无 OOM。

模型 snapshot 为 `1cfa9a7208912126459214e8b04321603b3df60c`；BF16、YaRN、4096-token prefill chunks、query segments=1。使用单一重复文本 prompt；输出等价验证不是广泛的长上下文质量或服务负载评测。

- Pair 1（serial→pipeline）两臂的 `generated_token_ids` 与 36×31 的 `payload.ids_history_by_layer` 完全相同。
- Pair 2（pipeline→serial）两臂的相同字段也完全相同。
- ABBA 输出前 32 tokens 和前 31 步逐层 IDs 与 pair 1 fresh serial 完全相同。
- 父代理确认六个 targeted test files 共 40 项通过（3.13 秒），包括当前 pipeline 与 overlap-analysis 回归；已撤回的 block-path 18 项不计入。Python 编译检查和 `git diff --check` 通过。

## 用户声明空闲后的模型复测

五次均为独立进程：Qwen3-4B pinned snapshot、32K 输入、Top-32、36 层、8 CPU 线程、`index_select=1`，关闭 selector CUDA graph、profiler 和 finite checks。ABBA 生成 48 tokens；两组 fresh pair 各生成 32 tokens。所有进程完成，无 OOM。

`drop4` 去掉前四个 decode steps；完整 mean/median 包含 first/setup step。pair 百分比为 pipeline 相对 serial 的描述值。

| 比较及运行顺序 | Arm | Drop4 mean / median (ms) | 完整 mean / median (ms) | 首步 (ms) | Decode steps 总和 (ms) |
|---|---|---:|---:|---:|---:|
| ABBA：逐步 S/P/P/S | Serial，n=21 | 138.363 / 138.861 | — | — | — |
|  | Pipeline，n=22 | 144.387 / 144.592 | — | — | — |
| Pair 1：serial→pipeline | Serial | 146.463 / 142.687 | 166.474 / 142.893 | 763.326 | 5,160.683 |
|  | Pipeline | 148.934 / 146.504 | 172.817 / 149.778 | 864.373 | 5,357.314 |
| Pair 2：pipeline→serial | Pipeline | 151.187 / 147.143 | 171.183 / 148.698 | 762.685 | 5,306.669 |
|  | Serial | 147.535 / 142.770 | 171.030 / 142.770 | 892.334 | 5,301.939 |

三个比较均为负向描述值。ABBA 的 pipeline decode mean / median 高 4.35% / 4.13%；ABBA 两臂位于不同 token positions，不是 same-query 配对。Pair 1 pipeline 的 drop4 mean / median 高 1.69% / 2.67%，完整 mean / median 高 3.81% / 4.82%。Pair 2 分别高 2.48% / 3.06% 和 0.09% / 4.15%。这组结果未显示加速，也不足以证明 pipeline 导致变慢。

ABBA 的 36 层 CPU gather mean / median：serial 25.898 / 25.887 ms，pipeline 28.382 / 28.378 ms。全程 ABBA 的 drop4 decode mean / median 为 141.445 / 139.875 ms；含 first/setup 的完整 mean / median 为 154.855 / 140.151 ms，first step 763.471 ms，47 steps 总和 7,278.162 ms。

| 运行 | 最后一步 36 层 CPU gather (ms) | 最后一步 h2d_pack (ms) | Host RSS before→after (GiB) | Host available after (GiB) | GPU allocated / reserved / peak (GiB) |
|---|---:|---:|---:|---:|---:|
| ABBA mixed | 27.687 | 9.301 | 1.837→7.391 | 16.884 | 8.620 / 9.197 / 8.941 |
| Pair 1 serial | 27.397 | 8.397 | 1.838→7.407 | 16.882 | 8.620 / 9.197 / 8.941 |
| Pair 1 pipeline | 28.727 | 11.357 | 1.837→7.330 | 16.937 | 8.620 / 9.197 / 8.941 |
| Pair 2 pipeline | 29.104 | 10.763 | 1.837→7.344 | 16.912 | 8.620 / 9.197 / 8.941 |
| Pair 2 serial | 27.432 | 13.589 | 1.837→7.316 | 16.977 | 8.620 / 9.197 / 8.941 |

两组 fresh pair 的 `generated_token_ids` 和 36×31 的 `payload.ids_history_by_layer` 均完全一致；ABBA 前 32 tokens 和前 31 步逐层 IDs 与 Pair 1 serial 完全一致。last-step stage 是诊断值，不是全程或独占成本；pipeline 的 `cuda_h2d_stage_event_spans_cpu_v_gather=true`，不能把 CUDA event 当纯 memcpy 时间。

## 单步 Nsight overlap 观察

父代理对一个 profiled step 按 CUDA API `correlationId` 配对后，确认 36/36 层的 K H2D 与该层 V CPU gather 时间区间实际相交。K H2D device duration 合计 17.527864 ms，V gather 合计 14.874729 ms，区间交集合计 13.722485 ms。例：layer 10 的 K H2D 为 62.893958–63.380634 ms（0.486676 ms），V gather 为 62.867964–63.278467 ms（0.410503 ms），交集 0.384509 ms。

这只证明一个 profiled step 中存在实际重叠；13.722 ms 交集不能解释为节省的 TPOT，也未证明 CPU/GPU 带宽争用或完整因果机制。无端到端 Nsight 性能结论。

## 历史：背景负载未隔离的模型观察

以下均为未经背景负载隔离的数据。`drop4` 去掉前四个 decode steps；“完整”均值和中位数包含 first/setup step。pair 的百分比是 pipeline 相对 serial 的数值变化，仅为描述值。

| Fresh pair 与运行顺序 | Arm | Drop4 mean / median (ms) | 完整 mean / median (ms) | 首步 (ms) | Decode steps 总和 (ms) |
|---|---|---:|---:|---:|---:|
| Pair 1，serial→pipeline | Serial | 166.870 / 156.742 | 190.732 / 156.869 | 915.178 | 5,912.694 |
|  | Pipeline | 249.930 / 251.403 | 278.551 / 255.267 | 1,115.067 | 8,635.066 |
| Pair 2，pipeline→serial | Pipeline | 264.132 / 264.184 | 293.644 / 265.424 | 942.460 | 9,102.973 |
|  | Serial | 298.606 / 285.381 | 353.691 / 287.814 | 1,896.671 | 10,964.416 |

Pair 1 是负向观察：pipeline 的 drop4 mean / median 高 49.78% / 60.39%；完整 mean / median 高 46.04% / 62.73%。Pair 2 是正向观察：pipeline 的 drop4 mean / median 低 11.54% / 7.43%；完整 mean / median 低 16.98% / 7.78%。两组方向相反，且期间背景负载未隔离，不能把任一组当作本次改动的效果估计。

### ABBA

ABBA 在 4 个 pipeline warmup steps 后按 serial / pipeline / pipeline / serial 逐步切换。两臂位于不同 token positions，不是 same-query 配对；下表只比较分到各模式的不同 decode steps。

| 模式 | steps | Decode mean / median (ms) | 36 层 CPU gather mean / median (ms) |
|---|---:|---:|---:|
| Serial | 21 | 139.071 / 136.410 | 26.779 / 26.649 |
| Pipeline | 22 | 141.180 / 139.355 | 28.322 / 28.139 |

整次 ABBA 的 drop4 decode mean / median 为 140.150 / 138.864 ms；包含首步/setup 的完整 mean / median 为 186.163 / 139.172 ms，first step 为 2,298.753 ms，47 个 decode steps 总和为 8,749.672 ms。该 ABBA 样本中 pipeline 模式较慢；不同 token positions 和背景负载都不支持因果解释。

### 最后一步的 host stage 与内存

`last_decode_stage_ms_36_layers` 是每个进程最后一个 decode step 的诊断值，不代表整轮 gather/H2D 的独占成本。pipeline 模式的 `cuda_h2d_stage_event_spans_cpu_v_gather=true`，因此该 CUDA event 区间不能解释为纯 memcpy 时间。

| 运行 | CPU gather (ms) | h2d_pack (ms) | CUDA H2D event 跨 CPU V gather |
|---|---:|---:|---|
| ABBA 混合步 | 28.565 | 10.542 | 是 |
| Pair 1 serial | 35.376 | 10.953 | 否 |
| Pair 1 pipeline | 35.578 | 16.557 | 是 |
| Pair 2 pipeline | 36.832 | 13.137 | 是 |
| Pair 2 serial | 49.612 | 20.462 | 否 |

| 运行 | Host RSS before→after (GiB) | Host available after (GiB) | GPU allocated / reserved / peak (GiB) |
|---|---:|---:|---:|
| ABBA | 1.852→7.473 | 16.487 | 8.620 / 9.197 / 8.941 |
| Pair 1 serial | 1.838→7.374 | 16.809 | 8.620 / 9.197 / 8.941 |
| Pair 1 pipeline | 1.837→7.397 | 15.084 | 8.620 / 9.197 / 8.941 |
| Pair 2 pipeline | 1.838→7.424 | 14.484 | 8.620 / 9.197 / 8.941 |
| Pair 2 serial | 1.838→7.501 | 16.290 | 8.620 / 9.197 / 8.941 |

## CPU gather 与 gather+H2D 微测

这些也是背景负载未隔离的探索性数据，不是 TPOT。两项使用 synthetic KV source，不能代表真实逐层 source 或完整 decode。

`benchmark_m20_gather.py` 对同一个 pageable BF16 K/V 源 `[512,64,8,128]`、pinned stage `[2048,8,128]`，轮播 M19 记录的 1,116 组真实 layer-step IDs；K/V 输出逐元素一致，stage pinned 属性和地址保持不变。每个方法每线程档三轮，单次 gather 复制 8 MiB；合成源在全部 IDs 间复用。

| CPU 线程 | Flat token-index (µs) | Whole-block dim0 (µs) | Block / flat |
|---:|---:|---:|---:|
| 1 | 871.77 | 841.76 | 0.966× |
| 2 | 632.35 | 556.24 | 0.880× |
| 4 | 378.28 | 662.27 | 1.750× |
| 8 | 315.31 | 634.00 | 2.010× |

whole-block 在 1/2 线程档观测较快，在 4/8 线程档观测较慢；因此运行时 block 路径已撤回，不从该微测推导稳定性能结论。

`benchmark_m20_gather_h2d.py` 使用 8 线程、64 组 M19 IDs、每次 8 MiB K+V payload，测量 CPU gather 到 CUDA copy 完成的墙钟时间（不含 attention/model）。每份 JSON 记录了每轮全部四种方案的中位数：

| 轮次文件 | Serial (ms) | K/V overlap (ms) | Chunk-8 (ms) | Chunk-16 (ms) |
|---|---:|---:|---:|---:|
| `m20_gather_h2d_32k.json` | 1.417632 | 1.194507 (-15.74%) | 1.314334 (-7.29%) | 1.191807 (-15.93%) |
| `m20_gather_h2d_32k_2.json` | 1.398061 | 1.188064 (-15.02%) | 1.232343 (-11.85%) | 1.180163 (-15.56%) |

两份合成微测给出 K/V overlap 的正向探索信号；Chunk-16 只略低于 K/V overlap。背景负载、单一重复 source 和非模型工作负载意味着不能宣称稳定 15% 搬运收益；当前候选保留简单的 K/V overlap，不采用 chunk 变体，pipeline 仍默认关闭。

## 原始结果与复跑命令

原始文件：

- 空闲复测模型：`bench_logs/m20_quiet_decode_abba_32k.{json,log}`、`bench_logs/m20_quiet_decode_ab_1_0.{json,log}`、`bench_logs/m20_quiet_decode_ab_1_1.{json,log}`、`bench_logs/m20_quiet_decode_ab_2_1.{json,log}`、`bench_logs/m20_quiet_decode_ab_2_0.{json,log}`。
- 背景负载模型历史：`bench_logs/m20_decode_abba_32k.{json,log}`、`bench_logs/m20_decode_ab_1_0.{json,log}`、`bench_logs/m20_decode_ab_1_1.{json,log}`、`bench_logs/m20_decode_ab_2_1.{json,log}`、`bench_logs/m20_decode_ab_2_0.{json,log}`。
- gather 微测：`bench_logs/m20_gather_micro_32k.json`；输入 IDs 来自 `bench_logs/m19_decode_raw_ab_3_0.json`。
- gather+H2D：`bench_logs/m20_gather_h2d_32k.{json,log}`、`bench_logs/m20_gather_h2d_32k_2.{json,log}`。
- Nsight：`bench_logs/m20_quiet_decode_pipeline_32k_step4.{json,log,nsys-rep,sqlite,stats.log}`；`bench_logs/m20_quiet_overlap_32k_step4.json`。Windows 复制件 `NsightReports/M20` 的 `.nsys-rep` SHA256 为 `A7E73EC8F1112E92A7D5B023595DAEF8B60C826F732D98221D82D05D2DD959D1`。

五条模型命令各自启动独立进程；以下是本轮实际执行的命令，`m20_quiet_` 输出文件现在已存在。如需再次复测，先统一换成新的输出前缀（例如 `m20_quiet_repeat_`），不要覆盖本轮或背景负载记录：

```powershell
.\scripts\wsl-nanovllm.ps1 -Command "python benchmarks/benchmark_m15_decode.py --seq-len 32768 --gen-tokens 48 --index-select 1 --cpu-threads 8 --kv-pipeline --gather-abba --record-selected-ids --output bench_logs/m20_quiet_decode_abba_32k.json > bench_logs/m20_quiet_decode_abba_32k.log 2>&1"
.\scripts\wsl-nanovllm.ps1 -Command "python benchmarks/benchmark_m15_decode.py --seq-len 32768 --gen-tokens 32 --index-select 1 --cpu-threads 8 --record-selected-ids --output bench_logs/m20_quiet_decode_ab_1_0.json > bench_logs/m20_quiet_decode_ab_1_0.log 2>&1"
.\scripts\wsl-nanovllm.ps1 -Command "python benchmarks/benchmark_m15_decode.py --seq-len 32768 --gen-tokens 32 --index-select 1 --cpu-threads 8 --kv-pipeline --record-selected-ids --output bench_logs/m20_quiet_decode_ab_1_1.json > bench_logs/m20_quiet_decode_ab_1_1.log 2>&1"
.\scripts\wsl-nanovllm.ps1 -Command "python benchmarks/benchmark_m15_decode.py --seq-len 32768 --gen-tokens 32 --index-select 1 --cpu-threads 8 --kv-pipeline --record-selected-ids --output bench_logs/m20_quiet_decode_ab_2_1.json > bench_logs/m20_quiet_decode_ab_2_1.log 2>&1"
.\scripts\wsl-nanovllm.ps1 -Command "python benchmarks/benchmark_m15_decode.py --seq-len 32768 --gen-tokens 32 --index-select 1 --cpu-threads 8 --record-selected-ids --output bench_logs/m20_quiet_decode_ab_2_0.json > bench_logs/m20_quiet_decode_ab_2_0.log 2>&1"
```

本轮按授权只复跑五条模型命令，没有重复 CPU 或 gather+H2D 微测。微测原始路径和历史数字保留如上。测量结束时尚未提交或推送；本次收尾将实现、测试和报告一同纳入版本管理。
