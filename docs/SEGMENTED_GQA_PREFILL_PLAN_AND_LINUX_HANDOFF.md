# 分段 KV 的 GQA Prefill 算子：方案、执行清单与 Linux 接手

日期：2026-10-05。状态：**方案与只读分析完成，尚未实现、尚未获得新算子性能结果。**

用户选择的路线是：保留 NanoKV 的系统成果，开发一个可独立评测的 attention 算子，有证据后接回 NanoKV。用户计划下一步在原生 Linux 系统继续；本次仅提交文档，不安装环境、不启动模型 benchmark、不修改 runtime。

## 0. 新接手者先读什么、先做什么

建议依次阅读：

1. 仓库 [AGENTS.md](../AGENTS.md) 与 [环境契约](environment.md)：区分当前 WSL 基线与尚未验收的原生 Linux 环境。
2. 本文：当前选择、范围、第一项任务和停止条件。
3. [项目全历程与交接](NANOKV_PROJECT_HISTORY_AND_HANDOFF.md)：M0–M22 的决策、失败、修复及质量限制。
4. [M22 收尾](PROJECT_CLOSEOUT.md) 与 [质量报告](m22_quality_closeout_results.md)：已完成成果，不要重新包装为无损稀疏注意力。

**下一项工程任务是成本基线，不是直接写复杂 CUDA 内核。** 原生 Linux 尚未验证，应先确认系统、GPU、checkout、Python、依赖与模型；然后按用户的下一条执行指示进行。

接手者应能够准确复述：

> NanoKV 的 M22 系统基线已经收尾，代码基线为 d805917。2026-10-05 用户选择独立算子验证后再集成的新路线。目前候选是 segmented-KV GQA prefill：直接读取 selected history、sink、previous recent、current chunk 四段 GPU K/V，保持当前检索结果与 attention 可见性不变。省拼接有工程意义，但旧 trace 不支持把它当主要瓶颈；FA2 是必须面对的强基线。当前没有新算子实现或加速结果。先完成原生 Linux 环境验收和成本基线，不恢复全部旧优化路线。

## 1. 状态、来源与权限边界

### 已经确认

- 2026-10-05 当前权威运行环境仍为 `NanoVLLM-Ubuntu:/opt/nano-vllm`。
- 分析时的 Git 基线：`d805917c24ec1531a541476b87323f4d97b95ca3`，分支 `main`。
- GitHub：`https://github.com/Tao20060602/nano-vllm-kv-optimizer`。
- Python 3.12.3、PyTorch 2.7.1+cu128、Triton 3.3.1、FlashAttention 2.8.3.post1、CUDA toolkit 12.8。
- RTX 3080 Laptop、16 GiB、SM86。
- 阅读了实际 runtime、模型 runner、attention 分支、benchmark、测试和已安装 FA2 接口，并重新查询已有 M18 SQLite trace；未进行新模型测量。

### 尚未确认

- 目标原生 Linux 的发行版、驱动、checkout 路径、Python 环境与依赖兼容性。
- 原生 Linux 的模型与 ignored trace 是否已经迁移。
- 分段内核是否比当前或改良的 packed FA2 更快。
- 是否存在新的端到端 prefill 收益。
- 最新计划的实验命令与新 benchmark driver：尚未实现，不得声称命令已跑通。

### 不在本轮范围

- 不改 Top-K、Q 检索摘要、representative 选择、图检索或动态预算。
- 不重做 CPU KV offload、GPU 历史 cache、copy pipeline 或 page allocator。
- 不做训练 backward、dropout、完整多请求 serving 或跨 GPU 支持。
- 不安装 Windows 原生推理依赖，不删除 WSL，不租用付费 GPU。
- 旧质量问题保持公开；不能借算子工程修改宣称检索质量得到解决。

## 2. 实际数据路径：为什么选择分段，而不是先做 paged

当前 sparse 后续 chunk 的 prefill 为：

```text
完整 CPU 历史 KV
  -> GPU representative selector -> selected block IDs 回传 CPU
  -> CPU gather 到 pinned staging
  -> selected history 上传 GPU
  -> [selected history | sink | previous recent | current chunk] 连续拼接
  -> flash_attn_func，bottom-right causal
  -> 当前 K/V 入 CPU 历史，更新 reps/sink/recent
```

关键源码入口：

- [prefill_chunk](../nanovllm/sparse/m12_runtime.py)：检索、gather、拼接、attention、最后 store。
- [attention dispatch](../nanovllm/layers/attention.py)：首 chunk 与后续 chunk 走不同分支。
- [ModelRunner](../nanovllm/engine/model_runner.py)：sparse M12 不分配全历史 GPU paged KV；不能把 dense 分支的 block table 当 sparse 当前数据结构。
- [布局回归](../tests/test_m14_prefill_layout.py)：已有 recent 覆盖与小型 attention 对照，不等于新算子的完整正确性协议。

典型配置（足够长的历史、无边界缩短时）：

| 段 | token 数 | 可见性 | BF16 K+V 大小，Hkv=8、D=128 |
| --- | ---: | --- | ---: |
| selected history | 2048 | 所有当前 Q 可见 | 8 MiB |
| sink | 64 | 所有当前 Q 可见 | 0.25 MiB |
| previous recent | 512 | 所有当前 Q 可见 | 2 MiB |
| current chunk | 4096 | current position j <= query position i | 16 MiB |
| packed 合计 | 6720 | 历史全可见 + 当前 causal | 26.25 MiB |

以上是 shape-derived payload，不是 Nsight 硬件流量测量；保护窗口必须去重，边界处段长度可变。

当前 decode 的复用 packed buffer 只覆盖 selected+sink+recent；后续 prefill 还包含整个当前 chunk，因此该分支仍分配更大的 packed K/V。selected staging 先 `.to(device)` 再写入 pack，也有临时 GPU 张量与复制的优化候选。

### 新算子的目标

输入 Q 与四段分别存放的 GPU K/V，直接输出 attention；不要求最终连续拼接。selected history 的 CPU gather 与 H2D 仍然存在，不能假装它们已被消除。

四段需要统一的 softmax 归一化。可在分段循环中维护同一套 online softmax 状态；若分别计算 partial attention，则必须用各段 LSE 正确加权合并，而不是相加或平均已归一化输出。

### 两种 Q 分块不是一回事

- 当前 Q 摘要/平均用于检索块，是算法质量问题，本轮不改。
- 新 kernel 的 Q tile 用于并行计算，每个 token 的完整 Q 仍计算自己的 attention，不把整个 chunk 平均成一个 Q。

### 为什么不是直接套 FA2 paged 接口

FA2 已支持 GQA、paged KV、不同 Q/K 长度和 causal，所以不能把这些通用能力本身当原创成果。

本机安装版本 2.8.3.post1 的接口文档，以及 [v2.8.3 C++ API](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/flash_api.cpp)，要求 paged KV 的物理 page 大小为 256 的倍数。本项目的 64-token retrieval block 不是同一概念。当前四段不同分配的 K/V 也不能无转换直接当成该接口要求的单一 paged cache。

可以以后评估 256-token 物理页、64-token 子块 mask 或其他 backend，但本轮先重建 GPU allocator 会把算子项目变成系统重构。首版先做 segmented 输入。

## 3. 旧 trace 的重新分析：省拼接不等于大收益

来源是本地 ignored 的 `bench_logs/m18_prefill_16k_step1.sqlite`，关联同名 `.nsys-rep`。2026-10-05 只读重聚合，不是新跑的测量。

| 项目 | count | 时间累计 |
| --- | ---: | ---: |
| CPU NVTX `nanokv.prefill.step` | 1 | 1778.421 ms |
| CPU NVTX `prefill_selector` | 36 | 851.021 ms |
| CPU NVTX `prefill_cpu_gather` | 36 | 47.732 ms |
| CPU NVTX `prefill_h2d_pack` | 36 | 48.710 ms |
| CPU NVTX `prefill_attention` | 36 | 12.615 ms |
| CPU NVTX `prefill_store_kv` | 36 | 710.459 ms |
| GPU BF16 GEMM 两个主要 kernel 组 | 108 + 36 | 747.091 ms |
| GPU `flash_fwd_kernel` | 36 | 313.392 ms |
| GPU H2D memcpy | 149 | 24.707 ms，288.065 MiB |
| GPU D2H memcpy | 146 | 177.333 ms，576.009 MiB |
| GPU D2D memcpy | 288 | 3.706 ms，693 MiB |

主要 GEMM 分别为 `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_tn` 与 `cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>`；不是仅凭 `Kernel2` 的短名称猜测分类。

限制与解读：

1. NVTX CPU 范围通常记录提交与等待，不等于该阶段 GPU 完成时间。attention 的 CPU 12.615 ms 与 GPU 313.392 ms 正说明异步归因问题。
2. selector 的 851.021 ms 可能包含此前模型计算的等待；不能据此断言 representative 扫描算了 851 ms。
3. 活动累计值可能重叠，不得求和当 wall time，也不能作为严格关键路径占比。
4. D2D memcpy 不覆盖所有 kernel 实现的复制；也不是所有 D2D 都由 prefill 最终拼接产生。3.706 ms 不是拼接收益的严格上限。
5. 这些值是单个旧 profile 的诊断证据，不是原生 Linux 或新算子的速度结论。旧 trace 的完整运行元数据、背景负载和源树状态需要从关联旧报告/日志追溯，不能补写成已完全匹配。
6. 它支持“attention 值得研究、只省显式复制空间有限”，不支持“zero-pack 已经能显著加速”。

已有 M17 也展示 CPU gather 局部变快不必然转成 prefill wall 改善。首轮必须做成本基线，不能跳过强对照。

### 原始文件识别与复核

SHA256（以本次检查的实际文件为准）：

```text
bdc8a0413376ba2be111e0d40d6b172f2a7d0468e6c09106d5f70e4b551fddf1  m18_prefill_16k_step1.nsys-rep
ad22dd66ec81b027b18be6fde6a7fd6f61a9246287e5c9b55137119be1c4749d  m18_prefill_16k_step1.sqlite
```

文件不在 GitHub clone 内，迁移前需单独保存；不要因为新 clone 没有它们就称历史 trace 不存在。导出 SQLite 后再改动数据库会改变 hash。

无需 SQLite CLI，Python 标准库可只读查询：

```bash
# 在已确认 checkout 内、激活正确 venv 后执行；不启动模型。
python - <<'PY'
import collections
import re
import sqlite3
from pathlib import Path

trace = Path('bench_logs/m18_prefill_16k_step1.sqlite').resolve()
assert trace.is_file(), 'trace 是 ignored artifact，需从旧环境单独迁移'
db = sqlite3.connect(trace.as_uri() + '?mode=ro', uri=True)
groups = collections.defaultdict(list)
for label, ms in db.execute('''
    SELECT coalesce(n.text, s.value), (n.end-n.start)/1e6
    FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
    WHERE n.end IS NOT NULL
'''):
    if label:
        groups[re.sub(r'layer\d+', 'layer*', label)].append(ms)
for label, values in sorted(groups.items()):
    print(label, len(values), round(sum(values), 3), 'CPU NVTX sum_ms')
for row in db.execute('''
    SELECT copyKind, count(*), sum(end-start)/1e6,
           sum(bytes)/1048576.0
    FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind
'''):
    print('memcpy kind/count/sum_ms/MiB:', row)
for row in db.execute('''
    SELECT s.value, count(*), sum(k.end-k.start)/1e6
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName=s.id
    GROUP BY s.value ORDER BY sum(k.end-k.start) DESC LIMIT 3
'''):
    print('kernel:', row)
db.close()
PY
```

该查询已在旧 WSL 的 SQLite 上验证；新系统仍需提供同一文件与兼容 Python。Nsight 其他版本 schema 若不同，应先查 schema，不盲改统计。

## 4. 技术范围与正确性契约

首版只做 forward、单请求、BF16、D=128、Hq=32/Hkv=8，scale 与当前 runtime 一致，无 dropout。Q/K 已经过模型需要的处理；算子不重复做 RoPE。

具体约束：

- GQA 映射 `kv_head = q_head // 4`；不提前 repeat 全部 K/V 来制造隐性内存开销。
- 三段历史对当前 Q 全可见，current 中只有 `j <= i` 可见。
- 所有段共同归一化，FP32 softmax 状态/累加；不承诺 BF16 运算 bitwise 等同 FA2。
- sink/recent/selected 去重语义保持 runtime 现状，不能重复计入同一历史 token。
- 输入 stride、段长度、设备、dtype、head 配置有明确约束；不支持的形状直接拒绝或走原 backend，不静默计算错误。
- 空历史、尾部非整 tile、短 recent、极端 logits、GQA 映射和因果边界要覆盖。
- 当前 K/V 必须在旧 recent 被使用之后更新入状态，保留 M14 修复的顺序。
- 每个独立请求必须重置 sparse 状态，保留 M16 修复；不能让旧 needle/历史泄漏。
- 正确性参考为小规模 FP32 attention oracle 与同输入 packed FA2；误差阈值按数值分布预先定义并公开 max/mean/RMS 等，而不是看答案后放宽。
- 算子数学语义保持不变不等于所有生成 token 必然相同；集成后仍检查 greedy 输出，差异时先诊断数值与 mask，不能直接宣布质量无损。

Tensor Core 证据要看编译产物的 MMA/HMMA 指令，并报告 profiler 发现；仅有 `tl.dot` 不构成证明。当前硬件是 Ampere SM86，不采用 Hopper-only TMA/WGMMA 作为首版依赖。

## 5. 对照与测量：强基线不能省略

| arm | 内容 | 用途 |
| --- | --- | --- |
| A | 当前 runtime 的 pack + FA2 | 对照已有系统 |
| B | 复用足够大的 pack、减少临时 GPU 张量/可行时直接 H2D 到目标 slice + FA2 | 防止只赢过可轻易修好的旧路径 |
| C | 新 segmented attention，不做最终 pack | 核心候选 |
| D，可选 | 分段 FA2 + 正确 LSE merge | 判断已有 backend 组合能否代替自写内核 |
| oracle | 小规模 FP32 reference | 检查正确性，不作主要性能基线 |

B 与 D 都是待验证方案，不是已实现的功能。D 的 LSE API、布局、额外调用成本要先确认；不要为了获取 LSE 无意分配巨大的 attention probability 张量。现有 `flash_attn_with_kvcache(..., return_softmax_lse=True)` 是候选接口，不是已验证的本项目 dispatch。

三层性能必须分别报告：

1. **kernel**：所有输入已在 GPU，剔除 compile/autotune 与 setup；记录输出写入等实际 kernel 成本。
2. **operator chain**：pack 分配/复用、复制、FA2 或新 kernel、launch 与完成边界；注明是否含 H2D，至少同时给 GPU-resident 与真实搬运场景。
3. **system**：完整 prefill 的同步 wall，以及后续 chunk wall；首次 chunk 单列，模型 setup/tokenization/首 token sampling 是否包括必须明确。

初期形状矩阵：Q 长度 64/256/1024/4096，selected history 长度 0/512/2048；sink/recent 用实际上限并补空段、缩短及非整 tile 尾部。不是要求一开始支持全部模型和 head dimension。

测量协议：

- 相同输入、dtype、scale、mask、selection；seed 与输入 hash 固定。重排 K/V 或变化 K 不属于同口径优化。
- JIT/autotune 单列，不放入稳态 kernel 结果，也不隐瞒冷启动。
- GPU event 测 kernel 完成，同步 wall 测真实链路；profiler 与非 profiler 测量分开。
- 热缓存/重复访问与来自真实请求的输入要标明，不把缓存热微测泛化到 offload 系统。
- 报告 median、尾部延迟、重复样本及原始记录；正式系统比较先做至少 3 组 fresh-process 交替 A/B，记录运行顺序和背景负载。
- CPU 线程数显式固定并记录，初期可用 8，所有 arm 相同；需確認当前 driver 没有覆盖该配置。
- 保存 Git commit 与 dirty diff/source hash、依赖版本、GPU/驱动、模型快照、输入 IDs/hash、selected IDs 或其完整可审计记录、配置、时间序列、峰值 alloc/reserved 与 host RSS。
- Nsight Systems 看依赖、launch、copy、等待；Nsight Compute 仅捕获选定 kernel/小范围，观察寄存器、共享内存、占用、访存与 stall。权限不足交给用户，不绕过系统权限。
- 不以 profiler 下的时间作为正式加速比例，不累加不同实验的百分比，不取最好的一对代表稳定收益。

## 6. 执行阶段、交付与停止条件

### P0：原生 Linux 环境验收

交付：系统/驱动/GPU、checkout SHA、Python/依赖、模型路径与迁移 artifact 清单。先做只读检查，不全套重跑旧实验。

停止条件：模型或依赖缺失、硬件不匹配、GPU 权限不足时，准确汇报缺项；不自动下载大量数据或转租 GPU。可先继续 CPU 文档/源码工作。

### P1：成本基线

任务：提取实际分段布局，建立 A/B 对照，测 pack、FA2、operator chain；必要时用短 Nsight capture 定位复制与等待。

交付：可复跑 driver、配置/输入/源码记录、原始结果与成本预算报告。先把 B 修到合理水平，不能故意留下明显多余分配来衬托 C。

决策：若省 pack 的空间很小，明确调整目标为 attention 内核研究，不承诺系统收益；若已有 backend 组合已经足够，应接受无需复杂定制的结果。

### P2：最小 Triton 正确性原型

任务：用完整 Q、分段寻址、统一 online softmax 与 GQA 映射实现 forward。可以参考匹配本机版本的 [Triton 3.3.1 fused-attention 教程](https://github.com/triton-lang/triton/blob/v3.3.1/python/tutorials/06-fused-attention.py)，保留归属，明确新增布局/GQA 支持；不能把教程复现当原创高性能成果。

交付：独立 operator API、误差与边界测试、A/B/C 形状结果。首版不为了接口通用性实现 backward/multi-batch/all-dtypes。

停止条件：mask、LSE 或尾部有错误，先修正确性，不进行带错误内核的性能宣传。

### P3：单热点优化

根据真实 profile 选择 tile 大小、warp/stage 参数、访存与寄存器布局，或 GQA 组内 K/V 复用。GQA 共享可能增加寄存器/共享内存与降低并行度，不先断言必快。

交付：前后同输入对照、MMA 证据、硬件指标与根因解释。若需要 CUDA C++/CuTe 深化，只做一个已定位热点，不并行维护两套完整 attention。

### P4：受控接入 NanoKV

接入后续 chunk 的 pack/attention 分支，保留默认 FA2 和 feature-off 路径；CPU gather、IDs、Top-K、状态更新不变。

交付：Qwen3-4B 的 fresh-process prefill A/B、输出/selection 检查、相关状态与布局回归。先小规模固定输入，不自动恢复全部 64K/128K 旧补测。

决策：只有 operator-chain 及系统结果支持时才考虑默认开启。kernel 变快但 system 没变快，应分别报告。若 C 比改良 B 慢，可以保留独立算子研究与失败分析，但不硬接默认路径。

### P5：公开报告与面试表达

交付：独立算子 benchmark、集成边界、原始结果、失败形状、硬件证据、复跑与 handoff。可以是同仓库独立模块；是否拆仓库后续讨论，不先建立第二套完整推理框架。

允许的叙事是“真实布局 -> 强基线 -> profiler -> 数值/性能验证 -> 集成判断”。不允许在数据之前写“超越 FA2”“prefill 大幅提升”“质量无损”。

## 7. 复用旧 benchmark 时容易踩的坑

[benchmark_m14_prefill.py](../benchmarks/benchmark_m14_prefill.py) 当前存在以下默认差异：

- `--query-segments` 默认 4，近期参考主线使用 1；比较时显式传 1。
- `--index-select` 是 store_true，不传则 driver 会覆盖引擎默认并使用旧 gather；比较时显式传该开关。
- 不传 `--model` 会从缓存快照中排序选一个；正式实验必须指定 snapshot。
- driver 的 `max_tokens=1` 用于 prefill 测量；不等于独立 decode benchmark。末次 prefill 包含首 token 采样，模型 setup/tokenization 不在这些 step 时间中。
- 现有 JSON 不足以满足新算子的完整 input/selection/source provenance，需要后续增加记录，不能假定已具备。

下面是**旧 packed FA2 基线命令模板**，不是新算子的命令，也未在本次重跑。原生 Linux 验收后，将 `NANOKV_MODEL` 设置为确认存在的同一 4B snapshot，`NANOKV_OUT` 设置为不覆盖旧结果的新文件：

```bash
# 已在确认 checkout 内、激活正确 venv，且用户已要求跑基线。
: "${NANOKV_MODEL:?set the verified Qwen3-4B snapshot path}"
: "${NANOKV_OUT:?set a fresh output path}"
test -d "$NANOKV_MODEL"
test ! -e "$NANOKV_OUT"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python benchmarks/benchmark_m14_prefill.py \
  --model "$NANOKV_MODEL" --seq-len 16384 --chunk-size 4096 \
  --query-segments 1 --prefill-top-k 32 --index-select --backend flash \
  --rope-mode yarn --output "$NANOKV_OUT"
```

初期原生 Linux 的 4B 系统 A/B 沿用用户已选择的近期模型，不擅自换成 0.6B。微测本身不需要模型；0.6B 仅作为原有环境 smoke 模型，不能混报为 4B 性能结果。

## 8. Linux 迁移：哪些能 clone，哪些必须另存

### GitHub 带得走

tracked 源码、tests、benchmarks、阶段报告、公开小型结果、完整历史文档与本计划。提交号包含具体发布状态；正文 `d805917` 指本轮文档前的代码基线，不声称是 push 后最新 HEAD。

### GitHub 带不走

- `/opt/nano-vllm/.venv`：在新系统重建，不把旧 venv 当可移植包。
- `/usr/local/cuda-12.8`、NVIDIA 驱动、系统依赖：分别确认，不复制 Windows CUDA 或旧系统驱动文件。
- `/opt/models/Qwen3-0.6B` 与 Qwen3-4B 的 HF cache：按磁盘容量决定保留/下载；本次没有执行迁移或新下载。
- `bench_logs/` 下 ignored prompts、prediction、日志与 Nsight trace；至少保留本文列出的 M18 trace pair。完整项目取证还应保留 M22 全量 prompts/predictions 和 M17–M21 对应日志。
- Windows `NsightReports/` 中 GUI 副本：先确认是否与 WSL 文件重复，不根据文件名默认内容相同。

HF snapshots 可能使用指向 `blobs/` 的软链接。只复制 snapshots 子目录可能得到断链；应保留相关 HF hub 的 blobs/snapshots/refs 结构，或做有检查的解引用导出，并在目标系统确认 tokenizer、config 与全部权重 shard 可读。不要仅用“目录存在”判断模型迁移成功。

迁移验收前不删除旧 WSL、不注销发行版、不递归清理缓存。源代码上传成功不等于模型、环境与原始证据都已保存。

### 原生 Linux 的首次只读接手

1. 用户选择一个真实 clone 路径，新对话在该目录开始；不猜测它必为 `/opt/nano-vllm`。
2. 读 `AGENTS.md`、`docs/environment.md` 与本文；检查 `git status --short --branch`、`git remote -v`、`git log -5 --oneline` 和 `git rev-parse HEAD`。
3. 用 `uname -a`、`cat /etc/os-release`、`nvidia-smi` 确认是原生 Linux 及 GPU/驱动状态。原生 Linux 不运行 `wsl.exe` 或 PowerShell wrapper。
4. 确认 Python `<3.13` 的项目约束、venv、PyTorch CUDA runtime、CUDA toolkit、Triton 与 FA2，优先保持旧可比版本。重建环境属于下一次任务，不是本次已完成事项。
5. 在明确选择的 Linux venv 内检查 import、GPU 型号/SM、dtype 支持与模型路径；不要在 system Python 中不断安装依赖。
6. 将该机实际路径/版本记录到新的环境报告。旧 WSL 设置是 reference，不是原生 Linux 已验证结果；新驱动/依赖或功耗环境变化时，旧 wall 数字不能作新系统对照。
7. 检查 ignored artifacts 和模型软链接迁移完整性，然后按用户指示开始 P1。

旧 `scripts/check-wsl-environment.sh` hard-code `/opt/nano-vllm`、旧 venv 与 0.6B；它适用于 WSL 基线，不是原生 Linux 通用验收脚本。本次未创建或声称已有新的 Linux setup 脚本。

## 9. 本次发布内容与未做事项

本轮为 documentation-only：发布完整 M0–M22 历史交接、当前分段算子方案与 Linux 接手边界，并修正文档导航/状态。

没有新增算子、性能数字、安装记录、quality 声明或新模型测试。旧 trace 重聚合是已有证据的诊断摘要，不能记成新实验成功。

下一轮若开始执行，先更新本页状态并按阶段记录“提出 / 实现 / 验证 / 否定”的区别，保存失败结果与实际决定。
