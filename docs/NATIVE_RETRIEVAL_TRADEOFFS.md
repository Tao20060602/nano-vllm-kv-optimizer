# Native NanoKV 检索策略与算子分块：讨论备忘

2026-10-07。用途：解释冻结的稀疏检索路径、可讨论的少量变量及其验证门槛。
这是一份设计讨论文档，不是实现、质量结论或性能结果。当前阶段按用户选择，
固定稀疏参数和检索策略，先研究首 token 延迟；本文件不授权更改路由策略。

## 当前冻结点

近期对照的代表性配置是 Qwen3-4B BF16、64-token retrieval block、每块每个 KV head
`r=4` 个真实 K 代表、prefill/decode 历史 Top-K 均为 32 块、sink 64、recent 512、
prefill chunk 4096、`q-summary=1`、FlashAttention backend 和 direct pinned gather。
它是 M22/近期系统对照的固定配置，不等于所有 `Config` 默认值：例如通用配置里的
recent 和 Top-K 默认值不同。复现时应显式记录整组参数。

首阶段保持以上数值及下表中的检索语义。可单独研究 attention 实现、临时布局或
operator 的 Q tile，但不能把这些低层调度参数叫作“多 Q 检索”或“改变 representative”。

## 这几个名字分别指什么

| 名称 | 当前实现与作用 | 是否改变检索结果 |
| --- | --- | --- |
| Q summary / `prefill_query_segments` | M12 后续 prefill chunk 将实际 query 按 token 位置均分，每段沿 token 轴取均值。`1` 是整段均值；`4` 会产生四个局部均值。 | **会**。summary 用来给历史块打路由分。 |
| K representative / `r` | 对每个历史块、每个 KV head 存 `r` 个真实 post-RoPE K。M12 的构造先选方向最接近块均值方向的 token，再逐个选与已有代表最不相似的方向；归一化只用于构造，保存的是原始 K。 | **会**。representatives 的位置或数量会影响块分数。 |
| Top-K 历史块 | 先为候选块算 representative 路由分数，再对排除保护块后的全局块分数取 Top-K。近期对照固定每层 32 块，即 2048 个远端历史 token。 | **会**。它是远端历史预算，不是 `r`。 |
| sink / recent | 近期配置为 sink 64 token、recent 512 token；保护集合不参与远端 Top-K。后续 prefill 将 selected history、sink、previous recent 和 current chunk 一起 attention，current chunk 内遵守因果 mask；attention 后才保存当前 K/V。 | **决定可见上下文**，但不是远端 Top-K。窗口长度变化是独立的质量变量。 |
| operator `Q tile` / `tokenkernelQtile` | 分段 attention 算子的 Q 分块/并行粒度。当前 NanoKV runtime 没有此参数；handoff 明确每个 token 仍用自己的完整 Q 计算 attention。 | 按定义**不改检索**，也不把多个 token 的 Q 平均。它可能改变数值舍入，之后的层仍可能传播出不同选择。 |

当前 M12 打分可概括为：对每个 query summary、query head/GQA 组、候选块和该块的
`r` 个代表计算原始点积；在代表、同 KV head 下的 query heads、summary 和 KV heads
方向逐级取最大值，得到一个全局块分数；屏蔽 sink/recent 保护块，再取全局 Top-K。
因此这不是“每个 head 各拿固定数量块”。当多个 query 关注不同块时，单一全局预算
可能由少数高分块占满。

实现入口是 [M12 runtime](../nanovllm/sparse/m12_runtime.py)、[Config](../nanovllm/config.py)
和 [ModelRunner wiring](../nanovllm/engine/model_runner.py)。`tokenkernelQtile` 的边界见
[分段 GQA prefill handoff](SEGMENTED_GQA_PREFILL_PLAN_AND_LINUX_HANDOFF.md)。本 checkout
没有同名 runtime 开关。

### 当前 M12 与旧 CPU 原型的边界

`nanovllm/sparse/representatives.py` 保留了 M10 CPU 原型：合成 mean-key 分数，以及对
真实代表取 max 分数；该原型的 real representatives 使用欧氏距离覆盖启发式。它不等于
当前 M12 GPU runtime：M12 用方向余弦启发式构造真实代表，并在 GPU 上用 max 聚合。
CPU 原型可作为另一个实验起点，但不是当前 GPU selector 的可切换选项。当前 M12 没有
real-representative 分数的 min/mean 模式，也没有 per-head 固定配额。

## 可讨论的有限变体

下面是供之后选择的单变量实验，不是建议立即实现。所有比较都先固定其他稀疏参数。

| 变量 | 代码已有程度 | 预期成本与可能收益 | 主要质量风险；prefill / decode 差别 |
| --- | --- | --- | --- |
| Q summary：1 → 4 个连续分段均值 | 已有配置和 M14 实现，默认仍为 1。 | 多算几组 Q·representative 分数，Top-K 与 KV 搬运预算不变。分段可保留整段均值相互抵消掉的局部意图。 | selector 仍对 summaries 取 max、再共享一个全局 Top-32；局部高分会竞争同一预算，不能保证每段都被覆盖。只影响 chunked prefill 路由；decode 每步本来只有一个新 Q，不使用 chunk summary。M14 的单个 16K needle gate 中，目标块命中层数是 q1 30/36、q4 29/36，证据不足以把 q4 设为默认。 |
| representative 数量 `r`：如 2 / 4 / 8 | `r` 是配置项；当前 GPU 构造与 max 扫描均使用该值。 | 更多代表增加构造、存储和 selector 分数工作，扫描量大致随 `r` 增长；历史 KV gather/H2D 的 Top-K payload 不变。更多代表可能覆盖更多方向。 | 现有 max 聚合下，增加代表也增加块得到偶然高分的机会，排名会变，固定 Top-32 仍会丢弃其余块。prefill 每个 chunk 路由一次；decode 对每个生成 token、每层重复路由，成本更直接落在 TPOT。`r` 变更同时影响这两条路径的召回。 |
| 改变每块 K 表示或代表分数聚合 | 当前运行路径仅为方向覆盖的真实 K + 对 `r` 个点积取 max。CPU M10 原型有合成 mean-key/max-real 两条参考；M12 没有 real-score 的 min/mean 聚合。 | mean-key 可减少每块打分数量；均值或 min 聚合也可作为新对照，但需增加/改写 selector。 | mean-key 可能把只出现在单个 token 的稀有方向平均掉；max 对局部峰值敏感，也可能让噪声块挤进预算；min 会压低具有局部相关 token 的整块分数。prefill 的一次选择影响整个 chunk；decode 的选择随每个新 Q 变化，误选会持续影响当前生成步。若比较，应把“代表如何构造”与“代表分数如何聚合”拆成不同实验。 |
| GQA/head/segment 配额或候选并集 | 当前只有 max-reduce 后的全局 Top-K；固定配额未实现。 | 可以防止一组高分 query 独占全局预算，但多路配额取并集可能超过总 Top-K；若仍限总预算，需要定义配额冲突与去重规则。 | 可能增加覆盖多种 query 意图的机会，也会牺牲全局最高分块。prefill 多个 token summary 可能各有方向，适合单独研究配额；decode 通常只有一个 query token，但多个 query head/GQA 组仍共同竞争预算。先确定预算和冲突规则，再决定是否值得实现。 |

`Q tile` 单独调的是 attention kernel 的工作分组，应该在固定 Q/K/V、固定 selected block
IDs 的同输入下衡量数值和时间。它不能回答“summary 选得好不好”。反过来，改 summary、
representative 或配额会改变 attention 输入长度/内容，不能用来证明 tile/kernel 更快。

## 已有质量证据应怎样限制讨论

M22 固定在 q-summary=1、`r=4`、Top-32、sink64/recent512、4096 chunk；其样本数为 8K
每任务 20 条、32K 每任务 10 条。这个 bounded RULER-derived screen 发现：8K multi-key distractor 是 dense 100%、sparse 80%；8K VT
reference-item recall 是 97% / 92%；32K multi-key sparse 为 0/10，而 32K single-needle
是 10/10。32K 没有配对 dense 结果。该报告说明当前配置并非普遍质量保持，也未定位
失败属于 prefill 还是 decode；不能把某个单独 selector 变量称为已确认根因。

M14 的 q1/q4 对照只有一个 16K needle routing gate，目标块选择是 30/36 vs 29/36，
不是广泛任务质量评价。M22 中 baseline 与 M21 static-mask 的 80/80 相同 token IDs，只
说明该系统优化在那些 matched prompts 上没有观察到答案变化，不说明 sparse 等同 dense。
细节见 [M22 quality closeout](m22_quality_closeout_results.md) 和
[M14 prefill results](m14_prefill_results.md)。

此外，2026-10-07 的分段算子 bridge 在冻结 q1/Top-32/sink/recent 与现有稀疏参数时，
首个生成 token 相同，但 96-token 尾 chunk 的后续层有 deterministic selector ID 集合
差异。该次只有两条 prompt、三组新进程，完整 prefill 没有稳定加速；数值误差通过既定
界限也没有证明完整模型质量相等。见 [adapter report](NATIVE_SEGMENTED_ADAPTER_REPORT.md)。
所以低层舍入和后层路由传播也要记录，不能只看首 token。

## 若日后选择检索变体：冻结与采集协议

当前代码没有训练出来的 router，以上候选均为确定性启发式；本轮没有训练步骤。若未来
加入学习型代表或 router，训练、调参和最终 holdout 必须分开，并保存训练样本、代码、
权重与数据哈希。无论是否训练，都先冻结：代码 commit/source hashes、Qwen3-4B snapshot
及 tokenizer、prompt token IDs、chunk boundaries、RoPE/YaRN、dtype、attention backend、
模型运行配置和每个稀疏参数。M22 已公开过的样本不是新的盲 holdout；可以作为已知回归
用例，最终结论要有首次用于定案的新 holdout。

建议把实验分两类并分别定案：

1. **检索策略对照**：一次只改 Q-summary、`r`、表示/聚合或 quota 中的一项。冻结同一
   attention backend 和 `tokenkernelQtile`。dev 用于筛选；最终 holdout 至少覆盖已知
   single-needle、multi-key distractor、VT 类任务和两个上下文长度，并在看结果前固定
   seed、样本数、位置/干扰项分布及打分方法。32K 因无 dense 配对，只能比较 sparse
   变体并报告这一限制，不能声称相对 dense 的质量差距。
2. **kernel/tile 对照**：冻结检索逻辑与参数。记录每个 later prefill chunk、每层的
   selected block IDs 顺序/集合；另做同输入注意力数值比较。记录 tile 配置、首 token
   ID 与后续 token IDs。若注意力舍入造成后续选择漂移，单列首次分歧所在层/chunk，不能
   固定 baseline IDs 来掩盖漂移。

两类都需要保存：

- 每请求生成的 token IDs、原始答案、score/reference；
- 每层每个 prefill chunk 的 selected block IDs；decode 的每层、每个生成步的 IDs；
- 各 arm 的输入 token IDs/hash、模型和代码 provenance；
- request-to-first-token 的 TTFT，以及内部 synchronized prefill step wall time。明确是否
  包含 tokenization/队列等待；排除模型加载、编译和预热，或将它们单独报告。`max_tokens=1`
  可用于首 token/prefill screen，但不产生 TPOT 结果；decode 检索成本应另以足够长的
  decode 负载测量；
- 新进程配对 A/B、多次运行及顺序平衡。最终 TTFT 使用无 profiler 的计时轮次；逐层
  IDs/细粒度同步若会扰动计时，则在同配置的独立审计轮次采集。

在跑之前先约定主要质量指标、TTFT汇总口径、可接受的答案/ID漂移与回退规则。只做
kernel tile 调优且目标是保持输出时，可把首 token ID 和所测输入的 selected IDs 一致
作为严格门槛，同时仍检查注意力误差及后续 IDs。若做 Q/representative/配额变化，选择
策略本身会改变 IDs；此时不能要求 ID 完全一致后又声称在公平测试新策略，而应比较
预先冻结的任务质量、检索诊断、TTFT 和边界案例，并报告具体分歧。

## 何时需要用户决定

当前不需要选新的检索算法：按固定参数先完成首 token 优化的证据链。只有在首阶段结果
已交付、且要进入检索策略实验前，才需要选择优化目标及可接受的质量/TTFT取舍、盲
holdout 的任务/规模，以及“输出必须完全相同”还是允许经质量门槛审查的策略变化。
若没有达到事先约定的质量门槛，不应从性能优势推导默认策略改变。
