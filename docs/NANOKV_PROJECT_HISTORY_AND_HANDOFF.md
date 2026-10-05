# NanoKV 项目全历程、决策复盘与最终交接

整理日期：2026-10-05。面向项目本人及未来接手者。

后续状态更新：用户在本复盘之后选择了“独立算子验证后接回 NanoKV”的新路线，且准备迁往原生 Linux。当前计划和接手入口见 [分段 GQA prefill 方案与 Linux 交接](SEGMENTED_GQA_PREFILL_PLAN_AND_LINUX_HANDOFF.md)。下文的“项目收尾”“不自动优化”和 Git 状态描述保留为 M22 收尾时的历史快照，不代表新路线已经实现；最新 Git 状态仍需实时检查。

本文回答的不是“代码放在哪里”，而是：我们最初想解决什么问题，为什么逐步走到现在的方案，哪些数据改变了方向，哪些想法实施后没有成功，最终到底交付了什么，还有哪些不能宣称已经解决。

项目已经按用户要求完成收尾。本次只整理资料与说明，不重跑模型、不启动优化、不重新开放工程路线。未来展望是供理解和选择的候选方向，不是正在执行的任务。

## 阅读说明：先把不同版本放到正确的位置

本文依据当前仓库源码、阶段报告、已发布结果，以及旧 M12 交接文档整理。正文中的数字是历史测量的复述，不是 2026-10-05 的新测量。

本次只读确认时，权威仓库为 `NanoVLLM-Ubuntu:/opt/nano-vllm`，分支 `main`，HEAD 为：

`d805917c24ec1531a541476b87323f4d97b95ca3`

当时工作树干净，本地跟踪的 `origin/main` 与它一致。本交接文档是在此之后新增的本地文档；“已有成果已发布”不代表本次新文档也已经 push。接手时仍应检查实时 Git 状态。

来源有先后关系：

- 旧 `HANDOFF_NEXT_CHAT.md` 是 M12 时期快照。其中 HEAD、未修复问题、README 状态和下一步指令已经过时。
- M13–M21 报告保留实验当时的分支、默认值和“未 push”等描述，不代表这些阶段现在还没有合并。
- 最终状态以当前源码、`PROJECT_CLOSEOUT.md`、`limitations.md` 和 M22 质量报告为准。
- 我们不把“提出过”写成“做过”，不把“局部更快”写成“整体更快”，也不替尚未定位的问题编造根因。

本文中的“为什么选择”尽量对应已记录的约束、对照和结果。必要的工程解释会明确标为解释或假设，不能理解为所有想法在实施前都已被充分证明。

---

## 一、先用一段话理解整个项目

我们基于 nano-vLLM 做了一个有限显存下的长上下文推理原型。

最初做的是“已有前缀的 KV 能不能保存在 CPU，下一次请求直接恢复，不用重新算”；后来转向“单条很长的上下文放不进 GPU 时，能不能把完整历史放在 CPU，只把当前问题需要的少量历史搬上 GPU”。受 AlayaDB 的查询感知检索思路启发，我们探索了 block 稀疏注意力、代表向量和图检索。对照发现：图方法有检索价值，但本项目的 Python 图遍历太慢，因此改用 GPU 批量扫描每块的少量真实 Key 代表。

这条路线用 Qwen3-4B、YaRN、4096-token 分块 prefill 和固定历史预算，完成过 128K 输入及真实稀疏生成。接下来，数据和代码审计暴露了计时口径、历史覆盖、跨请求状态等问题。我们先修正这些问题，再尝试 Q 分块、动态 Top-K、降低 prefill 预算、CPU gather、CUDA Graph、搬运重叠和静态 mask。真正留下来的主要收益是 FA2 prefill、直接写入 pinned staging 的 gather，以及可选的静态 mask；图缓存、动态预算、CUDA Graph 和 pipeline 没有成为可靠的默认优化。

最后采用固定版本的 NVIDIA RULER 生成器及评分函数补了质量证据。结果证明：M21 系统优化在这组样本上不改变原稀疏路径的回答，但原稀疏路径本身存在质量损失，尤其 32K 相似 Key 干扰任务全部失败。项目因此以“有实际链路、对照数据和明确局限的研究工程原型”收尾，而不是包装成一个质量无损、全面超越 dense 的推理引擎。

## 二、贯穿所有决策的四个矛盾

理解后面的转折，先理解这四件事。

### 2.1 GPU 算得快，但 GPU 内存有限

16 GiB 显存不仅要装 KV，还要装模型权重、临时激活和执行缓冲。

以 Qwen3-4B 的记录配置为例，128K 全量 BF16 K/V 约 18 GiB，单是历史 KV 就超出整张卡的容量。把历史移到 CPU 是容量方案，但 CPU 不是免费的无限扩展：它消耗主存，也引入 gather、PCIe 传输和同步。

因此“把长输入跑起来”和“比 GPU dense 更快”是两个不同目标。前者取得了进展，后者不能普遍成立。

### 2.2 少看历史会减少搬运，但可能漏掉答案

固定 Top-32 每层选 32 个 64-token 历史块，即 2048 个远端历史 token。sink/recent 等保护窗口另行参与，不在这 2048 个远端 token 里。

预算减少通常会减少传输张量，但回答可能依赖某个非常不起眼的历史块。简单重复文本的输出不变，不能证明复杂检索不受影响。这也是后来低 Top-K 没有成为默认值的原因。

### 2.3 检索更聪明，不等于实现更便宜

图检索、多 Q 摘要、动态 Top-K 都是合理假设。但增加搜索、归约、分支或同步，有可能花掉省下的搬运时间。

我们不是只优化“选了多少 token”，而是要看：

`检索 + IDs 回传 + CPU gather + KV 搬运 + attention + 其余模型计算`

整个路径是否更短。

### 2.4 一段函数的耗时，不一定是这段 GPU 计算的耗时

CPU 可以提交 CUDA 工作后立即返回；也可能在稍后的 `.cpu()`、标量判断或同步处等待 GPU。

所以某个 CPU 范围很长，可能包含前面排队工作的等待；某个 H2D 函数返回很快，也不意味着 GPU 已经拷贝完成。后来的计时修正、Nsight 和未开启 profiler 的 A/B，就是为了不把这些现象误读成收益。

## 三、历程地图

这张表是导航，不代替后文的因果解释。

| 阶段 | 要回答的问题 | 实施或观察 | 最后决定 |
| --- | --- | --- | --- |
| M0–M7 | 相同前缀能否跨 GPU 淘汰后复用？ | CPU KV store、身份校验、恢复、TTFT 对照 | 保留前缀复用；收益依赖长度 |
| Llama 适配 | 能否扩展模型加载能力？ | Llama adapter，TinyLlama 验证 | 保留适配；不宣称广泛 Llama 3 验证 |
| M8 | 稀疏选择怎样才有可检查的参考？ | exact block oracle、误差和召回 | 保留实验基准，不当加速方案 |
| M9 | 真实模型 KV offload 是否可用？ | post-RoPE 真实数据、CPU 路由与搬运 | 验证可用，但同步路径很贵 |
| M10–M11 | 图检索是否值得接入真实生成？ | 代表向量、KNN、query-guided 图及生成对照 | 当前实现选择 flat representatives |
| M12 | 能否在这张卡跑 128K？ | Qwen3-4B、YaRN、chunked prefill、GPU selector | 容量链路跑通；审计发现证据缺口 |
| M13 | 上一步选中的块是否值得常驻？ | 1080 次 transition，reuse 均值 32.2% | 不做简单 previous-set cache，改 gather |
| M14 | prefill 能否更正确、更高效？ | 补 previous recent、FA2、4 个 Q 摘要 | 修复与 FA2 留下；多 Q 不默认 |
| 补测/M15 | 哪些旧结论真的成立？ | 同口径 benchmark、Q=64、PyTorch trace | 不夸大 quality；去掉逐层调试同步 |
| M16 | decode 预算能否动态缩小？ | K=16/24/32 与 adaptive；修复状态泄漏 | payload 减少，但无 TPOT 收益 |
| M17 | prefill 能否减小 K、改善搬运？ | prefill K=24/16、gather、D2H 试验 | 有局部收益和质量退化，不默认减 K |
| M18 | 用 Nsight 后最值得落地什么？ | timeline、三组 gather A/B | direct pinned gather 默认开启 |
| M19 | CUDA Graph 能否减少 selector 提交成本？ | 修正 capture 开销；fresh/ABBA 对照 | 方向不稳定，默认关闭 |
| M20 | gather 与 H2D 重叠能否更快？ | 微测、K/V pipeline、真实 overlap、空闲复测 | overlap 成立，E2E 不快，默认关闭 |
| M21 | 是否有更小、确定的冗余同步？ | protected mask 缓存，108 次额外 copy/wait 消失 | 小范围性能成立；保留可选，不改默认 |
| M22 | 最终质量到底怎样？ | 80 prompts、220 generations、官方固定评分 | 公开质量损失；完成收尾 |

---

## 四、M0–M7：从前缀复用开始，先学会管理 KV 的身份和生命周期

### 4.1 最初的问题不是稀疏注意力

上游已有 GPU prefix cache。问题是：GPU 物理块被重新使用后，旧前缀没有独立的持久归宿。相同前缀再次出现，可能仍需重新 prefill。

最初目标很具体：把可复用的完整 KV 块存到 CPU，在后续请求中恢复到 GPU，只算新的尾部。这里复用的是相同 token 前缀的完整 KV，不是根据语义挑选少量历史。

### 4.2 为什么分成多个里程碑

我们没有一开始就同时改调度、缓存和注意力，而是逐步建立边界：

- M0：确认上游 baseline 可以正常运行。
- M1：建立 lookup、prefill、load、TTFT 等计数和计时，确认 instrumentation 不改变输出。
- M2：区分谁管理块身份、谁持有实际 K/V，建立可复用 block store 抽象。
- M3：CPU 存储、容量、pageable/pinned、LRU 与句柄生命周期。
- M4：ContextDB 最长连续前缀匹配。
- M5：接入真实 scheduler/model runner，验证恢复和失败回退。
- M6：比较 cold、GPU hit、CPU pageable、CPU pinned 和部分命中。
- M7：整理交付。

这里的重点不是里程碑编号，而是先证明状态和复用正确，再讨论速度。

### 4.3 为何必须有身份校验、完整块和失败事务

同一个 GPU 物理地址不代表同一个历史上下文；同一 hash 也不能作为 token 完全相同的证明。模型、RoPE、dtype 和 KV 布局不兼容时，更不能复用旧数据。

因此采用 fingerprint、token-ID 再校验、版本化 CPU handle 和 LRU 失效清理。只恢复完整块，不跨越前缀中间的缺口；部分尾部重新计算。还至少保留需要执行的 prompt 尾部，以产生首 token logits。

如果恢复中途失败，不能让“半恢复”的数据参与推理，而是撤回新分配块，按完整 prefill 重算。相关注入失败的测试验证了回退后 token 与 cold 一致。

这是后来各种优化都绕不开的经验：缓存首先是正确性和生命周期问题，其次才是速度问题。

### 4.4 数据告诉我们：缓存不是越多越好

后续补测仍使用 Qwen3-0.6B，5 次 warmup、20 次记录。4K 前缀时，TTFT 中位数为：

| 4K 前缀模式 | TTFT |
| --- | ---: |
| Cold | 181.58 ms |
| GPU hit | 32.58 ms |
| CPU pageable | 97.60 ms |
| CPU pinned | 76.67 ms |

但 CPU 复用在 1K 及以下没有稳定收益，约 2K 开始出现本环境下的 crossover。短前缀重新计算很便宜，传输和固定管理成本反而可能更大。

另一个 1K 前缀、生成 32 tokens 的对照中，GPU hit 改善了 TTFT，但整个请求墙钟没有可靠下降：后续 decode 没有变快，仍占主要时间。

因此留下的结论是“长共享前缀有机会降低首 token 延迟”，不是“使用 CPU cache 就一定提高整次生成速度”。

### 4.5 这条线后来没有消失，但也没有与稀疏模式合成一套系统

M0–M7 保留在仓库里，是独立能力。当前 sparse 配置明确不支持与这套 reusable/CPU prefix cache 同时启用。

不能把两条线的成果拼成“已经同时实现跨请求复用、长上下文稀疏和生产调度”。我们做过各自的原型，但没有完成组合系统。

证据：[前缀复用设计](end_to_end_reuse.md)、[补测](benchmark_backfill_results.md)、[历史正确性](correctness.md)。

## 五、模型适配插曲：为什么没有沿 Llama 授权路线继续

项目增加过 Llama model adapter，提交记录说明使用 TinyLlama-1.1B 验证。它说明代码不只局限于原先的 Qwen 加载路径，但不是对所有 Llama 3 长上下文行为的验证。

曾考虑官方 Llama 3.2 checkpoint，后来遇到 gated 权限问题。为了尽快形成可运行成果，后续选择已经可用的 Qwen3-4B，不继续把进度压在授权等待上。

这里是可用性、时间和可比性的取舍，不是测出 Qwen 在所有任务上优于 Llama。后续主线的证据对应 Qwen3-4B，不能挪给未测过的模型。

## 六、M8：先建立 exact oracle，避免不知道“近似检索漏了什么”

### 6.1 为什么突然研究 block 稀疏

前缀复用减少的是跨请求重复计算。对于第一次出现、特别长的单条输入，它不能解决所有历史 KV 驻留 GPU 的容量问题。

受 AlayaDB 启发，我们提出另一条路线：对当前 Q 找到重要历史，只 attend 相关块，CPU 保存完整历史。为了让后续近似方法有参考，先做 exact 版本。

### 6.2 为什么按 block，而不是按单个 token

token 级路由粒度更细，但索引、搬运和控制更碎。按连续 block 选择，可以拿到相对连续的 K/V，减少碎片化搬运，也比较容易与现有缓存结构衔接。

代价是块里可能只有一个 token 重要，却要把整个块搬回来；块越大越容易多搬，块越小越容易增加路由和管理工作。

早期上游物理 KV 块为 256 tokens，检索块先用 64 tokens。两者不是同一参数，不能因为检索改 64，就认为上游 allocator 也全部改成 64。

### 6.3 Exact 方法起到了什么作用

我们对每个块计算真实 Key 的最大 Q·K 分数，建立固定 Top-K、Block-DIPR 阈值选择、GQA 多 head 合并、去重和 packed attention，检查重要 token 召回、attention mass 和输出误差。

由于 exact 选择自己就扫描完整历史，它更适合作为 oracle，不是“免费的稀疏加速”。

补测中，8K 随机张量的 exact Block-DIPR 可以做到 critical-token recall=1.0，但仍选约 52% token，且选择加稀疏 attention 比 dense 更慢。正确性基准成功，不代表性能路线成功。

### 6.4 这一阶段对后续的影响

它告诉我们两个事情：近似检索应该和真实分数比较；不同 head 合并后，原本每个 head 看似很稀疏的集合，可能整体变得很大。

证据：[早期 block 稀疏设计](block_sparse_design.md)、[M8 补测](benchmark_backfill_results.md)。

## 七、M9：从随机张量转向真实模型，但同步 CPU 路径已经很贵

### 7.1 为什么随机张量结果不够

真实模型 Q/K 的结构、GQA 关系、RoPE 位置和注意力分布都影响检索。随机张量上正确，不能证明真实模型上重要信息能保住。

因此从真实 Qwen3-0.6B 捕获 post-RoPE 的 Q/K/V，先在单层 replay 中研究 CPU offload：

`当前 Q → 选择历史块 → CPU gather → pinned staging → H2D → packed attention`

### 7.2 结果是什么

补测中的一个 beta=48 配置选择约 75% token，恢复约 98% attention mass，critical-token recall=1.0，相对 L2 误差约 0.02。但 CPU search 中位数约 3.97 ms，同步 offload 没有胜过 GPU-resident attention。

这里 attention mass 是单层、给定 Q 的数值诊断，不是最终回答质量。不能把“98% mass”解释成“模型有 98% 答题准确率”。

### 7.3 为什么没有停在 CPU exact search

它让我们确认数据通路可行，也显示 search/gather/传输开销很高。要继续，就必须压缩检索工作，而不是只把原本 GPU 能快速扫描的全量计算挪到 CPU。

另一个重要区分：M9/M10 的 GPU dense replay 主要是 PyTorch 数值参考，不是后来真实引擎 paged FlashAttention 的同一个性能基线。

## 八、M10–M11：图检索的想法合理，为什么实际还是选了代表向量全扫描

### 8.1 最初的假设

如果长历史块很多，是否可以通过图导航只访问一小部分候选，而不是每次扫描所有块？

我们实现并比较了 mean representative、每块多真实 Key representative、KNN 图、基于 sampled queries 的简化 query-guided 图，以及 DIPRS-style 遍历和 exact refine。

query-guided 的出发点是：仅凭 K 与 K 的近邻不一定能表达“真实 Q 会查询什么”；用真实 query 的邻居关系建图，可能更贴近检索需求。

### 8.2 比较并没有证明图毫无价值

M10 补测中，一个可比的 degree=16、visited=64、beta=80 配置，query-guided graph recall=0.609，高于 KNN 的 0.500。即使速度不理想，也有检索层面的正向信号。

问题是时间：flat r=4 search 约 1–2 ms，Python 图搜索约 24–53 ms，上述 query-guided 点约 43 ms。

图减少了候选访问，却增加了 Python 控制流、遍历和不规则访存。这是当前实现层级的代价，不是图算法不可加速的数学结论。

### 8.3 M11 必须证明稀疏结果真的决定了生成

此前是单层 replay。M11 把 packed attention 接到真实 attention 输出、o_proj、sampler 和 generation，检查稀疏 decode 没有偷偷调用 dense fallback。

初始报告里有一套短输入结果，但 needle prompt 的末尾 question 被截断，旧的答题一致性不能用作质量证明。后来补测修复了输入，保证 question 完整存在。

修复后的 2K 例子，dense、flat-real、KNN 和 query-guided 都答对。decode p50 约：

| 路径 | p50 |
| --- | ---: |
| Dense paged FlashAttention | 26.4 ms |
| Flat real representatives + CPU offload | 437.2 ms |
| KNN graph + CPU offload | 906.2 ms |
| Query-guided graph + CPU offload | 909.9 ms |

在这个测试上，图路径约比 flat 慢 2.1 倍。flat 自己也远慢于放得下的 GPU dense。

### 8.4 最终选择的逻辑

选择代表向量全扫描，理由是当前块数量、硬件和项目期限下，规则的批量计算比 Python 导航便宜。GPU 批量扫轻量代表，比 CPU 图遍历更容易形成真实可运行的闭环。

没有选择的是当前 Python 图路径，不是永久排除生产 ANN、GPU 图检索或其他硬件上的图方法。

这也是第一次重要的“原先想法变了”：项目不再把“最终必须用图”当目标，而把“在这台机器上总成本更低”当目标。

证据：[M11 历史报告](nanokv_m11_results.md)、[M8–M11 重测](benchmark_backfill_results.md)。

## 九、M12：从短输入实验走向 Qwen3-4B + 128K，但跑通不等于完全验证

### 9.1 为什么更换实现路线

M11 的 dense prefill 本身要先放进 GPU，长输入容量仍受限；CPU 检索也很慢。M12 不只是换个模型，而是调整系统分工：

- CPU 保存每层完整历史 K/V。
- GPU 保存轻量 representatives、sink/recent 和可复用 packed buffer。
- GPU 对 representatives 规则批量扫描。
- CPU 只 gather 被选中的历史。
- prefill 分成 4096-token chunks，限制一次执行的输入规模。

模型改为本地可用 Qwen3-4B，通过 YaRN factor=4 从原生 32K 扩到 128K。

### 9.2 Representatives 为什么取 4 个真实 Key

每个 64-token block、每个 KV head 内选 4 个真实 Key。先选接近块内 mean direction 的真实 Key，再做方向上的 farthest-point coverage。挑选时看归一化方向，实际代表保存原始 Key。

直觉是：一个均值可能抹掉不同方向；少量真实 Key 能提供多方向覆盖，同时维持较轻索引。历史实验支持它作为当时可用的工程折中，但没有完整 r=1/2/4/8 质量—延迟网格来证明 r=4 全局最优。

代表只用于找块。找到块后，attention 使用该块完整的 K/V。准确说法是“检索近似，选中集合内部的注意力使用真实完整数据”，不是“全历史注意力精确无损”。

### 9.3 真实数据流是什么

Decode：单个新 token 的 Q 在 GPU 给历史块打分，选 Top-32，IDs 回到 CPU，CPU gather 完整块，H2D 后与 sink/recent 合并，执行 attention 并生成下一个 token。

Prefill：第一 chunk 无历史，执行 dense；后续 chunk 一边看选中的更早历史，一边看当前 chunk 的 causal K/V。它们不是 32 段互相看不见的独立推理，但也不是每段都完整 attend 所有过去历史。

Chunk=4096 是“一次处理多少输入”；检索 block=64 是“挑历史时的粒度”。分块 prefill 主要限制峰值资源，不自动证明总计算变少或质量不变。

### 9.4 真正交付了什么

历史运行完成 131072-token prefill，即 32×4096 chunks，并生成 16 tokens，无该次运行的 OOM、NaN 或越界，没有 sparse decode 的 dense fallback。

位置到 131072 时曾发生 RoPE cache 越界，修复后才能继续 decode。YaRN/RoPE 的离散位置数值对照支持实现正确，但不是扩展后语言质量的证明。

容量与 payload 的历史记录：

| 项目 | 对应口径 |
| --- | --- |
| 128K 完整 CPU K/V | 约 18 GiB |
| 128K GPU KV 相关 structures/buffers | 按记录张量形状约 1.0 GiB |
| GPU allocated / reserved | 某组运行约 8.62 / 10.78 GiB，包含模型等，不是纯 KV |
| 固定 Top-32 的历史 KV payload | 8 MiB/层，36 层合计 288 MiB/decode token |
| 128K dense-equivalent 全历史搬运 | 约 18 GiB/token，理论比较对象，不是实际执行量 |

因此 98.4% 说的是固定预算历史 KV 相对 128K 全历史理论搬运的 payload 减少，不是比 GPU-resident dense 加速 98.4%。GPU-resident dense 的历史原本不用每步跨 PCIe 搬一遍。

### 9.5 审计发现两个重要漏洞

第一，旧计时用 CPU timer 包围异步 CUDA 操作。selector 已经包含 IDs D2H，报告又把 D2H 加一次。32K/64K/128K 的 153.9/140.9/165.6 ms 是旧 stage-sum estimate，不能称严格 TPOT，更不能据此比较 dense speedup。

第二，prefill selector 把 previous recent blocks 排除，原本是因为这些块应从常驻窗口补回。但当时 prefill 没把它们补回，导致覆盖缺口。重复文本上 greedy 8/8 token 一致没有捕捉这个问题，旧 chunked “full coverage”说法不成立。

这一步提醒我们：功能测试能通过，不等于 attention 布局正确；输出一致也可能只是输入过于简单。

### 9.6 当时尚未证明什么

没有广泛 128K 质量，没有长 decode 超过 recent window 的验证，没有稳定的 128K 多进程性能，也没有可比 dense speedup。M12 是容量与链路里程碑，不是整个研究问题已经解决。

## 十、M13：为什么先测 reuse，再决定不做 GPU previous-set cache

### 10.1 原始优化假设

如果相邻 decode token 常选到相同历史块，可以把上一轮 selected blocks 留在 GPU，只搬新块。这能减少 CPU gather 和 H2D。

先测真实复用率，是为了避免写完缓存后才发现每步都换块。

### 10.2 数据是什么

32K prompt、32-token generation，36 层×30 次 transition，共 1080 个样本：

- reuse mean 32.2%，median 31.2%。
- 平均每步每层 miss 21.68/32 个块。
- 部分层或步骤复用高，但整体每步仍需换掉约三分之二。

历史报告的 gate 写法不一致：范围说明写了 mean≥60% 或 miss≤12/32，结论又用了低于 40% 的否决口径。32.2% 在两种表述下都不满足高复用条件，因此这次不做的方向不变；但不能把 40% 当一个经过优化推导的普适阈值。

### 10.3 为什么这个结果导致换方向

简单 previous-set cache 的理论最大免搬比例只有约 32%，还要付出 lookup、slot 管理、替换和额外 GPU 空间。按照尽快交付的目标，选择先不做它，转向更小、语义不变的 gather 优化。

这里不是“cache 被实测证明永远不值得”，因为并未完成 cache 本体 A/B。实际结论是：这一 workload 的复用率不够理想，不优先投入这个简单设计。更大缓存、不同替换策略、不同 prompt 仍未被排除。

### 10.4 Fallback 为什么简单又有效

原路径先从 CPU KV 取 selected blocks，产生 pageable 临时张量，再复制到 pinned staging。改成 `index_select(..., out=pinned)`，直接写已有 staging，省掉中间数据和额外 copy。

M12 已经有 block-major layout、向量化 gather、可复用 pinned buffer 和 nonblocking H2D。M13 新意不是“第一次 pinned”或“第一次批量搬运”，而是消除 pageable 中间物。

记录中的 gather 从 54.4/102.4 ms 到 36.8/61.6 ms，约下降 32–40%，微测 0 diff，输出和 selected IDs 一致。

### 10.5 18.2% 应该怎样理解

当时 best-observed steady median 是 393.09→321.47 ms/token，下降 18.2%；另一对是 241.14→228.30，约 5.3%。同配置 run 的波动很大，不能把最好的一对当普适或稳定平均收益。

后来补测和 M18 增加了更完整的 paired 证据，比单独引用这组 noisy best pair 更适合解释项目。

证据：[M13](m13_fast_decode_reuse_results.md)。

## 十一、M14：先修 prefill 的语义，再换高效 attention；多 Q 没有被验证为更好

### 11.1 为什么把 correctness 放在优化前

previous recent 缺失意味着 baseline 的历史覆盖本身不对。如果直接调 Q 或 K，质量变化可能来自 bug 与策略同时变化，比较难解释。

M14 把后续 chunk 的输入修成：

`选中的远端历史 + sink + previous recent + 当前 causal chunk`

而且先消费 attention，再保存当前 K/V，避免覆盖旧 recent。此问题后来已修复，不是最终版本仍然存在的缺点。

### 11.2 为什么 FlashAttention-2 是一个明确的优化方向

后续 chunk 不是一个 Q，而是大量 Q，要对 packed K/V 做 attention。使用成熟 FA2 能避免普通 Torch attention 的中间矩阵与额外内存访问，同时保留同一 packed 集合和 causal 布局。

这是“选哪些块不变，怎么在选中块上算更高效”。FA2 和 Torch reference 做了 toy 数值/layout 对照。

最初 q=4 的 8K 单组 generate-wall A/B 为 8381.19→6673.01 ms，下降 20.4%。这包含首 token 采样，不是纯 model-only prefill，且只有一对。

### 11.3 后来补测得到更扎实的性能证据

生产 routing q=1，三个交替 runs/后端，同步的 positive-step prefill wall：

| 长度 | Torch | FA2 | 测量下降 |
| --- | ---: | ---: | ---: |
| 8K | 8162.94 ms | 3826.35 ms | 53.1% |
| 32K | 42003.26 ms | 12718.44 ms | 69.7% |

峰值 allocated 也由约 10.18/10.25 GiB 降为 8.46/8.53 GiB。

第一 dense chunk 没有使用后续 chunk 的 backend 切换，不能归因给这个优化。不同阶段的 20.4% 与 53.1%/69.7%来自不同设置和统计，不互相替换，也不代表所有请求的用户 TTFT 都提升同样比例。

### 11.4 为什么 4096 个 Q 取平均引起怀疑

用户提出的担心很合理：一个 chunk 中可能有事实、问题、格式、噪声，整体均值会稀释方向，不同语义甚至可能抵消。

因此做 q=4：4096 个 Q 分成四段，各自平均，再对同一 block 的分数取 max，最后仍只选一套 Top-32 历史。它增加方向覆盖，不增加历史预算。

### 11.5 为什么没有默认开启 q=4 或重做 representative

16K needle 测试中，两种设置都生成正确答案，但 target block 被选中层数 q=1 为 30/36，q=4 为 29/36。并没有形成稳定优于旧方案的证据。

这是很窄的样本，不能证明整体均值是最佳，也不能证明多 Q 没潜力。只足以决定：先不因为直觉把默认值改掉。也没有证据单独归因给 r=4 representative，所以不同时改 Q 和 representative，以免失去归因能力。

证据：[M14](m14_prefill_results.md)、[后续补测](benchmark_backfill_results.md)。

## 十二、补测与 M15：从“跑完很多步骤”转向“哪些判断真的有证据”

### 12.1 用户质疑得有道理

大量 smoke tests 可以发现崩溃、配置错误、NaN 和明显回归，但它们不能回答性能是否更好、质量是否可靠。

这次补测的价值不在“再跑一些测试”，而在修正计时定义、补对照、拆 prefill/decode、统一模型与 prompt，并保留失败。

### 12.2 Dense 对照改变了项目定位

Qwen3-4B、同口径 16-token generation 的一次 matched 对照中：

| 长度 | Dense steady decode | Sparse steady decode |
| --- | ---: | ---: |
| 8K | 37.40 ms/token | 138.31 ms/token |
| 16K | 35.85 ms/token | 140.50 ms/token |
| 32K | 分配门槛失败，无延迟值 | 147.92 ms/token |

Sparse 在放得下的 8K/16K 远比 dense 慢。它的优势是降低 GPU 历史驻留、扩大当前配置下能处理的输入范围，而不是替换所有短输入 dense 推理。

32K dense 在记录的 scheduler/块分配预算下失败，不是一次 CUDA OOM；也不能据此说所有 16 GiB 配置都绝不可能跑 32K dense。没有尝试的调参空间必须保留。

### 12.3 为什么“64K 3/5”不是可靠质量指标

它表示 5 个手写用例里，有 3 个生成文本包含 expected substring。先前 4/5 和后来的 3/5，生成上限或测量条件并非完全一致，不能直接推断模型总体质量变差。

它能发现失败；不能代表广泛上下文质量，也不是百分之六十的综合语言能力。调大生成预算后某个多 Key 例子通过，还说明过短输出截断会制造假失败。

128K 单 needle 1/1 通过同样只是一个观察，不能抵消复杂干扰任务中的失败。

### 12.4 用户提议 Q block=64，我们实际做了什么

4096-token chunk 划成 64 个 64-token Q 摘要，对每个历史块在这些摘要上取最大分数，最后选共同的 Top-32。

注意，它不是“每 64 个 Q 各自选 Top-32 并独立 attention”，也不是全 Q 与所有历史 KV 的完整精确计算。后两种路线的传输量、复用和内核需求完全不同。

32K 的七个 later chunks 中，q=64 比 q=1 总时间中位数慢约 3.1%；64K 三对方向混合，总中位数差约 1.9%，不足以证明收益。若把高波动、与 routing 无关的第一 chunk 加进来，会得到误导的“改善”。

64K 同五用例，q=1 与 q=64 都 3/5：q=64 找回 90% 位置例子，却丢掉 10% 例子。needle-block layer recall 均值 q=1 为 65.6%，q=64 为 58.3%。

解释上，多 Q 的 max 会把更多“任意一段觉得相关”的块推到前面，但最终共享预算仍 32，有可能不同 query 在抢预算。这个解释是机制假设，不是本实验已经逐层证明的根因。

决定：q=1 保持默认。我们没有否定优化 Q 的需求，只是否定“Q 分得细就一定更好”的直接推断。

### 12.5 PyTorch profiler 找到了一个不改变算法的热点

一次 decode trace 有大量 GPU 活动间隙、scalar checks 和 `cudaStreamSynchronize`。发现逐层 `isfinite(o).all()` 变成 CPU bool，会强制等待 GPU。

把它改成默认关闭的 debug-only 检查，不改变正常 attention 数学。三组 A/B 输出 IDs 一致，各组更快，中位数的汇总从 161.56→149.43 ms/token，下降 7.5%。

这不等于完全解决同步问题，但提供了一个明确、有源码和实测支持的“小改动大影响”案例。

### 12.6 为什么不能用 GPU 空隙直接说利用率

该 PyTorch trace 的活动区间有较多 gap，只说明所记录进程/stream 在事件之间没有记录到工作。它不是整个 GPU 的 SM utilization，也不是 Tensor Core occupancy。

这个区别后来在 Nsight 分析继续保留，避免看见图上的空白就断言“GPU 只用了多少”。

## 十三、M16：动态 decode Top-K 降了 payload，为什么没让 TPOT 变快

### 13.1 想法来自哪里

用户判断：有些 token 只需最近内容，有些需要全文，固定 Top-32 可能浪费。于是尝试 query-dependent 预算：在 K=16/24/32 之间选择。

策略先得到最多 32 个代表分数，用归一化 router-score 的累计权重阈值 0.90 决定保留多少。

这是路由分数启发式，不是实际完整 attention mass。它没有与 recent window 一起校准，所以没有真正证明“此 token 只需要 recent”的判别能力。

### 13.2 性能 pilot 的结果

32K repeated prompt，16-token generation：

| 策略 | steady median 各 run | 平均 K / payload |
| --- | --- | --- |
| Fixed 32 | 144.41、148.56 ms | 32 / 288 MiB |
| Adaptive | 147.70、150.45 ms | 23.48 / 211.33 MiB |
| Fixed 24 | 148.95 ms（一次） | 24 / 216 MiB |
| Fixed 16 | 146.31、135.09 ms | 16 / 144 MiB |

Adaptive payload 下降约 26.6%，但实测比 fixed32 略慢，不能宣称 TPOT 加速。

为什么可能这样？减少 KV bytes 只减少一部分成本；模型投影/MLP、selector、CPU 管理和同步仍在，动态策略本身还有判断开销。哪些因素抵消了收益没有被完全隔离，这是解释方向，不是已完成的因果证明。

### 13.3 质量测试意外发现更严重的问题

初次 8K single needle，dense=100，fixed sparse=20。错误输出经常带上前一个样本的号码。

追查发现同一 LLM 实例连续 generate 时，CPU KV/reps/recent 的逻辑状态没有按新请求清空。修复为“保留已分配 buffer，但清空长度和 selection 状态”，同一组 sparse 回到 100。

这不是 adaptive 算法的成功，是基础生命周期 bug 被修复。修复前结果无效，不能当作最终模型质量，也不能拿来计算 adaptive 的“巨大提高”。

### 13.4 为什么仍然不开 adaptive 默认

修复后的社区 RULER 小样本未发现 adaptive 与 fixed32 差异，但样本少、社区数据按 Llama-2 token sizing、NIAH cap 32 与官方 128 不同。它支持有限回归观察，不支持广泛质量不变。

更关键的是端到端收益没出现，因此默认关闭。把 budget 选择做得更复杂之前，先要证明收益大于策略本身的成本。

证据：[M16](m16_dynamic_topk_results.md)。

## 十四、M17：prefill Top-16 为什么不能简单当成最优方案

### 14.1 为什么把 prefill 与 decode 预算拆开

同样的 Top-32 名字，prefill 是用一整段 Q 的摘要选择“后续 chunk 需要哪些过去历史”；decode 是用当前 token Q 选择历史。

Prefill 漏掉某个事实，会影响后续层和 token 的表示。即使 decode 再找到那段 KV，也不保证能完全补回 prefill 阶段的损失。因此不能用“decode Top-16 看起来没问题”推导 prefill 也安全。

### 14.2 32K later-chunk 时间确实降低了

保持 decode32、只改 prefill：

| Prefill K | 七个 later chunks 两次均值 | 相比 K32 | 历史 payload/later chunk |
| --- | ---: | ---: | ---: |
| 32 | 12802.35 ms | baseline | 288 MiB |
| 24 | 11579.30 ms | -9.55% | 216 MiB |
| 16 | 11476.21 ms | -10.36% | 144 MiB |

K16 与 K24 仅相差约 0.9%，样本不足以确定 K16 更好。第一 dense chunk 不变且波动大，所以重点比较后七段，而不是拿整个 run 的最大好看百分比。

### 14.3 但质量门槛没有通过

五个简单 single needle 都通过，不足以区分预算。更难的三个 multi-key 例子中，K32=1/3，K24=0/3，K16=0/3。

其中一个原来正确的 `1908841`，K24 输出 `1908871`，K16 输出 `1908`。这至少证明同一小测试中降低预算丢掉了一次已有成功。

不能据此估算普遍下降比例，但足够阻止默认启用。我们没有“提升 Q 后就一定能弥补”的证据，所以不把低 K 和未经证明的新 Q 策略绑起来宣称 quality recovery。

### 14.4 为什么不继续只盯 CPU gather

Prefill gather 的 index-select 改造，阶段均值下降约 34.6%，但七个 later chunks wall 仅约 -0.9%，在波动内。

一次 PyTorch profile 显示 GEMM、FA2 和全量新 K/V 的 D2H 才是更大活动项。Gather 只负责 selected history 的一小段，prefill 还要为整个当前 chunk 保存 CPU KV。

这解释了为何 decode gather 可能很有效，而 prefill gather 的同类改进没有明显改变全程：两个阶段的成本构成不同。

### 14.5 D2H 尝试也没有直接成功

试过从 GPU blocks 直接复制到既有 pageable CPU KV buffer，toy exactness 通过，但 later-chunk wall 与旧路径重叠，未形成收益，随后撤回。

Profile 中 pageable D2H 显眼，说明值得诊断，但不等于“换一个 copy 调用就已经优化”。更进一步 pinned staging/异步持久化没有在此阶段完成有效闭环。

发生过一次 K24 repeat 在 NCCL 背景线程崩溃、没有产出结果，成功重试后才计入表格；0.6B 功能 run 也没有混入 4B 性能对照。日志里一处 hard-coded K 打印错误被修正，真实模型配置单独核对。

决定：不再默认减 prefill K；先用专业 timeline 定位系统成本，而不是继续凭预算和总耗时猜。

证据：[M17](m17_prefill_budget_results.md)。

## 十五、M18：Nsight 帮我们选择更有把握的系统优化

### 15.1 为什么要引入它

过去知道“某段很长”，却不清楚 CPU 正在复制、提交工作还是等待。Nsight Systems 可以把 NVTX 标签、CUDA API、GPU kernel 和 memcpy 放在同一时间轴上。

NVTX 名称是我们写入的语义标签；CUDA API 是主机调用；GPU kernel/memcpy 是设备活动。三个视角不是三份可以相加的账单。

GUI 曾因 Windows 安装权限被阻塞，WSL CLI/GUI 先可用，后续用户安装了 Windows 原生 GUI。我们在 WSL 采集、复制报告给 Windows 阅读；旧 M18 文档里的“原生 GUI 未安装”是阶段历史，不应当作当前安装状态。

### 15.2 Trace 改变了对 prefill 和 decode 的理解

一个 16K later-prefill chunk 中，主要 GEMM kernel families 总计约 747 ms，FA2 约 313 ms，全 KV D2H 约 177 ms，H2D 约 25 ms。

一个 32K legacy decode trace，CPU gather 与 selector host ranges 各约 46 ms；packed attention GPU kernel time 只有约 6.53 ms，虽然对应 CPU NVTX 范围约 27.95 ms。

这些是带 profiler 的诊断值，不能直接做未 profile 墙钟百分比拆分。但足以提醒我们：先重写 attention kernel 未必能解决 CPU/GPU 同步与搬运主路径；而 prefill 的主要计算也不只在 retrieval。

### 15.3 为什么把已有 direct gather 变成默认

它的语义不变，工程风险较低，且补了三组 fresh-process A/B：

| Pair | Legacy steady median | Direct gather | 配对下降 |
| --- | ---: | ---: | ---: |
| 1 | 162.76 ms | 140.59 ms | 13.6% |
| 2 | 191.41 ms | 155.50 ms | 18.8% |
| 3 | 175.84 ms | 155.10 ms | 11.8% |

三个 pair 都同方向，生成 IDs 一致，平均配对相对下降约 14.7%。这是一个 repeated 32K prompt 的结果，不是广泛 workload 均值。

之前补测得到汇总 11.7%，M18 得到 14.7%，不说明代码自动又提高了 3 个百分点；不同时间和统计定义的测量有波动，不做这种相减。

Direct pinned gather 因此从 opt-in 变成 M12 默认。它没减少 Top-K、没变 retrieval、没减少 288 MiB payload，而是减少 CPU gather 的中间工作。

这也是“保留真实优化”与“只追新点子”的区别：有明确对照且复测同方向的小改动，比复杂但没有总收益的方案更值得进入默认路径。

证据：[M18](m18_nsight_gather_results.md)。

## 十六、M19：CUDA Graph 降低提交开销，为何仍没有默认启用

### 16.1 假设与适用范围

Decode selector 有多个 GPU score/reduce/Top-K 操作。捕获后 replay，有机会用较少 API 提交同样计算。

只捕获 selector，不把 CPU gather 和整个模型都装进图，也不是先前的 ANN 图检索。“图索引”和“CUDA Graph”是两个无关概念。

### 16.2 第一个实现为什么有明显问题

每层进入 `torch.cuda.graph` convenience context，带来同步、GC 和 allocator 清理等额外动作，36 层 setup 合计约 4.7 秒。

改为 warmed side stream 和 raw capture 后，32K setup 降到约 200–296 ms。这是具体实现成本被修掉，不是模型稳态自然快了 4.5 秒。

### 16.3 稳态结果为何矛盾

修正后 fresh-process 三对 median 相对 eager：

- 一对快 13.7%。
- 一对慢 22.2%。
- 一对慢 2.7%。

输出和 36 层所有记录 selected IDs 都一致，但速度方向不稳定。

同进程 ABBA 中 graph decode median 快约 15.1%，selector host time median 降约 69.6%。这有正向信号，不过 eager/graph 对应不同 token positions，不是完全同 query 的配对。Setup 和额外 allocated/reserved memory 也要计入真实使用成本。

### 16.4 决策

能证明提交结构减少，不能证明 fresh-process 一致加速。保留 opt-in，默认关闭。不拿最好的一对覆盖另两对，不拿 ABBA 当最终服务效果。

64K 短对照也没有把它变成可靠收益：median 接近，graph mean 和 total decode 更高。报告中的 trace node 模式还严重扰动时间，所以只用 light trace 看结构。

证据：[M19](m19_selector_graph_results.md)。

## 十七、M20：用户“查到就搬、用完就释放”的思路，我们怎样落实，为什么没有变快

### 17.1 首先澄清：显存复用不等于动态规划

把 selected KV 临时搬入 GPU、使用后复用 buffer，原路径已经在做。它不是必须把 CPU 原副本再搬回去；若 GPU 临时副本无需保留，可以复用 GPU 空间。

若是“全历史分批搬入、每批 attention 后释放”，那是另一种 streaming full attention，需要在线 softmax/LSE 合并，不能对每批 softmax 独立求平均。它可以缓解峰值显存，但不减少总全历史传输。

若是“按查到的 selected blocks 流水传输并计算”，则要处理 gather、拷贝和消费依赖。这条流水是当时更窄、可验证的系统候选。

我们没有把这类 buffer 生命周期与调度问题当成已经实现的动态规划算法。

### 17.2 为什么 decode 的 Q 并不像 prefill 那样容易拆

单序列 decode 一次通常只有一个新 token Q，prefill 才有一整段 Q。可以沿 heads 或 KV partitions 寻找并行，但不能照搬“把很多 query tokens 切开”的 FA2 理由。

所以没有直接宣称按 Q=64 做 decode 的收益，也没有实施一个完整新 attention kernel。

### 17.3 先做 gather 微测，发现整块复制并不稳定更好

相比 flat token-index，whole-block gather 在 1/2 CPU 线程看起来较快，但在 4/8 线程更慢；8 线程约慢 2 倍。

这些是合成源、轮播真实 IDs 的微测，且有背景负载。不能断言所有 block-major 操作都慢，但不支持把 whole-block 路径接成默认，实施候选随后撤回。

### 17.4 选择最简单的 K/V pipeline

先 gather K、提交 K 的 H2D，再在 CPU gather V，最后提交 V 的 H2D。复用既有 staging/stream，不先建设复杂双缓冲线程系统。

小型 gather+H2D 微测中，K/V overlap 约 1.40→1.19 ms，约 15% 正向信号。chunk-16 略快一点，但不够支持额外复杂度，因此选简单 K/V overlap 进入真实模型。

### 17.5 真实模型的结果与微测不同

第一批 fresh pair 方向相反。用户确认同时有其他 GPU 或高负载任务，因此不能将这批结果作为稳定收益。

电脑空闲后又完成 ABBA 和两组 fresh pair。pipeline 的 decode median 相比 serial 分别高约 4.13%、2.67%、3.06%，没有加速证据。此处仍不能仅凭这几次 run 证明 pipeline 因果上必然更慢。

Nsight 按 correlationId 对齐，确认 36/36 层 K H2D 与 V CPU gather 确实有交集。这证明实现真的重叠，不是只设置了 nonblocking。

### 17.6 为什么“重叠了”不等于“节省了”

如果重叠的工作不在总 critical path 上，或者提交/CPU gather 的调度成本变化抵消了好处，就可能没有净收益。也可能有带宽、缓存与主机调度因素。

这些是待验证解释；我们没有硬件计数器和完整因果隔离，不能直接宣布“就是内存争用”。

决定：K/V pipeline 默认关闭，whole-block 路径撤回。保留实验和失败，不把 13.7 ms 的 trace 交集改写成“每 token 省 13.7 ms”。

证据：[M20](m20_gather_results.md)。

## 十八、M21：最后有效的小优化不是搬更多 KV，而是少做无意义同步

### 18.1 问题如何被发现

继续看 selector，发现除了必需的 selected IDs D2H，还存在 protected-block mask 构造、GPU bool filtering 和高级索引赋值触发的额外小 copy/wait。

sink/recent 的保护集合在短 decode 中可复用，没必要每层每步以同样方式重新产生这些 GPU 元数据工作。

### 18.2 改了什么、不改什么

在 CPU 过滤有效 protected indices，缓存 GPU indices，用 `index_fill_` 应用 mask，并覆盖 reset、集合更新和历史失效。

不改评分公式、不改 Top-K、不改 prefill，不改变 selected KV 的 288 MiB。真正优化的是辅助元数据路径。

### 18.3 为什么证据链比“GPU 空了很久”更具体

Captured step 的 selector pre-ID 范围，额外 108 次 `cudaMemcpyAsync` 与 108 次 `cudaStreamSynchronize` 消失。IDs D2H 的 36 次 copy/36 次 wait 仍保留，因为 CPU gather 仍需知道选择结果。

说明这次不是把所有同步神奇消掉，而是移除可避免的一部分。

两组未 profile、不同顺序 fresh pair 的 Drop4 mean：

| Pair | Serial | Static mask | 下降 |
| --- | ---: | ---: | ---: |
| 1 | 128.265 ms | 118.769 ms | 7.40% |
| 2 | 131.264 ms | 117.537 ms | 10.46% |

对应 median 下降 7.86% 和 10.57%。生成 token 与 36×31 selected histories 全一致；额外 GPU allocated 仅约 18 KiB。

### 18.4 为什么没有说“项目总加速 25%”

M18 与 M21 使用不同运行及 baseline，百分比不能直接相加。也没有多 prompt、多长度、宽负载的大样本，所以 M21 保留 opt-in 默认关闭。

它是一个已实现、局部配置有数据支持的可选优化。M22 又补了它的回答等价证据；这不等于已经自动启用，更不等于修复稀疏算法质量。

证据：[M21](m21_selector_static_mask_results.md)。

## 十九、M22：质量证据的最终补齐，为什么结论比原先更“难看”但更可信

### 19.1 为什么不再自己扩写五个 needle

用户反复质疑 3/5 口径。问题不只是样本少，还包括手写输入、substring 判定、生成截断、无 dense 对照和任务覆盖单一。

最终采用固定版本 NVIDIA RULER 的生成器与原始评分，使用 Qwen3-4B 自己的 tokenizer 生成输入，而不是沿用之前按 Llama-2 长度生成的社区文件。

RULER 在这里是长上下文合成任务集合；我们选了单目标检索、相似 Key 干扰检索、变量跟踪。并未跑整个 RULER、NeedleBench 或 LongBench，所以这三个名称不能写成“全部完成的评测平台”。

固定版本与协议可以降低自由修改测试的空间；使用公开工具不代表每个分数都有无限解释力。

### 19.2 评测设计回答两个不同问题

问题 A：同输入上 sparse 相比 dense 是否丢质量？

问题 B：M21 系统优化是否改变原 sparse 的回答？

8K 3 个任务，各 20 prompts，dense/sparse/M21 三臂。32K 2 个检索任务，各 10 prompts，sparse/M21 两臂。共 80 个独立 prompts、220 次 generation、13 arms，seed=42。

32K dense 没有运行；不为它补造分数，也不把“没跑”写成“模型不行”。

### 19.3 官方评分到底衡量什么

采用固定 NVIDIA `string_match_all` 及原 postprocessing。

单参考 NIAH：答案字符串是否出现在生成文本里，平均得到百分比。相似 Key 任务是在很多干扰事实中问一个指定 Key，仍是一个参考答案，不是多答案检索。

Variable tracking：每例有五个参考变量，计算匹配多少参考项。97 分不是 97% 样本整条链完全正确，而是参考项平均召回。

Substring metric 也可能给“答案出现在冗长或部分错误文本里”计分，因此保留原输出、参考、token IDs 和逐样本结果，不只保留总分。

### 19.4 最终结果

| 上下文/任务 | 样本数 | Dense | Sparse baseline | M21 |
| --- | ---: | ---: | ---: | ---: |
| 8K 单目标噪声检索 | 20 | 100 | 100 | 100 |
| 8K 相似 Key 干扰检索 | 20 | 100 | 80 | 80 |
| 8K 四跳变量跟踪：参考项召回 | 20 | 97 | 92 | 92 |
| 32K 单目标噪声检索 | 10 | 未跑 | 100 | 100 |
| 32K 相似 Key 干扰检索 | 10 | 未跑 | 0 | 0 |

Sparse 与 M21 生成 IDs、原文、每例得分在 80/80 prompts 完全一致。M21 没有在这个 screen 上增加可观察的答案变化。

但是 8K dense-paired 任务清楚显示 sparse 的损失；32K 干扰任务 0/10，说明当前配置在此任务上不可靠。没有 32K dense，就不能完全区分模型、模板和 sparse 各自的作用。

### 19.5 为什么不能立刻归罪于 mean Q 或 r=4

输出失败只告诉我们结果不对。可能是 prefill 未选中事实、decode 路由遗漏、预算竞争、已选中却 attention/表示没保住，也可能涉及模型与模板。

我们没有完成对这些失败样本的 chunk/layer target rank、attention mass 和控制 ablation，因此不宣称找到了唯一根因。

用户提出“平均 Q 太粗糙”是一个合理候选，但不是这组 quality 结果已经证明的结论。

### 19.6 这个质量证据有什么局限

只选三种 task family，20/10 样本、一个 seed，旧 completion 模板，自定义 NanoKV inference adapter；不是完整官方 leaderboard，也没有广泛 64K/128K 质量。

NIAH cap=128，VT cap=30；所有 220 generations 都返回 cap 长度。记录的是 cap-hit，并非额外证实了独立 EOS 停止原因，也不宣称自由长度生成的表现。

最终独立审计重算分数、核对 tokenizer 长度/输入 hash、参考存在性、220 个输出 ID 的解码、source hashes 和结果路径。它不重跑模型、不改答案。

最终 61 项 focused tests 通过，是当次选中的相关回归，不是声称整个仓库所有历史测试重跑了一遍。

### 19.7 为什么这里停止，而不是继续追质量

用户明确选择补质量证据、合并 main、结束项目，之后有面试需求再启动。M22 的失败因此保留并公开，不删样本、不换更容易 prompt、不临时提高 budget 把成绩补漂亮。

收尾不是“所有问题已经优化到头”，而是当前交付范围完成，收益和风险都有证据，继续工程投入暂时停止。

证据：[M22 全报告](m22_quality_closeout_results.md)、[公开逐样本结果](../benchmarks/results/m22_quality/20261003-closeout/)。

---

## 二十、回头看：哪些想法没有走成，原因属于哪一类

| 路线/想法 | 有无实施证据 | 没有成为最终默认的原因 | 不能过度推论 |
| --- | --- | --- | --- |
| 所有短前缀都走 CPU restore | 有真实 benchmark | 短前缀固定/传输成本可能高于重算 | 不是 CPU cache 对所有场景无用 |
| CPU exact search | 有单层与生成实验 | 扫描/同步成本太高；主要作 oracle | 不是 exact attention 质量不好 |
| Python KNN/query-guided graph | 有 matched 对照 | 遍历约几十 ms，显著慢于 flat scan | 不是否定 AlayaDB/RoarGraph |
| Llama 3.2 checkpoint 主线 | 遇到授权阻塞后改线 | 权限与时间；可用 Qwen4B 优先 | 不是比较后认定 Llama 性能差 |
| 简单 previous-set GPU cache | 测 reuse，未建 cache 本体 | 平均约 68% 块要换，优先级不高 | 不排除更大多步 cache |
| q=4 / q=64 | 已实现及测量 | 未观察到稳定速度/质量改善 | 不证明所有 Q 设计无潜力 |
| Q 图检索 | 讨论过，无独立成功闭环 | 未证明其能解决当前路由质量/成本 | 不得当作已完成 benchmark |
| 所有 Q×全部历史 K 的 full route | 讨论过，没有完整新实现结论 | 不再是轻量路由，算力/容量/搬运账须重算 | 不能称“已证明不可能” |
| 每个 Q block 独立历史路由 | 当前 q=64 不是这种实现 | 更复杂路由/union、重复搬运与 kernel 需求 | 仍属未验证候选 |
| 自写块稀疏 Tensor Core kernel | 没有最终完整性能闭环 | 没有先证明 GPU attention 是关键成本 | 不等于没使用 FA2 等已有 kernel |
| 动态 decode Top-K | 已实现 pilot | payload 下降，无 TPOT 改善 | 不说明所有动态预算无效 |
| Prefill Top-24/16 | 已实施 | 有速度信号，小样本 matched 质量退化 | 不能宣传 quality-preserving |
| Prefill direct pageable D2H | 实施后撤回 | 正确但 whole prefill 无可靠收益 | 全量 KV 异步 offload 未被否定 |
| Whole-block CPU gather | 微测/试验后撤回 | 常用线程档不佳，没可靠模型收益 | 不证明所有 layout 更换都无用 |
| Selector CUDA Graph | 实施并修 setup | fresh-process 加速方向不一致 | 同进程收益不是宽负载定论 |
| K/V pipeline / chunk copy | 微测与模型/trace | overlap 真实，模型 wall 无净收益 | 不证明 overlap 天生无效 |
| KV quantization/compression、SSD、TP>1 | 没有当前项目的完成证据 | 没纳入已授权闭环，不是经实测淘汰 | 不可虚构为比较后弃用 |

有些路线是被数据否决为当前默认，有些只是没有时间/授权/证据走完。两类必须区别。

## 二十一、最终留下的成果与状态

### 21.1 成果分成五层，不混成一个“大加速”

1. 可用性：真实 CPU/GPU 分层 KV、稀疏 attention 进入 generation；历史上完成 128K 输入与短 decode。
2. 正确性工程：前缀身份与恢复事务、RoPE 边界、previous recent 覆盖、每请求 sparse reset。
3. 实际性能：FA2 later-prefill、默认 direct pinned gather、debug scalar sync 移出默认热路径、可选 static mask。
4. 决策证据：graph/cache/低 K/Q 分块/CUDA Graph/pipeline 的正负结果与限度。
5. 可复查交付：版本化 driver、阶段报告、backfill JSON、M22 逐样本公开证据，最终成果合并 main。

### 21.2 当前参考设置与实际开关

近期对照主要是 Qwen3-4B BF16，TP=1、单 sequence、eager、YaRN，block64、r4、Top32、sink64、recent512、chunk4096、q-summary=1。这些是实验 reference setup，不等于所有 Config 字段在任意调用下都会自动组成这套设置。

- Sparse/M12 是 opt-in，不是整个上游默认都改成 sparse。
- M12 direct pinned gather 默认 On，可关闭作对照。
- Later-prefill FA2 后端保留 Torch reference。
- Multi-Q、减 prefill K、adaptive decode、selector Graph、K/V pipeline、static mask 等仍为可选/实验路径，不能假装全启用了。
- 正常有限性 debug check 默认 Off，调试可启用。
- 最终 M22 的 M21 arm 显式开启 static mask；baseline 没有开它。

### 21.3 一段可诚实复述的最终定位

这是一个基于 nano-vLLM 的单机、单序列长上下文工程原型：在 16 GiB GPU 上将全量历史 KV 放到 CPU，用 GPU block representatives 选择少量历史，再执行 packed attention。它验证了容量扩展、真实生成以及几项系统优化，也记录了图遍历、动态预算和搬运 pipeline 未形成净收益的结果。它并不普遍快于 dense，且在标准化的 bounded quality screen 中存在可见损失，复杂干扰检索仍不可靠。

## 二十二、最终仍然存在的缺点：按重要性排序

### 22.1 最重要：任务质量不可靠

8K dense-paired 已检测损失；32K distractors 0/10。不能因为 single needle=100 就说全文理解好，也不能因为 128K 执行成功就说有效上下文为 128K。

漏信息来源尚未隔离；这比再抠几个 ms 更影响“能不能用于实际复杂任务”。

### 22.2 历史 budget 和 Q summary 都是启发式

统一 Top32、chunk mean Q、全局 temporal ranking 和 r4 都是工程折中，而不是经过充分任务覆盖证明的最优设计。

不同 query/head 可能需要不同历史；共享预算可能让方向竞争。该机制值得研究，但现阶段是未验证的原因候选。

### 22.3 CPU 解决显存问题，却把压力转到 RAM 和传输

128K 全 KV 约 18 GiB CPU memory，仍需加上模型/索引/staging/进程开销。WSL 可用主存也是限制，因此没有补一整套 128K 稳定性实验。

Top32 每 decode token 仍约 288 MiB KV H2D；ID D2H → CPU gather 的依赖仍在。静态 mask 不会消掉这条本质依赖。

### 22.4 Sparse 在 dense 能放下时更慢

当前系统主要是容量折中，不是低 latency dense replacement。真实 dense 8K/16K 的 decode 对照比 sparse 快很多。

如果未来落地 serving，必须决定短输入是否走 dense、什么时候 offload、怎样解释 latency-quality-memory trade-off；当前未完成这种自动策略。

### 22.5 只验证有限的短 continuation

recent window 为 512，近期生成远短于这个范围。历史审计提示 M12 新 decode tokens 主要放在 recent，不像 prefill 一样持续沉淀到 CPU history/reps；超出窗口的行为未完成验收。

不能宣称任意长度续写已经可靠。未来若要长 continuation，应先核查现有源码和 ring rollover 语义，而不是只改 max_tokens。

### 22.6 性能工作负载窄，测量噪声明显

主要优化 A/B 用 repeated-text、单序列、固定模型和较短生成；run-to-run 波动存在，M20 还有已记录的背景负载污染。

未覆盖多真实 prompt、并发吞吐、服务 P50/P95、长时间运行和多模型。已有加速数字不能被自动外推。

### 22.7 Profiler 到了 timeline 级，没有完整 kernel 硬件计数器闭环

使用过 PyTorch profiler 和 Nsight Systems，能对齐 API/copy/kernel，确认实际 overlap。

没有完成一个基于 Nsight Compute 的寄存器、shared memory、occupancy、Tensor Core 或 DRAM counters 驱动的自写 sparse kernel 优化。Device coverage 不是 GPU hardware utilization，不能混用。

### 22.8 工程能力还不是生产栈

近期验证为单 sequence、TP1、eager，没有 continuous batching、多请求并发、多 GPU、跨进程 cache、量化 KV、SSD/remote tier 的完整支持。

Prefix reuse 与 sparse 主线不能组合开启。TinyLlama adapter 验证不能当广泛 Llama 3 适配质量证明。

### 22.9 可复现交付仍有边界

部分 early scripts、完整 prompts、预测、日志和大型 nsys traces 在 ignored `bench_logs/`，不全部在 GitHub。公开有关键 reports、versioned harness 和 compact results，clone 不等于得到所有旧本机原始文件。

模型、venv 和 WSL disk 也不应提交。若未来换机器，先保存所需 evidence，再重建依赖，不能用清理 ignored files 的命令误删它们。

## 二十三、哪些历史问题已解决，不能继续当当前缺点讲

| 历史问题 | 后续处理 | 当前如何描述 |
| --- | --- | --- |
| Decode position=131072 RoPE cache 越界 | M12 已修复 | 是历史定位/修复，不是持续阻塞 |
| Chunked prefill previous recent 被排除却未补回 | M14 修复并补 layout/reference tests | 当前恢复 recent；旧 full-coverage 证据仍不能复活 |
| 新请求继承 prior CPU KV/reps/recent | M16 reset | 已解决；pre-fix quality outputs 无效 |
| 逐层 finite CPU bool 强制同步 | M15 改 debug opt-in | 默认 hot path 不再强制它 |
| Pageable gather 临时张量 | M13 实现，M18 默认开启 direct gather | 老路径留作 A/B |
| CUDA Graph convenience capture 4.7s setup | M19 换 raw capture | setup 大减，但稳态净收益仍未稳定 |
| README/面试 guide 停在旧阶段 | M22 更新首页、limitations 与历史标签 | 旧区块仍有历史数字，阅读要看阶段 |
| 只有手写 3/5 质量 headline | M22 增固定 RULER-derived screen | 证据变好，不意味着实际质量变好 |

性能改善、正确性修复和证据质量改善是三种不同进展。

## 二十四、未来展望：如果有明确需求再启动，怎么选方向

以下只是一组基于现有失败和约束的建议，不是当前 backlog，也不授权运行或付费 compute。

### 24.1 第一优先级候选：定位并改善复杂检索质量

如果面试官问“为什么 32K 干扰失败”，最有价值的下一步不是立刻换 representative。

先固定几个已公开失败样本：

1. 让相同模板/同一模型的 dense control 可运行，或明确分配更大 GPU 的成本与权限；必要时先在可配对的 8K 定位。
2. 记录目标 block 在 prefill 各 chunk/layer 和 decode 的 rank、是否 selected。
3. 选中却错：继续查 attention/output，不再只怪 routing。
4. 未选中：只改一个因素，例如 Q summary、budget、representatives 或 head/segment quota，做同预算对照。
5. 保留原失败和新的固定评测集，不能挑成功样本。

更细的 Q、max/min 多方向摘要、真实 Q 代表、每 Q block 独立路由、每 head 配额、动态预算都可成为候选。但应先查漏在哪，再选最小实验。动态 K 的阈值也应有质量校准，不能把 router score 当真实概率。

### 24.2 若需求是容量或长生成：先补语义，再扩大规模

长 continuation 首先确认 decode token 历史追加、代表更新、protected-set 更新、recent rollover 和 reset。否则扩大 generation 长度只是把未验证行为放大。

CPU memory 成为限制时，KV quantization/compression 是候选，但会引入新的数值与质量 trade-off，需要 dense/未压缩对照；不能仅看容量减半就宣称项目更好。

### 24.3 若需求是更低 decode latency：针对未重叠 critical path

仍有必要 IDs D2H/CPU gather，以及模型计算成本。先用未 profile matched wall 与 timeline 判断哪段有可回收空间。

GPU resident cache 可以重新评估更大容量、多步历史和非重复 workload，但 M13 的低 previous-set reuse 是已有反证，不能忽略。预测 next-layer/next-token 选择可能带来错预测额外搬运，复杂度要计入总成本。

Pipeline 若重启，要先回答 M20 为何 overlap 没转化成 wall benefit，而不是增加更多 streams 期待自然改善。

### 24.4 若需求是更低 prefill latency：围绕真实大项，而不是只砍 K

已有 FA2 是明确收益。进一步可研究全量新 K/V D2H 的 pinned staging/异步持久化，与模型/GPU工作交叠；需要处理 CPU source lifetime、event completion、索引构建时机和 RAM/pinned 容量。

Q block 独立路由加 packed/block-sparse attention 也可能减少 irrelevant work，但数据量、union 放大和 repeated KV copy 都需实测。真正自写 kernel 前，先量清楚 routing/transfer/attention 各自可节省上限。

不能宣称“prefill 到极限了”：我们只证明某些低预算/Q-summary/gather候选没有成为可靠默认，而不是证明所有 prefill 算法都无空间。

### 24.5 若需求是 kernel/Tensor Core 能力：另开有清晰目标的实验

这张卡是 Ampere SM86。应选实际 dispatch 和热点内核，先数值参考，再 profiler/counters，再优化，不能仅因“块内可以矩阵乘”就推断 Tensor Core 已满载。

Nsight Systems 回答等待和调度；Nsight Compute 更适合具体 kernel 的硬件分析。没有指定 kernel bottleneck 前，不把 kernel 重写当唯一方向。

### 24.6 若需求是产品落地：路由、质量和并发的优先级更高

先定义服务需求：最大长度、允许质量损失、首 token latency、吞吐、并发、成本。Dense 能放下的请求可能走 dense；超长请求选择 offload/sparse，但需要可靠告知 trade-off。

完整 RULER/真实任务集、LongBench 等可以作为更广质量验证的后续选择；这里未完成。不能先承诺“生产可用”，再把真正决定产品体验的能力留到后面。

### 24.7 为什么现在不自动执行这些方向

项目目标是形成真实、能解释的工程成果，不是无限研究。用户已选择阶段结束，后续应围绕具体面试问题或明确新目标，以最小的可复现任务重启。

我们有足够信息说明现阶段的优点与失败；不需要为了避免“项目不完美”而把所有候选都再做一遍。

## 二十五、如果重来一次，我们应怎样更有效

这部分是复盘建议，不是声称当时完全按此流程实施。

- 每轮先写清“想改善哪项成本、为什么它可能占关键路径”，再动实现。
- 算法质量改变和系统路径优化分开。例如更换 Q 与减少 K 不一起做。
- Smoke 用于快速 fail-fast，不能代替 benchmark；相关小测试通过后，把时间用于 matched 对照与 artifact 检查。
- 每个 quality 样本隔离状态，先 assert needle/question 存在，生成预算统一。
- Dense replay、真实 paged dense、stage timing、prefill wall、decode TPOT、TTFT 与整次请求 wall 分别命名。
- 固定 source/config/input hashes，避免只记 dirty tree 的旧 HEAD；记录首步和稳态，不把 setup 隐藏。
- Profiler 只用于定位；速度结论来自未 profile runs。背景负载影响必须记录。
- 新想法先做成本上限分析，再做微测，最后接真实模型；微测好不等于端到端好。
- 失败后区分“原因证实”与“可能解释”；不能事后给每次选择编一个完美故事。
- 报告保留否定结果和未测项，不靠 best run、换 prompt 或改指标制造胜利。

最需要改进的不是“有些猜想没成功”。研究工程允许猜想失败。真正需要改进的是少用小样本或不一致计时提前得出强结论，让每次进入默认路径的变更都有足够证据。

## 二十六、给未来接手者的操作交接

### 26.1 环境与仓库

| 项目 | 权威值 |
| --- | --- |
| Windows 协调目录 | `C:\Users\28898\Documents\ChatGPT\nano-vllm找实习` |
| WSL distribution | `NanoVLLM-Ubuntu`，不能换 generic Ubuntu |
| Repository | `/opt/nano-vllm` |
| Python venv | `/opt/nano-vllm/.venv` |
| CUDA | `/usr/local/cuda-12.8` |
| GPU | RTX 3080 Laptop，16 GiB，SM86 |
| 已确认 package 环境 | Python3.12.3，Torch2.7.1+cu128，Triton3.3.1，FA2 2.8.3.post1 |
| HF_HOME | `/opt/models/.cache/huggingface` |
| 环境检查模型 | `/opt/models/Qwen3-0.6B` |
| 近期可比实验模型 | Qwen3-4B snapshot `1cfa9a7208912126459214e8b04321603b3df60c` |
| GitHub | https://github.com/Tao20060602/nano-vllm-kv-optimizer |

4B 模型路径：

`/opt/models/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`

环境契约的 0.6B 不应被当成近期 4B A/B 的替代模型。不要用 Windows Python/原生 CUDA 诊断项目，也不要安装重复模型和依赖。

### 26.2 接手时只做必要的只读检查

先阅读仓库 `AGENTS.md` 与 `docs/environment.md`。Windows 入口：

```powershell
.\scripts\wsl-nanovllm.ps1 -Command "bash scripts/check-wsl-environment.sh"
.\scripts\wsl-nanovllm.ps1 -Command "git status --short --branch"
.\scripts\wsl-nanovllm.ps1 -Command "git log --oneline -8"
```

不需要仅为了理解项目重跑整套 pytest 或模型 benchmark。获得具体复跑/修改任务后，选相关测试和 fresh output names，避免覆盖旧 evidence。

### 26.3 看哪些文件，就能找到原始依据

| 想了解什么 | 文件 |
| --- | --- |
| 当前正式收尾与范围 | [PROJECT_CLOSEOUT](PROJECT_CLOSEOUT.md) |
| 当前局限 | [limitations](limitations.md) |
| 历史补测、dense边界、Q64、debug-sync | [benchmark_backfill_results](benchmark_backfill_results.md) |
| Reuse与direct gather起点 | [M13](m13_fast_decode_reuse_results.md) |
| Recent覆盖、FA2、多Q | [M14](m14_prefill_results.md) |
| Adaptive K、跨请求reset | [M16](m16_dynamic_topk_results.md) |
| Prefill预算/gather/D2H | [M17](m17_prefill_budget_results.md) |
| Nsight与gather默认 | [M18](m18_nsight_gather_results.md) |
| Selector graph | [M19](m19_selector_graph_results.md) |
| Gather/pipeline负结果 | [M20](m20_gather_results.md) |
| Static mask | [M21](m21_selector_static_mask_results.md) |
| 最终quality、协议、失败样本 | [M22](m22_quality_closeout_results.md) |
| 公开quality逐样本artifact | [results](../benchmarks/results/m22_quality/20261003-closeout/) |

若阅读的是 Windows 导出版，以上相对链接对应 WSL 仓库 `docs/` 或 GitHub 相同路径；可先从 [GitHub 文档目录](https://github.com/Tao20060602/nano-vllm-kv-optimizer/tree/main/docs) 打开已发布阶段报告。

工程入口只需知道：`nanovllm/sparse/m12_runtime.py` 为主线 runtime，`nanovllm/kvdb/` 为旧 prefix reuse，`benchmarks/` 为 driver，`tests/` 为回归。本文不要求读者掌握逐行实现。

Local ignored logs/traces 位于 `/opt/nano-vllm/bench_logs/`；Windows 的 GUI 报告副本在协调目录 `NsightReports/`。不要删除 ignored artifacts，不要提交模型/venv/整个 WSL 磁盘。

### 26.4 新对话应先准确复述

> NanoKV 已完成 M22 质量收尾并发布 main；当前任务仅是说明、交接或明确的复现，不自动重新优化。项目从 CPU prefix reuse 转到单序列 CPU full KV + GPU representatives 的 sparse 长上下文原型。图检索因当前 Python 实现开销没有成为主线。M14/M16 已修复 recent 覆盖和跨请求状态。FA2、direct pinned gather 和可选 static mask 有限定配置下的证据；adaptive K、selector CUDA Graph 和 K/V pipeline 未成为可靠默认。128K 跑通不等于 128K 质量可靠，M22 显示 sparse 相比 dense 的质量损失以及 32K 干扰检索失败。下一步必须由用户的具体需求决定。

如果接手者仍认为所有旧问题未修复、当前已无损、pipeline 已加速、main 停在 M12，或自动恢复 Llama gated checkpoint 路线，就没有正确理解项目。

## 二十七、最后的总评

这不是一条“每个点子都正确、每个改动都提高性能”的直线。

我们经历了从跨请求复用到长上下文 offload，从图导航到批量 representative scan，从减少历史预算到承认质量退化，从想靠 overlap 获益到实际发现没有净收益，再到借助 trace 删除多余同步。过程中还修正了不可靠计时、覆盖遗漏和样本状态泄漏。

最终最值得保留的不是一个最好看的百分比，而是这套因果链：在具体硬件约束下提出候选，用有限但明确的实验做比较，允许数据推翻想法，区分性能、容量、正确性与质量，最后公开仍然失败的任务。

已经做出来的部分有价值，没解决的部分也清楚。未来若重启，我们应该从具体未解决问题出发，而不是从“再找一个看起来激进的优化点”出发。
