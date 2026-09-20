# NanoKV：KV Cache 复用项目执行提示词

把本文件从头到尾读完后再开始行动。你是该项目的实现者，不是只做建议或讲解的顾问。持续工作，直到下面的 Definition of Done 全部得到代码、测试和运行结果的证明。

## 1. 项目背景

用户已经读完 nano-vLLM 源码，目标是申请 AlayaDB 的 AI Infra 实习。AlayaDB 的方向包括长上下文推理、KV Cache 复用、CPU-GPU 协同和 KV Cache + Sparse Attention。

本阶段只完成一个主题：**KV Cache 复用**。

项目以公开的 nano-vLLM 为推理引擎底座，但要实现属于本项目的新子系统。当前 nano-vLLM 已经有 GPU block 级 prefix caching，所以“跑通现有 prefix cache”不算完成。

本项目的核心增量是：

1. 将 nano-vLLM 隐式、GPU 驻留的 prefix cache，重构成显式、可观测的 reusable-context subsystem。
2. 支持跨请求 longest-prefix reuse。
3. 支持完整 KV block 的 GPU → CPU 保存以及 CPU → GPU 恢复。
4. 即使原 GPU physical block 已被淘汰，后续请求仍能从 CPU cache 恢复共享前缀。
5. 精确记录复用 token、重算 token、lookup、传输、prefill 和 TTFT。
6. 用确定性生成和 benchmark 证明正确性及性能边界。

项目暂定名为 **NanoKV**，描述为：

> An educational CPU-backed reusable KV-cache subsystem for nano-vLLM.

不要声称“从零实现了 nano-vLLM”或“完整复现了 AlayaDB”。必须明确保留 nano-vLLM 的 attribution 和许可证。

## 2. 当前仓库事实

工作目录：

```text
C:\Users\28898\Documents\ChatGPT\nano-vllm找实习
```

截至编写本提示词时，该目录只有一个尚无 commit 的空 Git 仓库，没有 nano-vLLM 源码。

开始时必须重新检查当前状态；如果状态已经变化，以当前文件和 Git 状态为准。禁止删除或覆盖用户后来加入的文件。

如果仓库仍为空：

1. 从官方仓库 `https://github.com/GeeeekExplorer/nano-vllm` 导入代码。
2. 固定一个明确的 upstream commit SHA，不要依赖漂移的 `main`。
3. 在 README 和 `docs/upstream.md` 中记录仓库 URL、commit SHA、许可证和导入日期。
4. 不要使用会破坏未知用户改动的 reset、clean 或 checkout 覆盖操作。

如果网络访问或包安装需要授权，按运行环境规定申请授权，不要用不安全方式绕过。

## 3. 最终目标场景

必须端到端支持并验证以下场景：

```text
请求 A：shared_prefix + suffix_A
    ↓
正常 prefill，生成逐层 KV
    ↓
将 shared_prefix 对应的完整 KV blocks 注册为可复用 context
    ↓
将其保存到 CPU cache
    ↓
释放或淘汰对应 GPU physical blocks

请求 B：shared_prefix + suffix_B
    ↓
以 token IDs 做 longest-prefix lookup
    ↓
CPU cache HIT
    ↓
为命中的逻辑 blocks 分配新的 GPU physical blocks
    ↓
CPU → GPU 恢复逐层 K/V
    ↓
只对未命中的 suffix 和不足一个 block 的 tail 做 prefill
    ↓
生成结果与完全重新 prefill 的基线一致
```

必须至少支持：

- 单机；
- 单张 NVIDIA GPU；
- tensor parallel size = 1；
- eager mode；
- batch size = 1 的完整验证；
- Qwen3-0.6B 或当前 nano-vLLM 官方示例支持的最小模型；
- greedy decoding；
- 仅复用完整 block；
- 进程内 CPU memory cache；
- 同步 CPU/GPU copy 作为第一版。

## 4. Definition of Done

只有以下所有项目都有直接证据时，任务才完成：

1. 固定并记录 upstream nano-vLLM commit。
2. 原始 nano-vLLM baseline 可以运行，或明确记录当前机器缺失的外部运行条件。
3. Feature flag 关闭时，行为与 upstream 一致。
4. 实现独立于 GPU physical block id 的 CPU-backed reusable KV block store。
5. 实现 token-block longest-prefix lookup。
6. full miss、full hit、partial hit 均有测试。
7. 非 block 对齐前缀只复用完整 block，tail 重新计算。
8. GPU block 被淘汰后，CPU 命中仍可恢复并继续推理。
9. 无压缩情况下，CPU round-trip 后的 K/V 数值与原值一致。
10. cached 与 uncached greedy generation 的 token IDs 一致。
11. 模型或 KV layout fingerprint 不匹配时禁止复用。
12. CPU cache 有容量上限和确定的 eviction policy，至少支持 LRU。
13. 连续至少 100 次 store/load/evict 循环没有 block ref-count 泄漏或容量越界。
14. 输出结构化 cache metrics，而不是依赖人工解析日志。
15. 有可重复运行的 TTFT benchmark，比较重新 prefill、GPU hit 和 CPU hit。
16. 保存原始 benchmark JSON/CSV，并由脚本生成结果，不手填数字。
17. README、架构文档、正确性说明和限制说明完整。
18. 没有实现、没有运行或没有验证的能力，不得在 README 中写成已完成。

如果当前环境没有可用的 NVIDIA GPU，可以完成 CPU 可测单元、接口、静态检查和文档，但**不得**把整个项目标为完成。需要明确列出仍待 GPU 验证的验收项和复现命令。

## 5. 明确的非目标

本阶段禁止扩展到以下内容，除非它们是修复 KV reuse 正确性所必需：

- Sparse Attention；
- top-k retrieval；
- DIPR；
- ANN/HNSW/graph index；
- SSD、NAS、S3、MinIO；
- KV quantization/compression；
- 自定义 CUDA kernel；
- 异步 CUDA stream pipeline；
- 分布式 cache；
- tensor parallel；
- O(1) GPU HBM Prefill；
- OpenAI server；
- 多模型通用框架；
- 与本目标无关的大规模重构。

不要为了展示功能数量而扩大 scope。

## 6. 全局工作规则

1. 一次只执行一个 milestone，通过验收后再进入下一阶段。
2. 每次修改前先读取真实源码，不得假设文件仍与某个网络版本相同。
3. 首先理解 tensor 的所有者、逻辑 block 与 physical block 的关系，再设计 copy 路径。
4. Scheduler/BlockManager 只管理逻辑状态时，不要让它直接偷偷访问不属于它的 CUDA tensor。
5. KV tensor 的复制应发生在实际拥有 KV tensors 的执行组件中。
6. 新能力必须通过 config/feature flag 开启，默认值不能无意改变 upstream 行为。
7. 先实现同步、单序列、TP=1 的正确版本，再考虑优化。
8. 不得通过保存 hidden states 或重新运行模型伪装成 KV reuse。
9. 不得只缓存 prompt 文本；cache identity 必须建立在 token IDs 和模型/KV 配置上。
10. 不得只比较最终文本；正确性测试优先比较 token IDs，必要时比较 logits/KV tensors。
11. 性能实验必须同步 GPU，并区分 lookup、H2D、prefill 和 first-token 时间。
12. 不预设一定加速，不删除“不好看”的结果。
13. 不伪造测试、显存、带宽或 latency 数字。
14. 保留用户已有改动；禁止 destructive git 命令。
15. 每个 milestone 完成后更新文档和测试，再提交小而清楚的 commit。
16. 除非遇到必须由用户决定的硬件、模型路径或权限问题，否则自行做范围内的合理决定，不反复询问用户。

## 7. 推荐的模块边界

实际文件名可以根据固定的 upstream commit 调整，但职责不能混乱。

```text
nanovllm/
├── engine/
│   ├── block_manager.py
│   ├── scheduler.py
│   ├── model_runner.py
│   ├── llm_engine.py
│   └── sequence.py
└── kvdb/
    ├── types.py
    ├── fingerprint.py
    ├── prefix_index.py
    ├── coordinator.py
    ├── metrics.py
    └── store/
        ├── base.py
        ├── gpu.py
        └── cpu.py

tests/
├── test_cache_fingerprint.py
├── test_prefix_index.py
├── test_cpu_block_store.py
├── test_cache_eviction.py
├── test_cache_roundtrip.py
└── test_prefix_reuse_integration.py

benchmarks/
├── benchmark_prefix_reuse.py
├── generate_report.py
└── results/

docs/
├── upstream.md
├── architecture.md
├── cache_identity.md
├── correctness.md
├── benchmark_methodology.md
└── limitations.md
```

不要为了满足这个目录示例创建无内容文件。只有承担明确职责时才创建。

## 8. 必要的数据结构语义

具体 Python 形式由实现决定，但至少需要等价概念：

```text
CacheFingerprint
    model/config identity
    dtype
    num_layers
    num_kv_heads
    head_dim
    block_size
    RoPE / rope_scaling identity

CacheKey
    fingerprint
    chained token-block hash

CPUBlockHandle
    stable CPU slot identity
    independent from GPU physical block id

LookupResult
    matched block count
    matched token count
    CPU/GPU handles
    hit tier

CacheMetrics
    requests
    hits/misses
    GPU-hit blocks
    CPU-hit blocks
    reused tokens
    recomputed tokens
    bytes H2D/D2H
    lookup time
    load time
    store time
    prefill time
    TTFT
    eviction count
```

建议复用 nano-vLLM 已有的 chained block hash 语义，但 CPU index 不能把 GPU physical block id 当作持久身份。

为防止 hash collision，命中后必须验证对应 token block 内容；不能只相信 hash。

## 9. Milestone 0：环境与 baseline

### 目标

建立可复现的 upstream 基线，明确当前 nano-vLLM 的真实 KV 路径。

### 步骤

1. 检查 Git 状态和全部现有文件。
2. 若仍为空，安全导入官方 nano-vLLM，并固定 commit。
3. 阅读：
   - README；
   - config；
   - `llm_engine`；
   - `scheduler`；
   - `block_manager`；
   - `sequence`；
   - `model_runner`；
   - attention layer；
   - model implementation。
4. 画出请求从 tokenization 到 prefill/decode 的调用链。
5. 记录：
   - KV tensor shape；
   - block layout；
   - 每层 K/V 存储位置；
   - block table 的创建和释放；
   - prefix hash 生成方式；
   - cached token 数如何传给 model runner；
   - attention 如何按 block table 读 KV。
6. 检查运行环境：Python、PyTorch、CUDA、GPU、模型路径。
7. 跑通官方最小示例和原始 benchmark。
8. 新增一个确定性 baseline 脚本，固定 token IDs、greedy decoding 和输出长度。
9. 将结果保存为机器可读文件。

### 验收

- `docs/upstream.md` 记录 upstream commit。
- `docs/architecture.md` 准确描述真实调用链和 KV layout。
- 能给出 baseline 命令和输出文件。
- 如果 GPU 环境不可用，有明确的环境证据和后续复现命令，不能假装已经运行。

未通过前不得进入下一 milestone。

## 10. Milestone 1：Instrumentation

### 目标

不改变推理语义，为后续复用建立可靠观测。

### 步骤

1. 新增结构化 `CacheMetrics`。
2. 在 logical prefix lookup 处记录命中 block/token。
3. 在 model execution 处区分：
   - lookup；
   - prefill；
   - first token；
   - decode。
4. GPU 计时使用 CUDA Event 或明确同步，不能用未同步的 wall-clock 数字冒充 GPU latency。
5. 提供从公开 LLM/engine API 获取最后一次或累计 cache stats 的方式。
6. feature flag 关闭时尽量没有额外开销。

### 测试

- cold miss；
- 相同 prompt 的 GPU hit；
- 一个完整 block 的 partial prefix hit；
- 完全不同 prompt 的 miss；
- 非 block 对齐共享前缀。

### 验收

- 指标能准确解释每个测试实际复用了多少 token。
- greedy token IDs 与 instrumentation 前一致。
- 没有用日志字符串作为唯一数据接口。

## 11. Milestone 2：抽象 reusable block store

### 目标

先解耦逻辑 block 管理和物理 KV 存储，不改变现有 GPU 行为。

### 步骤

1. 定义 cache fingerprint、key、handle、payload、lookup result、store stats。
2. 用 `GPUBlockStore` 或等价适配器包装现有 GPU block 行为。
3. `BlockManager` 继续负责引用计数、逻辑分配和 token block hash。
4. store 负责物理 KV 数据的位置和读取。
5. 明确 physical GPU block 被重新分配时，哪些 CPU metadata 仍有效。
6. 所有新路径放在 feature flag 后。

### 验收

- flag 关闭：upstream baseline 通过。
- flag 开启但 CPU tier 关闭：行为与原 GPU prefix caching 一致。
- 不存在 circular import 或跨层直接访问内部 tensor 的临时 hack。

## 12. Milestone 3：CPUBlockStore 独立实现

### 目标

在尚未接入 scheduler 前，正确保存和恢复一个完整 physical KV block。

### 步骤

1. 根据真实 KV layout 定义 `KVBlockPayload`。
2. 支持 pageable CPU memory。
3. 支持可配置的 pinned CPU memory。
4. 实现同步 D2H store 和 H2D load。
5. CPU slot 必须独立于 GPU block id。
6. 加入容量计算和 LRU。
7. 当覆盖/淘汰 slot 时，正确删除 prefix index 中的失效引用。
8. 记录 D2H/H2D bytes、时间和带宽。

### 测试

- 不同 layer、K/V、dtype、shape 的 round-trip。
- 多 block 存取顺序。
- LRU 顺序。
- 超容量行为。
- 删除和重复插入。
- fingerprint mismatch。
- 100 次 store/load/evict 循环。

### 验收

- 无压缩 round-trip 数值一致。
- CPU cache 永不超过配置容量。
- slot 回收不会让旧 handle 静默指向新数据；使用 generation/version 或等价机制防止 stale handle。

## 13. Milestone 4：PrefixIndex 与 ContextDB

### 目标

建立与物理 GPU block 无关的、基于 token IDs 的 longest-prefix index。

### 步骤

1. 使用完整 token block 的 chained hash。
2. hash key 必须包含 cache fingerprint。
3. 命中后验证 token IDs，防止 collision。
4. lookup 返回从第一个 block 开始连续命中的最长前缀；中间断裂后禁止跳跃复用。
5. 非完整 tail 不写入共享 index。
6. 设计最小 API，例如：

```python
session, uncached_token_ids = db.create_session(token_ids)
db.store(session)
stats = db.stats()
```

实际 API 可调整，但语义必须清楚：`create_session` 查找最长前缀，`store` 显式物化当前完整 blocks。

### 测试

- full miss；
- full block hit；
- 多 block longest-prefix hit；
- partial hit；
- 中间 block 不一致；
- 相同 hash、不同 token 内容的防碰撞模拟；
- 不同 fingerprint；
- eviction 后 index 清理。

### 验收

- 单元测试不需要启动完整模型也能验证 prefix 语义。
- CPU handle 生命周期与 prefix index 一致。

## 14. Milestone 5：端到端 CPU-backed prefix reuse

### 目标

接通最终目标场景。

### 实现原则

1. 先只支持 batch size = 1、TP=1、eager mode。
2. lookup 可以在 scheduler/engine 侧完成，但 H2D copy 必须由实际拥有 KV cache tensor 的组件执行。
3. CPU hit 后先为每个命中 logical block 分配新的 GPU physical block。
4. 在该请求开始使用 block table 前，完成所有层 K/V 的 H2D 恢复。
5. 恢复成功后，更新 sequence 的 cached-token metadata。
6. 只调度未复用的 suffix tokens。
7. 如果 load 失败，必须安全回退为重新 prefill，不能使用半加载 KV。
8. 不足一个 block 的共享 tail 在第一版重新计算。
9. GPU block 淘汰不能删除仍存在于 CPU tier 的 context identity。
10. CPU eviction 后必须变成真正 miss。

### 必须运行的集成用例

```text
Case A: cold request
Case B: exact repeated request, GPU hit
Case C: shared prefix + different suffix, GPU partial hit
Case D: store to CPU, force GPU eviction, exact CPU hit
Case E: store to CPU, force GPU eviction, CPU partial hit
Case F: non-block-aligned shared prefix
Case G: one token differs inside the first block -> miss
Case H: one token differs after N shared blocks -> only N blocks hit
Case I: CPU LRU eviction -> recompute fallback
Case J: disabled feature -> upstream behavior
```

### 正确性比较

每个可运行 case 至少比较：

- generated token IDs；
- matched token count；
- recomputed token count；
- KV round-trip；
- 如果接口可得，首个新 token logits 的 `allclose`。

### 验收

- GPU physical blocks 被释放后仍能从 CPU 恢复。
- 请求 B 实际只执行 suffix prefill，不允许只是 metrics 声称复用。
- greedy token IDs 与 cold baseline 一致。
- 所有失败路径安全回退。

## 15. Milestone 6：Benchmark 与分析

### 目标

回答“KV reuse 在什么条件下值得”。

### 对比组

```text
A. cold recompute：完全重新 prefill
B. GPU hit：现有 GPU prefix cache 命中
C. CPU hit pageable：CPU 完整 KV 恢复
D. CPU hit pinned：pinned CPU 完整 KV 恢复
E. partial CPU hit：恢复共享 blocks + 重算 suffix
```

### 输入矩阵

根据模型 context limit 和显存调整，但尽量包含：

```text
prefix tokens: 256, 512, 1024, 2048, 4096, 8192
suffix tokens: 16, 64, 256
reuse ratio: 25%, 50%, 75%, 100%
generated tokens: 固定 1（TTFT）以及固定 32（完整请求）
```

硬件不支持的点应标记为 not run，不得填估算值。

### 计时规范

1. 固定模型、dtype、block size、seed 和 token IDs。
2. greedy decoding。
3. 至少 5 次 warmup。
4. 至少 20 次正式测量，资源不足时说明缩减原因。
5. 报告 p50 和 p95。
6. GPU 阶段正确同步。
7. 分解：lookup、D2H/store、H2D/load、suffix prefill、first decode、TTFT。
8. 同时记录 peak allocated/reserved GPU memory、CPU cache bytes、H2D/D2H bytes。
9. 原始数据写入 JSON 或 CSV。
10. 图表必须由脚本读取原始结果生成。

### 必须回答的问题

1. prefix 多长时 CPU reload 开始比重新 prefill 快？
2. pageable 与 pinned memory 差多少？
3. lookup 开销占 TTFT 多少？
4. partial reuse 的收益如何随 reuse ratio 变化？
5. CPU cache 容量增加后能保存多少 context？
6. 哪些情况下 CPU reuse 更慢？为什么？

### 验收

- README 中的数字都能追溯到原始结果文件。
- 不要求得到正向 speedup，但必须解释测量结果。

## 16. Milestone 7：项目交付质量

### README 必须包含

1. 项目定位。
2. 与 upstream nano-vLLM 的区别。
3. 明确的 attribution。
4. 架构图。
5. Quick Start。
6. Feature Matrix。
7. Correctness Methodology。
8. Benchmark Methodology。
9. 结果图和硬件环境。
10. Design Decisions。
11. Limitations。
12. Roadmap，但不得将 Roadmap 写成已实现。

### 文档必须解释

- prefill 与 decode 的区别；
- KV Cache shape 和字节数计算；
- block-level prefix caching；
- 为什么只复用完整 blocks；
- 为什么 cache key 使用 token IDs；
- 为什么 cache fingerprint 必须包括模型和 RoPE/KV layout；
- 为什么 CPU reload 不一定总比 recompute 快；
- logical block、GPU physical block、CPU slot 的区别；
- GPU eviction 后 CPU context 为什么仍然有效；
- 当前实现与 AlayaDB/生产系统的差距。

### 最终清理

- 运行 formatter/linter（如果项目配置存在）。
- 运行全部 CPU tests。
- 运行所有当前硬件允许的 GPU integration tests。
- 检查 git diff，确保没有模型权重、密钥、巨大结果或临时文件。
- 将验证命令和真实输出摘要写入最终报告。

## 17. 每个 milestone 的固定汇报格式

完成一个 milestone 后，按以下结构汇报，然后继续下一个：

```text
Milestone N：<名称>

完成内容：
- ...

修改文件：
- path: 用途

验证：
- command
- result

关键证据：
- ...

已知限制：
- ...

下一阶段入口条件：
- ...
```

不要只说“已实现”或“测试通过”；给出实际命令、用例数量和关键结果。

## 18. 发生问题时的决策规则

### nano-vLLM 版本结构与计划不同

以固定 commit 的实际代码为准。先更新架构文档和修改计划，再做最小侵入实现。不要为了符合本提示词的文件名而复制一套平行引擎。

### 当前 nano-vLLM 已经支持某项功能

先用测试证明已有行为，然后将工作集中到它没有覆盖的部分。不得把 upstream 能力写成自己的贡献。

### Windows/依赖导致 nano-vLLM 不能运行

收集准确错误和环境信息；检查项目官方支持条件。完成可在当前环境验证的纯 Python 单元和静态部分，但不虚构 GPU 结果。如果端到端验证必须换到 Linux/CUDA 机器，给出最小复现清单并明确请求用户提供环境。

### CPU reload 没有加速

不要删掉结果。检查计时同步、pinned memory、copy 粒度和 prefix 长度；确认正确后把 break-even 和瓶颈作为项目结论。

### 正确性不一致

立即停止性能优化。优先检查：

- position IDs；
- RoPE 后的 K 是否被正确保存；
- layer/head/block tensor layout；
- cached-token count；
- block table 顺序；
- tail token 边界；
- dtype/device；
- GPU copy 是否在消费前完成；
- stale CPU handle；
- hash collision verification。

正确性未通过时不得继续 benchmark。

## 19. 最终回答要求

最终只在 Definition of Done 全部通过后宣布完成。最终回答必须包含：

1. 项目实现结果。
2. 架构摘要。
3. 最重要的正确性证据。
4. benchmark 的真实结论和 break-even。
5. 测试命令及结果。
6. 未实现内容和限制。
7. README、架构文档、benchmark 结果的文件链接。

如果仍缺 GPU、模型或其他外部条件，必须明确说明哪些条目尚未验证，不能将任务描述为完成。

