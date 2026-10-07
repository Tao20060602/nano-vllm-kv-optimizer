# 推理引擎与 GPU 算子的两个独立项目

日期：2026-10-06。状态：用户已确认 MoE 优先路线，独立算子仓库已建立并完成 M0/M1 首轮验证。

2026-10-07 更新：独立算子项目已完成 M20 pinned Qwen3-30B-A3B 真实 layer-0
输入/权重、冻结完整MoE接口的成熟对照，attention算子与系统接入边界也已有报告。
公开仓库为 [Tao20060602/llm-gpu-kernels](https://github.com/Tao20060602/llm-gpu-kernels)。
NanoKV 固定稀疏策略的 KV 保存路径优化完成独立三组配对，warm TTFT 降低3.99%，
见 [M23 系统报告](NATIVE_TTFT_DIRECT_STORE_REPORT.md)。本文其余内容保留10月6日
规划背景，具体当前结果以两个仓库的最新报告为准。

用户希望面向不同公司的岗位，分别展示推理引擎优化和算子优化。
两个项目各自提供问题定义、代码入口、正确性证据、性能对照和复跑说明。
用户进一步明确：算子项目需要练到 Tensor Core，并展示 attention/MoE 等
生产相关算子的理解。因此当前建议调整为：首个核心算子优先评估
MoE routed grouped GEMM 与局部融合，attention prefill 作为后续模块。
用户已确认按此顺序开始实施；旧 segmented-prefill 方案保留。

独立仓库：`/home/tmz/文档/ChatGPT/llm-gpu-kernels`，
分支 `codex/moe-bootstrap`，独立 `.venv`。已实现 BF16 dense GEMM、
host-known counts 的 ragged grouped GEMM，以及完整 MoE 的 FP32 语义参考。
边界数值与 5 个性能形状已运行，保存源码/输入 hash 和 PTX/cubin/SASS。
SASS 确认 HMMA；目前仅是 bootstrap，成熟 MoE backend 对照、GPU dispatch、
局部融合、完整 expert 层、attention 和 profiler utilization 尚未完成。
记录与复跑入口以独立仓库的 `README.md` 和 `docs/BOOTSTRAP_REPORT.md` 为准。

同日 M2 更新：独立项目已接入 CUTLASS3.8 classic grouped GEMM 的固定
BF16/SM86 配置，十形状四轮 A/B 与 FP32 oracle、固定 CUDA Graph 重放通过。
这是预分组 GEMM 基线，尚未调优或完成完整 MoE。实际矩阵、数值、工具链
兼容补丁和下一阶段判断见独立仓库 `docs/M2_BASELINE_REPORT.md`。
CUDA12.8/GCC14 构建工具放在用户目录，未替换 NanoKV runtime 或系统 compiler。

## 1. 项目一：NanoKV 推理引擎优化

沿用现有仓库 `Tao20060602/nano-vllm-kv-optimizer`。
定位为基于 nano-vLLM 的单序列长上下文推理研究原型，明确上游来源和个人增量。

主要展示内容：

- CPU/GPU 分层 KV、prefix reuse、块身份与生命周期。
- 稀疏检索、pinned staging、gather、搬运和状态更新。
- recent 覆盖、跨请求 reset、恢复失败回退等正确性工程。
- 通过 timeline 和 matched A/B 区分计算、搬运、同步与 host 提交成本。
- TTFT、prefill wall、decode latency、显存/主存及质量限制的分别报告。
- M0–M22 的成功、撤回方案与原始证据。保留 M22 质量损失的公开说明。

现有 M22 是已完成成果。原生 Linux 的依赖、GPU 和简短模型推理现已验收，
但旧 WSL 性能数字不作为新机器状态的测量结果；ignored traces 尚未迁移。
未来算子接入属于可选增量，NanoKV 的既有成果可以独立展示。

## 2. 项目二：独立 LLM GPU 算子优化

建议使用单独仓库、独立 README、Python 包、依赖和实验记录。
暂用描述名“LLM GPU Kernel Optimization on Ampere”；最终名称随选题确定。
一个独立项目分阶段覆盖 MoE 与 attention，不把每个学习练习计为新项目。

主要展示内容：

- 数学定义、GQA head 映射、causal mask、dtype/stride/shape 契约。
- Triton 实现、online softmax、分块与并行策略。
- 寄存器、共享内存、占用、访存和 Tensor Core 指令的实证分析。
- 同输入数值误差、边界形状、成熟 backend 对照。
- 编译/冷启动与稳态性能分开，公开形状矩阵、失败形状和原始记录。

核心包只接收张量、长度、stride、固定路由结果和计算参数。
独立 demo 与算子测量使用可复现输入，不要求 NanoKV、模型权重、CPU offload
或检索状态。真实模型输入可作为补充场景；NanoKV 集成由后续薄适配层完成。

### 当前推荐：MoE routed grouped GEMM

选题标准是 Tensor Core 训练价值、真实负载特征、可独立验证性、当前硬件
可执行性和有深度的优化空间；NanoKV 集成不是必需条件。

建议首版选择单 GPU、BF16、forward、固定 top-k expert IDs 和权重的
routed expert MLP。输入既可来自可审计的合成路由，也可来自真实 router
输出；首版先保持路由公式和结果固定，以免混入模型质量变化。

完整计算边界：

```text
token / expert IDs / routing weights
  -> dispatch / expert grouping
  -> grouped gate+up GEMM
  -> SwiGLU
  -> grouped down GEMM
  -> routing-weight combine / restore token order
```

先实现和优化 grouped GEMM 核心，再逐项评估局部融合。优化候选包括
tile、warp/stage、权重访问顺序、L2 局部性、expert/tile 调度，以及减少
padding 和中间张量。Persistent、split-K 或融合只是候选，不预先认定更快。
gate/up 与激活融合可能提高寄存器压力，需同时记录占用和完整链路成本。

实际负载必须覆盖：小 token batch、大 token batch、均匀/偏斜路由、空 expert、
非整 tile、不同 top-k。小 batch 可能主要受权重读取和提交开销限制，
不能用大 batch 的 Tensor Core 利用率概括所有 decode 负载。

可采用 Qwen3-30B-A3B 的单层算子形状作为一个现实来源：官方 config 中
hidden size=2048、MoE intermediate size=768、128 experts、top-k=8。
单层算子实验可使用生成的权重和输入，不要求下载或部署整个 30B 模型。
该场景表示真实模型形状，不表示真实路由分布或完整模型质量已验证。
正式实验前固定 config revision/hash、输入和 routing provenance，并先核算
全部权重、workspace、参考输出的内存预算。

对照包括数值 oracle、合理的 PyTorch/cuBLAS 实现，以及至少一种经确认
支持 SM86 的成熟 grouped GEMM/MoE backend，优先评估 CUTLASS 或 vLLM。
先确认其硬件/版本/数值语义，再固定来源；不能将只击败 Python expert 循环
的结果称为超越成熟 MoE。分别报告预先分组的 GEMM 和含 dispatch/combine
的完整 expert 层，独立列出冷启动与稳态结果。

生产相关工程证据包括：shape/dtype/stride 契约、路由和还原正确性、
workspace 生命周期、同 stream 的依赖、固定形状 CUDA Graph 可行性、
失败/回退行为及公平计时。这些应按阶段验证，不在规划中声称已经支持。
单 GPU 实验不证明 expert-parallel all-to-all、多卡 serving 或生产部署经历。

Tensor Core 证据应保存 PTX/SASS 中的 MMA/HMMA 与实际 profiler 指标，
结合 FLOPs、访存与误差分析；代码中调用 `tl.dot` 不是完整硬件证据。
RTX 3080 Laptop 是 SM86，首版用 BF16/FP16 与 FP32 累加。Hopper 的
TMA/WGMMA 和原生 FP8 路线不属于该硬件可直接验证的能力。

NanoKV 当前 Qwen3-4B 路径是 dense 模型，现有 engine 没有 MoE 模型接入。
MoE 算子作为独立项目完成；其应用验证不自动扩大为 NanoKV MoE 引擎开发。

### 后续 attention 候选：通用 GQA prefill / 分段输入

若实施 attention 模块，第一版收窄为 Ampere SM86、单请求、BF16、
D=128、Hq=32/Hkv=8、forward。
接口描述 GPU 上分开存储的历史段与当前 causal 段，避免把核心 API 绑定为
NanoKV 的 `sink`、`recent` 或 representative 数据结构。

NanoKV 四段布局是一个实际应用场景；连续 K/V 和其他长度组合也进入独立
对照矩阵。首版只支持明确列出的段数与布局，不先承诺通用 ragged serving。

对照至少包含：

1. 小规模 FP32 数值参考。
2. 连续输入的成熟 FA2 backend。
3. 分段输入经合理 buffer 复用/拼接后的 FA2 链路。
4. 新分段算子；可行时补分段 FA2 加正确 LSE 合并的对照。

研究价值来自布局适配、计算组织和可复查优化证据；GQA、online softmax、
split-KV 或参考教程本身有已有工作，引用来源并说明新增部分。
若新算子只在部分形状获益，准确描述适用范围；若没有性能优势，保留结果并
重新判断选题，不把教程复现或击败弱基线当成优化成果。

### 备选：GQA decode

若优先完成更小的算子闭环，可选择 Q 长度为 1 的 decode attention，
研究沿 KV 分区的并行、LSE 合并与 GQA 组内复用。
独立测量需加入成熟 decode backend，而不只对照 NanoKV 当前 FP32 PyTorch 路径。
短 KV 下并行归约开销，以及长 KV 下的带宽/占用，需要分别验证。

### 另一候选：代表评分与 Top-K 融合

此方向主要展示归约与检索算子能力。保持现有评分公式时，需要核对分数、
mask、selected IDs、并列分数处理和数据转换开销。
其与 NanoKV 的耦合比纯 attention 更强，故作为独立 attention 项目的优先级较低。

## 3. 两个项目的连接方式与成果归属

```text
独立 LLM 算子包：MoE / attention 的数值、计算链路和硬件证据
                    ↓ attention 可选适配
NanoKV：模型、检索、CPU/GPU KV、同步、完整请求测量
```

- 算子仓库保存核心实现、独立 reference、形状矩阵、调优和硬件分析。
- NanoKV 保存适配层、状态/selection 验证和端到端结果。
- 核心代码通过依赖与版本引用复用，避免复制后独立修改两份实现。
- 相同 kernel 加速结果只归属一份测量；引擎项目另测真实请求收益。
- kernel 更快不自动等于 TTFT/TPOT 更快。算子项目的独立完成不依赖
  NanoKV 集成取得性能提升。

## 4. 建议准备顺序

1. 保留 NanoKV M22 的证据边界，整理系统项目的贡献、架构和复跑入口。
2. 确定首个 MoE 或 attention 算子及其输入/数值契约，建立独立仓库。
3. 用 BF16 GEMM 做 Tensor Core 校准与学习，不把教程复现当成项目主成果。
4. 若接受当前建议，先完成 MoE grouped GEMM 的强对照和一个经 profile
   支持的优化，再评估局部融合与完整 expert 层的成本。
5. 首个算子报告完整后再决定 attention 模块。Attention 的 NanoKV 集成
   根据独立 operator-chain 证据另行决定。

首轮基础实现与测量已完成，未完成的优化和强对照不写成简历成果。

### Native Linux 进度补充（2026-10-06）

独立算子仓库：`/home/tmz/文档/ChatGPT/llm-gpu-kernels`，
branch `codex/moe-bootstrap`。M3 已保存为 commit `4eb6c41`。
M2 已完成固定 CUTLASS3.8 grouped GEMM 对照；M3 扫40组参数，训练复测后
冻结density规则，再用14个随机/热点/回退场景验证。随机T1024/T2049相比
原始Triton约少32–33% Graph耗时；小形状 prepared入口降低API开销，Graph
基本相同。部分场景仍落后固定CUTLASS，报告中保留了失败/回退和噪声。
详见该仓库 `docs/M3_OPTIMIZATION_REPORT.md` 及原始JSON。
这些结果仅为预分组gate/up GEMM，不属于NanoKV端到端结果。
M4已保存为 `05a3ab9`：全GPU运行时IDs分组、gate/up、SwiGLU、down、combine
forward及同地址动态路由值Graph重放通过。保持BF16舍入边界的gate/up+SwiGLU
局部融合，在本项目T1024 Qwen-MoE形状随机/热点完整forward上分别少约17%/14%
时间，8个A/B场景融合前后逐bit一致。详见独立仓库 `docs/M4_MOE_REPORT.md`。
这个M4收益只对比本项目unfused/fused；成熟完整MoE backend对照已在M5补上。
M5已保存为 `672f3e7`：隔离接入未修改的官方vLLM0.10.0完整fixed-route
MoE，9个功能/15个性能场景全部通过既定FP32阈值；记录舍入差异和默认config。
缓存8个编译launch的小形状API p50少约69–70%，Graph基本不变；Qwen形状
多数Graph场景仍由vLLM领先，例如T1024 random seed73为4.180ms vs本项目5.105ms。
来源、全部样本和复跑入口在独立仓库 `docs/M5_MATURE_MOE_REPORT.md`。
这些数据不代表真实MoE模型、完整serving或NanoKV请求收益。
attention、真实MoE模型和NanoKV系统收益仍是后续工作。

参考的项目职责分工：
[vLLM](https://github.com/vllm-project/vllm) 展示推理/serving 引擎，
[FlashInfer](https://github.com/flashinfer-ai/flashinfer) 展示独立推理算子库。
它们用来说明模块边界，不是当前项目已达到其覆盖范围或性能的声明。

选题参考：

- [NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)。
- [CUTLASS grouped scheduler](https://docs.nvidia.com/cutlass/4.3.5/media/docs/cpp/grouped_scheduler.html)。
- [PyTorch MoE locality analysis](https://pytorch.org/blog/accelerating-moe-model/)。
- [vLLM fused MoE source](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py)。
- [Qwen3-30B-A3B config](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)。

以上来源说明已有技术和实际输入契约；它们的硬件与加速数字不作为本机结果。

### M6–M8 进度（2026-10-06）

独立仓库M6 `f41b924`将histogram得到的rank用于dispatch，完整路径从8次
启动降至7次。小形状Graph约减少6–7%，Qwen大形状不足1%，保留逐bit与FP32证据。
M7 `ba58619`在seed409的6个完整MoE训练场景扫描12组tile并长复测，冻结
只依赖shape的规则后与原ranked/vLLM同轮比较15场景。T1024 random两个seed
相对原ranked少约25%，相对固定官方vLLM默认配置少约6.7–8.1%；T193/hot仍落后，
T2049对vLLM仅少约1.6–2.1%。不能将这些数据描述成普遍胜过成熟后端。
独立仓库docs/M7_FULL_MOE_OPTIMIZATION_REPORT.md保留选择边界、全部p50/p95、
FP32误差、编译产物及资源信息；不是真实MoE模型或NanoKV请求性能。

M8 `71d4267`完成四段GPU KV的复用缓冲区pack+未修改官方FA2，与prepacked FA2
的12场景成本基线。T64 Graph直接差异约29–36%，T256约14–16%，T4096约2%。
这是两条执行路径的成本对照，不是自定义kernel收益；pack-only不相减。
下一步直接读取四段KV的GQA prototype正在独立仓库实现。NanoKV运行入口未改，
任何算子到引擎的集成需另测状态、selection、同步和完整请求。

M9 `a449a73`已实现四段KV直读与G1/2/4对照，36功能及12性能场景全FP32验证。
G1在T64对强pack+FA2约少5–29% Graph耗时，大T回归完整保留。
M10 `6001926`统一逻辑KV单循环，12候选/6训练输入冻结选择；大T对M9少11–19%，
但T256回归约4–10%，且仍慢于强FA2。因此策略仅实验用途，不作为引擎默认。
M11短query split-KV与稳定FP32 partial merge正在独立测量，scratch和merge均计入。
算子核心与引擎请求收益分别归属，未将这些局部结果写作NanoKV系统成果。

### M11–M15 进度（2026-10-06）

独立算子仓库M11 `281de8f`测量完整split/merge，M12 `3b929a3`验证窄域split，
M13 `f6a6a6c`验证固定query64的新历史长度，均未通过预设性能门槛，冻结FA2回退。
M14 `369f5ad`调整query分块/warps/stages，训练选中的中间query规则在T127/128
长历史回归约16%，保留实验策略，不推荐默认启用。M10/M14的冻结实验策略包含
自定义kernel，不能描述成所有冻结策略都是FA2；当前推荐attention仍为FA2。

M15 `4b3053a`在currentT>0的非空可见key契约下简化softmax状态和mask处理；
保留Q读/O写mask，去掉内部填充行mask归约和冗余状态选择。两个固定stage配置
13项同配置A/B均减少Graph耗时，S2几何平均少约7.55%；短query六项比完整
pack+未修改FA2少7.1–34.5%，大T仍落后。48功能场景/8次Graph值更新和stream
检查、376 finite/误差metrics、26项源码hash运行前后通过。不是冻结适用域策略。
来源与全部回归/raw样本在独立仓库docs/M14_ATTENTION_RETILE_REPORT.md、
docs/M15_NONEMPTY_ATTENTION_REPORT.md和docs/PROJECT_STATUS.md。

M16有限query/key/head_group资源搜索已完成，结果见下一节。仍不将算子数字写成
NanoKV模型/请求收益，NanoKV引擎代码未改。

### M16 冻结短query路径（2026-10-06）

独立仓库commit `6594892`。八种token16/32、G1/2、key32/64资源分块在六项新
训练输入完成48候选对照，原0.008 FP32/finite界全部通过。训练及长复测后冻结：
T32–128、sink64/recent512/currentT、总history≤4096用M15默认token32/key64/G1/
warps4/stages2；其余FA2。大query没有达到成熟FA2收益门槛，保留回退。

44项原holdout通过后，另声明25项历史上限追加审计；原31项冻结source provenance
及两项audit source hash通过。36项实际custom路径的Graph p50几何平均比完整
pack+FA2少22.09%；六项T127/128长历史有0.4%–2.2%回归，最坏2.17%未超过原5%
上限，未缩域或事后改配置。越过history4096及其他metadata域外正确选择FA2。
69性能输入数值全通过，1563项训练/功能/性能finite及误差metrics通过；有限矩阵
不是连续域或真实请求分布保证。报告与使用契约：独立仓库
`docs/M16_ATTENTION_RESOURCE_REPORT.md`、`docs/ATTENTION_USAGE.md`。

M16是可复用独立算子prototype，不是NanoKV系统集成。下一步系统适配需要保持
selection、四段KV状态、跨stream依赖及Qwen3-4B pinned模型完整请求对照，不直接
将算子收益写成TTFT/TPOT收益。旧引擎入口未改，旧WSL环境与模型未删除。

### 真实模型边界验证与短尾修复（2026-10-06）

NanoKV commit `dc8cff5`修复short prefill recent窗口的重叠copy与旧窗口长度索引。
原历史4096/current96的失败保留，30项recent/sink/valid_len精确检查通过。
随后固定Qwen3-4B snapshot、archive/code两条16480-token请求、4096主chunk加96尾
chunk采集完成。每请求144个later-layer调用，reset/selection与12项采集post窗口
状态通过。13个模型文件重新按源manifest的SHA256/Git blob hash核对通过。
捕获driver、完整prompt/selection manifest及失败/修复审计已提交；大tensor保留
在本机ignored bench_logs。采集wall含同步与磁盘I/O，不是系统benchmark。

独立算子M17 commit `e90e2c0`恢复实际current V token stride6144，四arm对照旧
GPU pack/FA2、强复用raw pack/FA2、含V转换的冻结边界、contiguous core诊断。
12项真实tensor重放、204 finite/原0.008误差metrics、34项source及M16原provenance
通过。6项96尾chunk计入每次192KiB V转换后，Graph p50 geomean比强基线少37.26%；
4096主chunk仍选择FA2，不称为自定义加速。报告为独立仓库
`docs/M17_REAL_INPUT_BRIDGE_REPORT.md`。

第一次96形状prepare约1.03s，后续同形状绑定约0.37–0.44ms。当前绑定地址接口
不宜直接放进动态模型请求；下一阶段需形状编译缓存、安全动态地址重新绑定与
强复用FA2/H2D基线，然后至少三组fresh-process系统A/B及greedy/selection检查。
尚未切换NanoKV默认attention，也没有新请求加速或质量无损声明。

### 编译缓存与动态地址绑定（2026-10-07）

独立算子M18 commit `fa4da4d`新增指针无关编译模板local LRU和stream独立workspace。
只编译入口不执行warm attention；同一shape/stride/device允许每次传入新地址，登记
record_stream并计入必要V转换。Graph仍固定捕获地址，跨stream输入依赖由caller建立。
M17全部34份源码保持原hash，M16冻结策略未改。主12项真实重放及功能/补充布局审计
共227项finite/原.008精度metrics通过，max RMS .00226431、normalized max .00397238。

六项96尾chunk计入动态检查/绑定/生命周期/V转换，API相对强复用pack+FA2 geomean
少33.17%，固定Graph少37.11%；动态API比原固定地址入口多7.72%成本。T4096仍FA2。
首次只编译约321.76ms，已有disk/JIT缓存，不能视为全冷启动或与M17公平prepare A/B；
cache-hit workspace准备0.151–0.337ms仍在热forward之外。local模板上限不限制Triton
内部缓存或活跃workspace。通用fallback预留未使用Q形状output（T4096额外32MiB）已
公开；引擎adapter应只复用direct workspace，fallback另用强FA2路径。

报告及契约：独立仓库 `docs/M18_DYNAMIC_BINDING_REPORT.md`、
`docs/M18_DYNAMIC_BINDING_USAGE.md`。NanoKV默认入口仍未改；下一步为显式实验接入，
核对selection/窗口/层输出，再做至少三组fresh-process完整请求A/B。不将33.17%算子
数字当TTFT或请求收益。没有新的用户决策点。

### Native 实验接入与默认验收（2026-10-07）

NanoKV commit `542545e`实现显式flash_reuse/operator实验入口，全36层同stream共享
packed/selected scratch与一个direct workspace，约35.19MiB额外attention空间。
独立算子M18源码/冻结选择保持原字节，编译缓存复用新地址；原默认flash分支及selector/
store/decode等结构保持。GPU发生GFW/Xid119/154后，用户完全关机开机恢复；旧故障
diagnostics与启动失败保存。当前kernel7.0.0-38，项目venv/driver保持既有版本。

两后端audit共1080项finite/.008数值与864项later-layer窗口状态通过。三组新进程
三arm对照完成（9模型进程、18请求）：operator相对reuse FA2完整prefill geomean
少0.08%，相对原flash慢0.60%，没有稳定系统收益。reuse与原flash selected IDs/逻辑
状态及首token一致；operator尾chunk后续层集合变化，archive11层/code13层，每组三次
重复，首token仍相同。默认推广不获支持，保留实验入口，不宣称无损或请求加速。

完整报告与13份原始JSON：`docs/NATIVE_SEGMENTED_ADAPTER_REPORT.md`、
`benchmarks/results/operator_bridge/m19-native/`。独立算子commit `6114da4`保存系统边界
索引与两项目简历/面试证据 `docs/PORTFOLIO_EVIDENCE.md`；NanoKV另有
`docs/PORTFOLIO_ENGINE_EVIDENCE.md`。当前无需用户决策，完整接入/证据闭环已完成。
