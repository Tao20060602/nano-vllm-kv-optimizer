# Native NanoKV 分段算子接入与完整 prefill 对照

2026-10-07 Asia/Shanghai。状态：**实验接入、逐层审计和三组新进程对照完成；默认提升未获支持。**
默认 `sparse_prefill_attention_backend="flash"` 保持原路径。

## 工程结论

新增显式 `flash_reuse` / `operator` 后端，独立算子仓库与NanoKV源码继续分开。
全36层共享同stream的临时packed/selected scratch和一个direct workspace，不为每层
保存大chunk拼接空间。M18模板LRU独立于张量地址，重复请求复用，不再反复warm attention。

本轮相对复用FA2，完整prefill耗时比几何平均 **0.999172（少0.08%）**，配对范围
0.992644–1.004611；相对原FA2比 **1.005987（慢0.60%）**，六项配对全部略慢。
样本只有两条固定文本、三次新进程，不能据此断言普遍显著差异。

算子数值通过原误差界，首个生成token相同；后续尾chunk层的检索ID集合有确定性差异。
因此本轮没有证明完整模型选择完全一致，也没有证明新质量收益；保留实验入口并继续
默认FA2。M18真实张量重放的33.17%算子收益不能写成完整请求加速。

## 固定条件与实现

- 模型Qwen3-4B snapshot `1cfa9a7208912126459214e8b04321603b3df60c`；suite启动前13个
  文件逐项匹配模型源manifest的SHA256/Git blob hash。
- M17原archive/code完整prompt IDs，每项16480token，4096×4+96；BF16、TP1/eager、
  summary1、Top32、sink64/recent512、index_select、YaRN4、CPU线程8。
- `max_tokens=1` / greedy；最后prefill step生成首token，没有单独decode步骤，不报告TPOT。
- 本机项目venv Python3.12.14、torch2.7.1+cu128、Triton3.3.1、官方FA2 2.8.3.post1；
  RTX3080 Laptop16GiB/SM86，kernel7.0.0-38-generic，driver595.91.07。
- 独立算子M18 commit `fa4da4d`、冻结M16策略；operator加载时验证37份源文件和原策略
  31份source/training provenance。显式提供独立仓库路径，不进行环境安装。

`flash` 使用既有 `.to(device)` 临时selected GPU复制再拼接。
`flash_reuse` 直接从同一pinned staging复制到复用packed切片，保留阻塞H2D语义。
`operator`在冻结short-query域复制selected到共享GPU scratch后，直读四段，必要current V
转换纳入run；其他形状走同一reuse FA2路径。current KV依旧在attention后store。

调用限同device/stream顺序消费；另一个stream拒绝共享。direct结果在下一层可被覆盖，
上一层projection消费先在同stream排队。reset清理KV逻辑长度，保留临时容量与模板缓存。
本负载全部请求中只构建1次direct workspace，模板1次miss/1次hit，其后run重绑地址。
共用scratch **34.25MiB** + direct输出/转换 **0.9375MiB**，合计 **35.1875MiB**。
这是新增attention scratch，不含既有每层decode buffers、权重、history/reps或全部GPU内存。
测量进程GPU peak约flash8.458GiB、reuse8.484GiB、operator8.494GiB，含所有模型分配。

## 逐层审计

两种实验后端分别跑一次warm archive和两项正式audit请求，共864项later-layer状态检查。
每项检查selected不与旧protected相交、无重复IDs、store后recent等于自身历史窗口+current
的尾部、sink不变、valid/prefill长度正确。它们验证各arm自身状态契约，不表示跨arm KV
浮点值完全相同。

每层later output与同一输入官方FA2比较，全部96尾chunk再对独立全heads FP32 oracle。
共1080项数值指标finite且通过原RMS/max-normalized .008界，最大RMS **0.00177940**、
最大normalized max **0.00606061**。audit含shadow、窗口复制/同步，**不作为性能baseline**。

另对基线`dc8cff5`做AST结构核对：selector、store、decode、first-prefill、reset、原packed
attention及原pack分支保持相同。该检查是源码结构证据，运行证据另见审计与三arm结果。

## 新进程对照方法与原始结果

顺序ABC、CBA、BCA（A=flash、B=flash_reuse、C=operator），9个独立模型进程。
每进程单列模型constructor、只编译预热、一次完整archive warm request；随后archive/code
各一次。CUDA同步每个engine step，报告各step之和，排除导入/加载/编译/warmup、step间
状态收集和磁盘写出。不是应用前端网络延迟，也不包含多请求排队。

| 组 | prompt | 原flash完整prefill ms | reuse FA2 ms | operator ms |
| --- | --- | ---: | ---: | ---: |
| 0 | archive | 5843.72 | 5878.93 | 5868.99 |
| 0 | code | 5837.82 | 5862.41 | 5889.44 |
| 1 | archive | 5823.07 | 5904.60 | 5861.16 |
| 1 | code | 5834.72 | 5919.08 | 5886.35 |
| 2 | archive | 5835.72 | 5844.04 | 5850.00 |
| 2 | code | 5833.18 | 5838.38 | 5861.95 |

operator/reuse完整prefill几何平均0.999172；archive0.997318、code1.001028。
reuse/原flash几何平均1.006821（慢0.68%）；复用buffer在本系统测量中没有取得整体收益。
各进程warm request约7.20–7.45s，计时请求约5.82–5.92s；CPU history容量等准备成本
不能隐藏成完全冷请求结果。模型constructor约4.27–4.40s；operator显式预编译约5.34–5.75ms，
已有disk/JIT cache，不是全冷编译。

96尾chunk operator/reuse比几何平均0.991500（少0.85%），配对范围0.905666–1.042199；
archive平均慢1.26%、code平均少2.92%，方向不稳。不能把一次尾部快9.43%作普遍结论。
没有stage profiler新证据，不从这些整体时间推断某个kernel、CPU包装或H2D的独立成本。

## 选择差异：精度容差不能替代模型验收

18项计时请求的首token一致：archive2797、code57912；只覆盖这两个首token。
reuse与原flash每step的selected IDs和逻辑状态完全一致。operator所有4096后续主chunk
选择也一致；差异仅在96尾chunk的后续层，尾部layer0选择仍一致。

每组archive尾部35/36层的ID顺序不同，其中11/36层集合不同；code为31/36和13/36。
三组差异层完全重复。累计864项later-layer配对中，198项顺序不同、72项集合不同；
这实际是24个prompt/layer集合差异各重复3次，不是72个独立质量案例。

selector源码与预算未变，同输入attention误差已量化，而差异在首次custom输出之后出现。
由这些证据推断，attention舍入差异传播到后续Q和selection是主要解释；本轮没有完整
因果profiler/逐值trace，不能保证其他输入的输出或检索质量。没有强制复用baseline IDs
来掩盖这种传播，也没有根据结果重新调M16策略或阈值。

## 故障与证据保存

首次启动未进入模型constructor，GPU已显示GFW boot failure、Xid119/154与Reset。
进程查询返回[N/A]还触发了监控解析错误；失败记录保留，解析器现会明确拒绝不可用信息。
原始诊断在 `/home/tmz/ai/diagnostics/nanokv-gpu-20261007-1256`。
用户完全关机再启动后CUDA分配/同步恢复；本轮结束温度/功耗正常，当前boot内未检出Xid。
这只说明本次恢复和测量完成，不证明长期驱动稳定。未修改驱动或系统电源设置。
NVIDIA的[Xid目录](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html)
说明GSP错误及其恢复行动；本机初始故障根因仍未确定。

原始JSON及逐step/逐层ID已复制进Git：
[`m19-native/summary.json`](../benchmarks/results/operator_bridge/m19-native/summary.json)、
[`suite.json`](../benchmarks/results/operator_bridge/m19-native/suite.json)、
[operator audit](../benchmarks/results/operator_bridge/m19-native/operator-audit-recovered.json)、
[reuse audit](../benchmarks/results/operator_bridge/m19-native/reuse-audit-recovered.json)、
[早期失败](../benchmarks/results/operator_bridge/m19-native/operator-audit-bootstrap-failure.json)、
[默认源码结构审计](../benchmarks/results/operator_bridge/m19-default-source-audit.json)。
summary保存13份tracked副本与原ignored路径的SHA256；不覆盖旧记录。
全部43份worker源码及44份suite源码前后hash一致。复现见
[`NATIVE_SEGMENTED_ADAPTER_USAGE.md`](NATIVE_SEGMENTED_ADAPTER_USAGE.md)。

## 判断

实验入口可用于研究真实布局、动态地址与误差传播；当前证据支持保持默认FA2。
继续优化需先有真实workload成本证据及更完整质量验收，不凭单算子收益扩大系统宣传。
旧M22系统成果与质量限制继续有效，本轮未恢复其全部优化路线。
