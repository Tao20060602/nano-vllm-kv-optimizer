# Native Linux 分段算子接入前验证

先修复短尾chunk的recent状态复制，再从现有FA2路径采集真实Qwen3-4B输入。
独立算子仓库仍负责kernel实现；NanoKV只保存模型、KV/selection状态和请求证据。

## 输入与冻结边界

固定Qwen3-4B snapshot `1cfa9a7208912126459214e8b04321603b3df60c`，原生venv
Torch2.7.1+cu128/Triton3.3.1/FA2 2.8.3.post1，SM86。CPU线程8，TP1/eager。
两项固定生成文本prompt（archive/code），每项16480 token，chunk4096，因此正常
四个4096主chunk加96尾chunk；query summary1、Top32、sink64/recent512、index_select。
沿近期系统参考使用YaRN factor4，不调selection或检索预算，不做新质量优化。

仅benchmark实例包裹selector/prefill/attention方法，原FA2输出继续驱动模型。
采集layers0/17/35的第二个4096 chunk及96尾chunk，每prompt6项，共12项。
保存完整Q、四段KV、原输出、prompt IDs、selected/protected IDs、原始strides和
current/previous-recent状态。CPU tensor大文件在ignored bench_logs，JSON保存SHA256。
每请求检查状态重置，每层记录各later chunk selection；采集后recent/sink状态匹配。
数据在原pack之后读取，原current V跨度单独保存，重放时恢复该跨度。

## 重放与成本口径

在独立算子仓库加载可信本地capture文件并验证hash，使用未修改M16冻结policy。
全heads chunk64 FP32 oracle、TF32 off、finite和原0.008界、API/Graph前后及输入hash。
区分旧分配/逐段copy+FA2、复用pack+FA2、M16边界路径（必要current V转换计入）、
M16 contiguous core诊断。所有数据已GPU resident，排除capture、模型、CPU gather/H2D。
四轮正/逆、8样本2调用，Graph8forward；保存raw event/wall/p50/p95和冷prepare。
不使用真实数据重新挑M16配置、阈值或域。

这是实际模型tensor上的operator验证，不是已集成system A/B。采集调用带同步与
CPU/Disk I/O，步骤wall时间仅诊断，不能作为正式系统baseline或加速比例。
若数值失败，保存数据先诊断，不放宽误差或自动将规则接入默认runtime。

后续受控feature-off/on集成需先解决动态地址、非连续V、冷准备与workspace生命周期。
系统比较另做至少三组fresh-process交替A/B，检查greedy输出、selection、KV状态；
不会为了扩大自定义路径收益把4096主chunk改成小chunk，也不恢复64K/128K补测。
