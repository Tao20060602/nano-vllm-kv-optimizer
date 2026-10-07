# NanoKV：系统项目的简历证据

2026-10-07。基于nano-vLLM的单请求稀疏推理实验项目；GPU kernel实现保留在独立仓库。

## 可用描述

> 基于nano-vLLM迭代Qwen3-4B稀疏推理原型，维护CPU完整KV历史、GPU代表检索与
> sink/recent保护窗口；实现pinned批量gather、短尾状态修复和可显式开启的分段算子
> adapter，通过逐层数值/状态审计、新进程对照及来源hash评估性能与语义边界。

## 能展示的工程判断

| 主题 | 可追溯证据 |
| --- | --- |
| 全CPU历史、GPU代表与保护窗口、预算/质量权衡 | [项目历程](NANOKV_PROJECT_HISTORY_AND_HANDOFF.md)、[M22收尾](PROJECT_CLOSEOUT.md) |
| 当前KV为何在attention后store，短尾recent重叠/窗口索引修复 | [短尾状态修复](NATIVE_SHORT_TAIL_STATE_FIX.md) |
| 真实模型Q/K/V布局、snapshot与prompt/selection来源 | [真实输入桥接方案](SEGMENTED_OPERATOR_BRIDGE_PLAN.md) |
| 编译模板与数据地址分离、全36层顺序共用scratch | [实验接入报告](NATIVE_SEGMENTED_ADAPTER_REPORT.md) |
| 13模型文件hash、1080项精度/864项状态检查、9个新进程原始结果 | [证据索引](../benchmarks/results/operator_bridge/m19-native/summary.json) |
| 数值容差通过但尾部检索集合变化；本轮无稳定完整prefill收益 | [报告中的性能与选择边界](NATIVE_SEGMENTED_ADAPTER_REPORT.md) |

历史CPU gather/selector优化数字使用各自原报告和适用负载，区分历史WSL与当前native数据。
本轮只生成首token、两个固定文本，没有多请求serving、decode TPOT或广泛质量结论。
保持默认FA2，不能把独立算子33.17%写成NanoKV请求加速或无损稀疏检索。

查阅、无GPU证据核对与真实模型复现分别见[复现索引](REPRODUCTION_INDEX.md)。

## 与独立算子项目的分工

NanoKV展示状态、内存层级、CPU/GPU数据路径、系统测量及性能/质量验收。
LLM GPU Kernels展示Tensor Core、GPU route dispatch、MoE融合、GQA在线softmax、
冻结配置及成熟算子对照。两个项目有不同评价层级，性能数字不能重复计作两项系统收益。
独立仓库的具体可用表述见其 `docs/PORTFOLIO_EVIDENCE.md`。
