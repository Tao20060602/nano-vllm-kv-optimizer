# Native Linux 短尾 chunk 的 recent 状态修复

2026-10-06，受控算子接入前检查发现：历史4096后追加96-token prefill尾chunk，
原recent窗口移动使用重叠切片copy_，Torch2.7.1会报unsupported operation并停止。
失败记录保留在[before JSON](../benchmarks/results/operator_bridge/recent-carry-before.json)。

另外，窗口尚未达到512时，旧代码按更新后recent_len计算旧数据源区间；例如
旧历史128、当前96、new recent_len224、carry128，原代码读[96,224)，应读旧窗口
[0,128)。这会读到未定义旧窗口尾部，并丢失有效历史。

修复只在_store_kv的短current分支：使用old_recent_len=min(recent_tokens,valid_len)
选择旧窗口后缀，在覆盖重叠区域前clone；K/V两条路径一致。attention仍在store前
消费旧recent，selection/Top-K和基本模型路径未调整。

使用当前native项目venv，初始长度64/128/448/512/4096 × 当前2/32/64/96/128/512，
共30项独立状态检查通过：recent K/V逐bit等于连接历史与当前后的最后512个token，
sink K/V未变，valid_len正确。记录：[after JSON](../benchmarks/results/operator_bridge/recent-carry-after.json)。
该检查只声明窗口/长度状态，不证明非整block中间chunk的所有retrieval语义。

随后固定Qwen3-4B snapshot1cfa9a7208912126459214e8b04321603b3df60c、两条16480-token
prompt、4096主chunk/96尾chunk的FA2采集完成。每请求144个later-layer调用、请求重置
和12个目标采集的post recent/sink状态通过。模型输出仍由原FA2驱动，采集wall受
同步/磁盘I/O影响，不能作为性能或质量A/B。详见[计划](SEGMENTED_OPERATOR_BRIDGE_PLAN.md)
和[manifest](../benchmarks/results/operator_bridge/real-model-capture-manifest.json)。
