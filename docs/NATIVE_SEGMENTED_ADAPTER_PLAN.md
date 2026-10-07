# Native 分段算子实验接入与请求对照（预先声明）

2026-10-07 Asia/Shanghai。冻结独立算子M18 `fa4da4d` 与M16选择，NanoKV原默认FA2。

## 接入边界

新增显式 `flash_reuse` 和 `operator` 后端，仅M12/TP1/eager/单请求。
`flash_reuse`直接从同一pinned staging复制到可复用packed目标，保留阻塞H2D语义，
不增加copy stream或pipeline。`operator`在冻结small direct域读取selected/sink/
previous recent/current四段，其他shape与`flash_reuse`相同。

所有层在同一stream顺序执行，共用一套临时packed/selected GPU scratch和至多一个
动态direct workspace；下一层写入前，前一层消费已在同stream排队。跨stream调用拒绝。
避免为36层保存大chunk packed buffer和M18 fallback额外Q输出。KV状态、selector、
gather、Top32、summary1、RoPE、store次序均不改。reset只清KV逻辑，不清编译/临时容量。
operator依赖显式仓库路径并验证M18记录37份source与M16 provenance，无自动安装。

## 验证与测量

固定M17两条prompt原完整IDs，16480token，4096主chunk+96尾chunk，Qwen3-4B pinned
snapshot，BF16、YaRN4、8CPU线程、max_tokens1/greedy；不将主chunk缩小扩大算子收益。
先instrumented audit：每个later-layer output与同输入原FA2比，尾chunk独立FP32 oracle，
原.008/finite；逐层selection保护/无重复、recent/sink/长度/store次序核对。
记录audit时间但不当正式性能。

之后3组独立新进程配对三arm：flash/flash_reuse/operator，顺序ABC、CBA、BCA。
每进程模型加载、显式只编译预热、一次完整archive请求预热单列；随后archive/code各
一次请求。CUDA同步每个step，报告首chunk、后续主chunk、96尾chunk和全部prefill
step和，不含模型加载/编译/预热/磁盘写出；此口径只生成首token，无decode TPOT。
记录greedy ID、每step各layer selected IDs与逻辑状态（收集在计时区间外）。

至少比较operator vs强flash_reuse，并保留原flash对照；不从一次微小变化宣称系统
加速。各prompt三次新进程样本很少，给原始配对值/范围，不能证明普遍显著性。
算子舍入差异可能改变后续Q/selection或greedy；必须公开比较，不强制选择相同来
掩盖漂移，不把old sparse质量问题改写为无损。默认不切换，失败记录保留。
GPU与其他计算任务互斥，检测其他PID则停止本次测量。源码/模型/输入hash全保存。
