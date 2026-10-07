# Native引擎收尾：分段prefill成本诊断

2026-10-07，测量前登记。用于解释现有无稳定完整prefill收益的结果，不搜索新策略。
三种backend保持原样：flash、flash_reuse、operator。固定M17 archive/code prompt IDs、
Qwen3-4B snapshot、Top32、summary1、sink64/recent512、BF16、TP1/eager。

复用已冻结的 `benchmark_segmented_adapter.py` 驱动及其完整warm request；新wrapper
只在第二个请求archive的step3（T4096、prior12288）和step4（T96、prior16384）启用
现有PyTorch范围与CPU/CUDA profiler。三种backend各一个新进程，串行运行；共六个窗口。
不修改被冻结的runtime、策略或旧测量记录。源hash前后核对，保存失败与原始trace。

读取已有 `m12.prefill_selector/cpu_gather/h2d_pack/attention/store_kv` 及嵌套
`selector_id_d2h` 范围。输出CPU inclusive/self时间、设备活动区间并集、GPU kernel/
Memcpy原始名称分组、活动总量与调用次数。按名称分组仅用于辨认执行路径，不能代替
逐调用关联、硬件利用率或独占瓶颈分析。CPU范围可能包含等待，嵌套范围不可相加。

profile改变执行成本，全部worker step时间标记为instrumented，不用于性能比较。
正式净收益结论仍使用已有九个未profile新进程数据。所有诊断只覆盖两种chunk、一条
prompt、单次窗口；不证明所有负载的因果机制或质量。首token与schedule检查仍执行。

当前PyTorch2.7.1+cu128能采集CUDA活动；Nsight未安装，硬件计数器受管理员限制。
本次无需root或驱动变更。如果后续需要Tensor Core/DRAM/occupancy计数器，另外明确
提出所需权限和测量对象，不将静态HMMA或活动时间误称为利用率。
