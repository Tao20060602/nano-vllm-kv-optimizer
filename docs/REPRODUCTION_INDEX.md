# Native NanoKV复现与证据索引

2026-10-07。M19接入实验已完成，默认仍为FA2。独立算子的局部收益不计作请求收益。

| 查阅目的 | 入口 |
| --- | --- |
| 项目历史与原有质量边界 | [历程与交接](NANOKV_PROJECT_HISTORY_AND_HANDOFF.md)、[M22收尾](PROJECT_CLOSEOUT.md) |
| Native短尾recent状态修复 | [修复报告](NATIVE_SHORT_TAIL_STATE_FIX.md) |
| M19接入、性能、选择差异与GPU中断 | [接入报告](NATIVE_SEGMENTED_ADAPTER_REPORT.md) |
| 首token请求audit与九个串行worker | [复现命令](NATIVE_SEGMENTED_ADAPTER_USAGE.md) |
| 已提交原始JSON、hash、配对汇总 | [M19证据索引](../benchmarks/results/operator_bridge/m19-native/summary.json) |
| 求职项目表述与工程判断 | [系统项目证据](PORTFOLIO_ENGINE_EVIDENCE.md) |

## 无GPU核对两仓库

独立算子仓库提供仅用Python 3.12标准库的只读入口。从其根目录运行：

```bash
python3 scripts/check_evidence.py --nanokv-root /absolute/path/to/nanokv
```

它核对已提交M19文件、冻结源码、worker和原始数值/状态计数；从九个worker重算配对
性能与检索差异。无需模型、CUDA、外部Python包或大型采集张量。记录内原机器绝对路径
作为历史来源保留，检查读取当前checkout的已提交路径。详细范围与本机检查结果在算子
仓库的 `docs/REPRODUCTION_INDEX.md` 与 `results/evidence-integrity-20261007.json`。
这只证明记录完整，不能证明当前模型文件、驱动稳定、生产质量或性能收益。

## 有GPU重跑

使用当前native checkout与其venv，按[实验使用说明](NATIVE_SEGMENTED_ADAPTER_USAGE.md)
先audit，再运行suite。需要固定Qwen3-4B snapshot与model-local来源manifest；不能替换
为0.6B。driver使用已提交[采集manifest](../benchmarks/results/operator_bridge/real-model-capture-manifest.json)
中的prompt IDs，不需要M17采集的12份大型`.pt`文件。

M19只覆盖M12/TP1/eager/单请求、两个固定文本与首token。模型加载、预编译、warmup、
状态收集和磁盘I/O不在prefill计时中。默认保留FA2；实验operator须显式开启。
