# Native 分段算子实验入口与复现

实验入口默认关闭，见[结果与验收边界](NATIVE_SEGMENTED_ADAPTER_REPORT.md)。
只覆盖M12/TP1/eager/单请求；operator要求BF16 Q32/KV8/D128/default scale和SM86。
默认仍为 `sparse_prefill_attention_backend="flash"`。

## 配置

在既有M12配置上，复用FA2实验仅设置：

```python
sparse_prefill_attention_backend="flash_reuse"
```

分段算子实验设置：

```python
sparse_prefill_attention_backend="operator"
sparse_operator_root="/absolute/path/to/llm-gpu-kernels"
```

不自动安装或寻找另一份算子包。显式checkout须保持M18记录37份源码和M16 provenance。
共享scratch只允许同device/stream顺序调用。不能并发保存/消费同一direct输出；上一层消费
须先在同stream排队。reset保留容量和编译缓存，清理逻辑KV状态。

## 复现

从已验证的native NanoKV checkout，使用它自己的venv，保留CUDA12.8环境：

```bash
source .venv/bin/activate
unset CUDA_HOME CUDA_PATH CUDACXX LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python benchmarks/benchmark_segmented_adapter.py \
  --model /home/tmz/models/Qwen3-4B \
  --manifest benchmarks/results/operator_bridge/real-model-capture-manifest.json \
  --operator-root /home/tmz/文档/ChatGPT/llm-gpu-kernels \
  --backend operator --phase audit \
  --output bench_logs/operator_bridge/new-run/operator-audit.json
```

`new-run`为新的目录名，现存输出拒绝覆盖。需要固定模型及model-local源manifest。
此请求driver只使用tracked manifest中的prompt IDs，不需要12份大型capture tensor文件。
同一driver换 `--backend flash_reuse` 可做其逐层audit。

operator audit通过且源码未变，再执行至少三组新进程三arm对照：

```bash
python benchmarks/run_segmented_adapter_suite.py \
  --model /home/tmz/models/Qwen3-4B \
  --manifest benchmarks/results/operator_bridge/real-model-capture-manifest.json \
  --operator-root /home/tmz/文档/ChatGPT/llm-gpu-kernels \
  --audit bench_logs/operator_bridge/new-run/operator-audit.json \
  --output bench_logs/operator_bridge/new-run/fresh-suite
```

suite输出目录必须尚不存在。九个worker串行运行；发现其他GPU计算PID或不可用GPU信息
就保留失败并停止测量。模型13份文件、worker/suite源码、manifest、audit和每个结果均保存hash。
模型加载、预编译、完整warm request与测量区间分别记录，不能把warm条件当完全冷请求。
只生成首token，无decode TPOT或质量数据集验收；检索集合差异须查看suite summary与原逐层记录。
