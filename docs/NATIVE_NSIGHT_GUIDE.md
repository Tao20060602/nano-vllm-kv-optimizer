# Native Linux Nsight Systems CLI

更新时间：2026-10-07。Nsight Systems CLI-only 已解包到用户目录，版本和帮助检查通过。
下述 baseline / direct_store 有界 timeline 已由授权的 M23 worker 完成；编辑本文件时
没有启动 profiler、CUDA workload 或 GPU capture。

## 已核对的主机与工具

| 项目 | 实际值 |
| --- | --- |
| 主机 / OS | `tmz-OMEN-by-HP-Laptop-17-ck0xxx`，Ubuntu 26.04.1 LTS，Linux 7.0.0-38-generic，x86_64 |
| checkout | `/home/tmz/.codex/worktrees/nanokv-native-linux/nanokv`，分支 `codex/nanokv-linux-setup` |
| NanoKV venv | Python 3.12.14；PyTorch 2.7.1+cu128 / CUDA runtime 12.8；Triton 3.3.1；FlashAttention 2.8.3.post1 |
| GPU inventory | NVIDIA GeForce RTX 3080 Laptop GPU，driver 595.91.07，16 GiB |
| Nsight Systems CLI | 2026.5.1.161-265138896106v0，Debian architecture `amd64` |

NVIDIA 官方 Get Started 页面在 2026-10-07 列出的最新版为 2026.5.1，并提供 Linux x86_64 CLI-only `.deb`。下载无登录要求，HTTP 状态为 200。包依赖 `libc6` 与 `libglib2.0-0`；本机动态链接检查未发现缺失库。

下载文件：`/home/tmz/ai/profilers/downloads/nsightsystems-linux-cli-public-2026.5.1.161-3889610.deb`

下载元数据：`/home/tmz/ai/profilers/downloads/nsightsystems-linux-cli-public-2026.5.1.161-3889610.deb.metadata.txt`

SHA-256：`61829db6392e5c293ada1319df86a97c3356bc810335a1975cb551fcfe3eca08`

解包目录：`/home/tmz/ai/profilers/nsight-systems-2026.5.1`

固定可执行文件：

```text
/home/tmz/ai/profilers/nsight-systems-2026.5.1/opt/nvidia/nsight-systems-cli/2026.5.1/bin/nsys
```

它是指向此目录内 `target-linux-x64/nsys` 的相对符号链接。用 `dpkg-deb --extract` 解包；没有执行系统级 `.deb` 安装、维护脚本、全局 PATH 修改或驱动变更。原下载包保留用于审计/复用。

## 命令与捕获范围

调用时使用绝对路径，不需激活 NanoKV venv，也不需改 shell 或系统 PATH：

```bash
/home/tmz/ai/profilers/nsight-systems-2026.5.1/opt/nvidia/nsight-systems-cli/2026.5.1/bin/nsys --version
/home/tmz/ai/profilers/nsight-systems-2026.5.1/opt/nvidia/nsight-systems-cli/2026.5.1/bin/nsys profile --help
```

本机输出确认 `--trace` 接受 `cuda,nvtx,osrt`，`--capture-range` 接受 `cudaProfilerApi`，且 `--sample=none` 与 `--cpuctxsw=none` 都是有效选项。`cudaProfilerApi` 只捕获应用调用 `cudaProfilerStart/Stop` 的范围。

## M23 有界采集记录

两个 profile worker 均以 `status=passed` 退出，生成 report 与 SQLite：

| arm | report | SQLite | worker record |
| --- | --- | --- | --- |
| baseline | `bench_logs/m23-nsys/baseline.nsys-rep` | `bench_logs/m23-nsys/baseline.sqlite` | `bench_logs/m23-nsys/baseline-worker.json` |
| direct_store | `bench_logs/m23-nsys/direct_store.nsys-rep` | `bench_logs/m23-nsys/direct_store.sqlite` | `bench_logs/m23-nsys/direct_store-worker.json` |

每个 capture 仅覆盖冻结的 `archive-T16480` 请求 step index 3（零起始），即第 4 个
4096-token prefill chunk；该 step 开始前已有 12,288 个历史 token。capture-range 在
该 step 的 `cudaProfilerStart/Stop` 内开始并结束，随后应用完成报告导出。M23 profile
wrapper 为每层添加了 `m23.store_k_copy.layerN`、`m23.store_v_copy.layerN` 与
`m23.build_representatives.layerN` NVTX 范围；report 也包含所选范围内的 CUDA API 与
GPU timeline。此次用户态 capture 成功，没有权限阻碍，也没有改驱动或系统配置。

这只证明有界诊断采集已完成；不能当作 candidate 的性能或质量成功。它只采了一个
prompt 的一个 prefill step，而且 profiler/NVTX 会改变执行开销。不要从 timeline 的
CPU range、GPU activity 或两臂 profile 时间推导净 TTFT 改善；净性能按无 profiler 的
M23 fresh-process 对照判断。

### 完整重现命令

以下命令展示相同 workload、capture 参数与结果结构。原报告已存在；重跑必须选一个新的
输出目录，不能覆盖 `bench_logs/m23-nsys/`。`--nsys-step 3` 是零起始 step index。

```bash
cd /home/tmz/.codex/worktrees/nanokv-native-linux/nanokv
source .venv/bin/activate
unset CUDA_HOME CUDA_PATH CUDACXX LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST=8.6
export HF_HOME=/home/tmz/models/Qwen3-4B/.cache/huggingface
mkdir -p bench_logs/m23-nsys-rerun-20261007

/home/tmz/ai/profilers/nsight-systems-2026.5.1/opt/nvidia/nsight-systems-cli/2026.5.1/bin/nsys profile \
  --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop \
  --export=sqlite --sample=none --cpuctxsw=none \
  -o bench_logs/m23-nsys-rerun-20261007/baseline \
  .venv/bin/python benchmarks/profile_native_ttft_nsys.py \
  --model /home/tmz/models/Qwen3-4B \
  --fixture benchmarks/results/native_ttft/fixture.json \
  --arm baseline --phase profile \
  --output bench_logs/m23-nsys-rerun-20261007/baseline-worker.json \
  --only-case archive-T16480 --nsys-step 3

/home/tmz/ai/profilers/nsight-systems-2026.5.1/opt/nvidia/nsight-systems-cli/2026.5.1/bin/nsys profile \
  --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop \
  --export=sqlite --sample=none --cpuctxsw=none \
  -o bench_logs/m23-nsys-rerun-20261007/direct_store \
  .venv/bin/python benchmarks/profile_native_ttft_nsys.py \
  --model /home/tmz/models/Qwen3-4B \
  --fixture benchmarks/results/native_ttft/fixture.json \
  --arm direct_store --phase profile \
  --output bench_logs/m23-nsys-rerun-20261007/direct_store-worker.json \
  --only-case archive-T16480 --nsys-step 3
```

`profile_native_ttft_nsys.py` instruments only the requested step and validates the original
`_store_kv` source before applying the `direct_store` substitutions. Its worker JSON records
source, fixture and model-manifest hashes plus `status`; inspect that status before treating a
report as complete. Choose a fresh output-directory suffix for every run.

### 打开 report 查看时间线

CLI-only package 不含 GUI。若用户已有 Nsight Systems GUI，可在 GUI 中用 **File → Open Report**
打开相应 `.nsys-rep`（例如 `baseline.nsys-rep`）；SQLite 是 CLI/查询副本。选中该 bounded
step 后展开 CPU thread、CUDA context/stream 与 NVTX 行，在 NVTX 搜索/筛选中查找
`m23.store_k_copy.layer`、`m23.store_v_copy.layer` 和 `m23.build_representatives.layer`，
逐层查看 K/V store 与 rep construction 的相对顺序和范围；在 CUDA API 行定位
`cudaStreamSynchronize`，再沿相关 stream 检查同步前后的 GPU kernel/copy 活动。GUI 若在
另一台支持的主机上运行，也可以打开复制过去的 `.nsys-rep`。

这些范围包含 instrumented 运行时的 CPU 提交/等待关系，且 CUDA 活动可能与 CPU range
重叠。它们用于找同步和搬运位置，不是独占阶段耗时；不要把层级范围相加，也不要把
timeline 当无 profiler 的 TTFT 或净加速结果。

`--sample=none` 关闭 CPU instruction-pointer/backtrace sampling；`--cpuctxsw=none` 关闭 OS thread scheduling/context-switch trace。它们不关闭 CUDA/NVTX/OS runtime API trace。此前主机状态记录了 `RmProfilingAdminOnly=1`；它没有阻止本次 Nsight Systems 时间线采集。此次没有读取 GPU 硬件计数器，也没有运行 Nsight Compute，因此不能据此推断计数器访问权限；无需为这些 timeline 改动驱动参数。

安装验证只运行了 `nsys --version`、`nsys profile --help` 与 `nsys stats --help`；后续 M23 workers 创建了上表列出的 `.nsys-rep` 与 `.sqlite`。可用同一 CLI 阅读已完成报告：

```bash
/home/tmz/ai/profilers/nsight-systems-2026.5.1/opt/nvidia/nsight-systems-cli/2026.5.1/bin/nsys stats /path/to/report.nsys-rep
```

## 官方资料

- [NVIDIA Nsight Systems Get Started / Downloads / Release Notes](https://developer.nvidia.com/nsight-systems/get-started)
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
