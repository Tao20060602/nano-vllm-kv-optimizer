# NanoKV 服务化第一阶段：真实引擎、请求队列与可复现压测

日期：2026-10-13。基于仓库 `6a6f085`。

## 完成范围

新增 `nanovllm/serving/`，直接接入本仓库的 `LLMEngine`。没有另建推理引擎，
没有把模拟器的加速、质量、成本数据带入 NanoKV 的实测结果。

当前 M12 sparse runtime 的历史 KV、代表向量、recent/sink 与 staging 由单序列
共享，配置明确要求 `max_num_seqs=1`。因此本阶段提供多个 HTTP 客户端的
**有界 FIFO 接入与串行推理**：一个 active，其余 queued。这是服务化基础，
尚非 continuous batching、多序列稀疏 attention 或 prefill/decode 混合调度。

已有 `sparse_quant_history` 与 fused dequant 实验仍保留，默认关闭；历史
M0–M6 prefix reuse 与 M12 sparse runtime 当前不兼容，不在本阶段强行拼接。

## 代码与设计取舍

| 文件 | 职责与取舍 |
| --- | --- |
| `nanovllm/serving/core.py` | 单个 owner thread 创建模型并执行 encode/add/step/cancel；避免 HTTP 线程覆盖 shared staging。active + queued 达上限返回 429。 |
| `nanovllm/serving/server.py` | 文本 chat completion、SSE/非流式输出、真实 token ID/usage、health 与 Prometheus 文本指标。 |
| `nanovllm/serving/benchmark.py` | 对真实 SSE 接口计时，保存逐请求结果、整个压测窗口、配置、commit、工作负载 hash 与后端快照。 |
| `LLMEngine.add_request` | 向后兼容地返回 Sequence，服务可观察每一步新生成 token。 |
| `LLMEngine.cancel_request` | 引擎 owner thread 上移除请求；dense 释放 block，sparse 清理 active history。 |
| `tests/test_serving_foundation.py` | 无 CUDA 的契约测试、本地真实 HTTP/SSE 集成测试。FakeEngine 仅出现在测试中，不提供演示生产模式。 |

失败处理：

- 文本/参数/长度错误：拒绝该请求，引擎继续服务。
- active/queued 取消、排队 deadline、总请求 deadline：释放请求；active 在下一次
  `step()` 边界取消，已经发出的 CUDA 工作不能在中途打断。
- 模型执行/反解码异常：停止接入，所有 queued 请求收到失败，health 返回 503；
  必须重启进程。没有声称从 CUDA OOM/设备错误中自动恢复。
- 客户端断连：SSE 心跳或写输出失败时设置取消；断连检测并非瞬时。
- 输出事件最多受 `max_output_tokens` 和 outstanding 上限限制；没有无限生产 token。

## 实际运行环境与验证边界

此次开发在独立 Linux 云端 checkout 完成，没有 GPU、模型、torch 或 FlashAttention。
未修改用户电脑上的环境，未安装重复模型，未运行 GPU 性能基准。
CPU unittest 与 localhost HTTP 集成验收只证明服务/测量契约，不能替代
Qwen3-4B、CUDA、tokenizer、M12 reset 和 driver 的本机验收。

在仓库记录的 native Linux checkout/venv 上运行下面命令。保持当前已验证的
Qwen3-4B snapshot，先查看 `docs/environment.md` 并确认自己的实际模型路径。

```bash
source .venv/bin/activate
python -m unittest discover -s tests -p test_serving_foundation.py -v

python -m nanovllm.serving.server \
  --model /home/tmz/models/Qwen3-4B \
  --engine-config configs/serving_m12.json \
  --max-outstanding 4 --max-output-tokens 256 \
  --queue-timeout 60 --host 127.0.0.1 --port 8080
```

`configs/serving_m12.json` 是需验收的启动示例，不是新的性能结果。
256 是原引擎 paged block 配置约束，64 是 sparse retrieval block，两者不同。
128K 是 context 上限，输入和允许生成的 token 总和不得超过上限。
新增 `LLM` 配置错误会显式报错，避免原构造器静默丢弃未知字段。
Qwen3 是否启用 thinking 由 tokenizer chat template 默认控制，本阶段没有新增
thinking 开关；A/B 必须固定 chat template 与输入 token，不能混用旧的裸 prompt 数值。

另开终端，先 smoke（不是性能基线）：

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/metrics
curl --no-buffer http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-4B","messages":[{"role":"user","content":"介绍 KV cache"}],"max_tokens":32,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'

python -m nanovllm.serving.benchmark \
  --requests benchmarks/serving_smoke.jsonl --concurrency 2 \
  --model Qwen3-4B --endpoint http://127.0.0.1:8080 \
  --warmup 1 --engine-config configs/serving_m12.json \
  --output bench_logs/serving/c2-smoke
```

输出目录必须不存在，避免覆盖旧基线；换目录运行 c1/c2/c4。API key 可通过
`NANOKV_API_KEY` 同时设置服务和压测，不写入报告。默认仅监听 localhost。
这不是完整 OpenAI API：只支持 text messages、model、max_tokens、temperature、
stream、stream_options.include_usage 和 timeout_s；其他生成参数明确拒绝。

## 指标口径

- 客户端 TTFT：HTTP 发出到首个生成 token ID 或非空 content/reasoning content。
  role、usage、finish、心跳不计首 token。没有 token IDs 的通用后端只能观测首内容事件。
- TPOT：`(最后生成事件 - 首生成事件)/(实际输出 token 数 - 1)`；只输出一个 token 时
  为 null。token 数来自服务 usage 或显式 token IDs，不能用 SSE 消息条数。
- E2E：HTTP 发出至最后生成事件；另报包含终止协议的 request_wall_ms。
- 吞吐：成功输出 token 总数 / 整个压测墙钟窗口（含 paced 到达和 drain）。
  计数不完整时 token 吞吐为 null；没有猜测 token 数。
- stream_event_gaps_ms：事件间隔，不自动当成逐 token ITL。一个事件可能含多个 token。
- SLO 达成率：同时满足 TTFT/TPOT 的成功请求 / **所有提交请求**，失败也进分母。
  阈值是实验参数，默认阈值不是服务承诺；goodput 是达标请求/秒或 token/秒。
- queue_ms：服务器 accepted 到 worker 开始处理；服务器 TTFT 包含这段等待。
- `qps=0`：固定客户端并发；`qps>0`：按间隔提交，但仍受客户端线程上限限制，
  必须观察 client_admission_delay_ms。它不是无限并发的纯开环负载发生器。
- 服务 `/metrics` 的延迟 quantile 来自最近 1024 个成功请求，不是跨实例可聚合 histogram；
  不包含 GPU/KV 水位假值，不提供扩缩容或降低卡数的实测结论。
- warmup 会影响 backend prefix cache；本工具不会自动清缓存。必须固定冷/热条件。

## 本机验收与下一阶段

1. 相同 chat template / greedy 输入，服务输出 token IDs 与直接引擎输出一致。
2. 两个并发客户端成功排队；四/八并发观察 queue_ms 和 429，确认没有第二条
   Sequence 进入 sparse runtime，取消后下一条请求不继承 KV。
3. 32K 长请求 + 短请求混合，记录串行基线的 TTFT、token 间隔、排队和成功率。
4. int8 A/B：使用 `sparse_quant_history`（已有功能），其余策略不变；重启服务、
   固定 warmup/负载，各做新进程配对。延迟必须同时记录质量结果。
5. 下一阶段先把 M12 state 按 request ID 拆开、验证 staging 与 KV 生命周期，
   再实现 decode 优先的 chunk 调度。真正 batching 的收益必须在 GPU 上测量。
6. radix tree 接入应复用真实 KV block/代表向量及 fingerprint 生命周期，不把
   本次参考包的整数 block 模拟器复制成所谓真实缓存。跨请求 Q-summary 依赖需要先审计。

## 来源与内化

参考用户提供的 `LLM-Inference-Acceleration-Project.zip` 中实验矩阵、服务治理、
压测和报告组织方式。原压测器的 SSE 计数/墙钟分母问题、MockEngine 切换问题、
前缀树覆盖问题没有直接搬入。新增代码在 NanoKV 真实引擎契约下重写，沿用本仓库
MIT 授权；原 nano-vLLM attribution 保留。参考包中的生产数字不属于本次成果。

应能够自行解释：为何 worker 单独持有引擎、排队不等于 batching、为什么 TPOT
分母为 N-1、取消时谁释放 KV、为什么压测窗口不能用最长单请求耗时替代，以及
量化/前缀复用/调度收益要如何单独验证。
