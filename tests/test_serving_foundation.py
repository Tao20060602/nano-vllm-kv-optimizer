"""CPU-only contracts for the real-engine service boundary (no GPU claims)."""
import ast
import io
import json
import queue
import threading
import time
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.serving.core import Service, ServiceError
from nanovllm.serving.server import make_server
from nanovllm.serving.benchmark import main as benchmark_main, measure_one, read_events, run, summarize


class FakeEngine:
    max_model_len = 100

    def __init__(self, delay=.002, fail=False):
        self.delay = delay
        self.fail = fail
        self.closed = False
        self.cancelled = 0
        self.owners = []
        self.seq = None

    def encode(self, messages):
        self.owners.append(threading.get_ident())
        return [1, 2]

    def decode(self, ids):
        return "x" * len(ids)

    def add(self, ids, params):
        if self.seq is not None:
            raise AssertionError("more than one sequence entered the sparse engine")
        self.seq = SimpleNamespace(is_finished=False, completion_token_ids=[], max_tokens=params.max_tokens)
        return self.seq

    def step(self):
        self.owners.append(threading.get_ident())
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("injected engine failure")
        seq = self.seq
        seq.completion_token_ids.append(7)
        if len(seq.completion_token_ids) == seq.max_tokens:
            seq.is_finished = True
            self.seq = None

    def cancel(self, seq):
        self.cancelled += 1
        seq.is_finished = True
        self.seq = None

    def close(self):
        self.closed = True


def terminal(job):
    while True:
        event = job.events.get(timeout=3)
        if event["type"] != "token":
            return event


class ServiceTests(unittest.TestCase):
    def start_service(self, engine=None, **kwargs):
        engine = engine or FakeEngine()
        service = Service(lambda: engine, **kwargs)
        service.start(timeout_s=1)
        self.addCleanup(service.close)
        return service, engine

    def test_two_clients_share_one_owner_without_state_overlap(self):
        service, engine = self.start_service()
        a = service.submit([{"role": "user", "content": "a"}], 4)
        b = service.submit([{"role": "user", "content": "b"}], 2)
        self.assertEqual(terminal(a)["output_tokens"], 4)
        self.assertEqual(terminal(b)["output_tokens"], 2)
        self.assertEqual(len(set(engine.owners)), 1)
        self.assertEqual(service.snapshot()["outstanding"], 0)

    def test_capacity_rejection(self):
        service, _ = self.start_service(FakeEngine(delay=.05), max_outstanding=1)
        job = service.submit([{"role": "user", "content": "a"}], 3)
        with self.assertRaises(ServiceError) as ctx:
            service.submit([{"role": "user", "content": "b"}], 2)
        self.assertEqual(ctx.exception.status, 429)
        terminal(job)

    def test_active_cancel_releases_state_for_next_request(self):
        service, engine = self.start_service(FakeEngine(delay=.01))
        a = service.submit([{"role": "user", "content": "a"}], 20)
        self.assertEqual(a.events.get(timeout=1)["type"], "token")
        a.cancelled.set()
        self.assertEqual(terminal(a)["status"], "cancelled")
        b = service.submit([{"role": "user", "content": "b"}], 1)
        self.assertEqual(terminal(b)["output_tokens"], 1)
        self.assertEqual(engine.cancelled, 1)

    def test_queued_cancel_does_not_enter_engine(self):
        service, engine = self.start_service(FakeEngine(delay=.01))
        a = service.submit([{"role": "user", "content": "a"}], 3)
        b = service.submit([{"role": "user", "content": "b"}], 2)
        b.cancelled.set()
        terminal(a)
        self.assertEqual(terminal(b)["status"], "cancelled")
        self.assertEqual(len(engine.owners), 4)  # one encode + three steps

    def test_request_deadline_cancels_engine(self):
        service, engine = self.start_service(FakeEngine(delay=.02))
        job = service.submit([{"role": "user", "content": "a"}], 20, timeout_s=.005)
        self.assertEqual(terminal(job)["status"], "timeout")
        self.assertEqual(engine.cancelled, 1)

    def test_queue_deadline(self):
        service, _ = self.start_service(FakeEngine(delay=.02), queue_timeout_s=.005)
        a = service.submit([{"role": "user", "content": "a"}], 3)
        b = service.submit([{"role": "user", "content": "b"}], 2)
        terminal(a)
        self.assertEqual(terminal(b)["status"], "timeout")

    def test_engine_failure_fails_closed_and_drains_queue(self):
        service, _ = self.start_service(FakeEngine(delay=.01, fail=True))
        a = service.submit([{"role": "user", "content": "a"}], 2)
        b = service.submit([{"role": "user", "content": "b"}], 2)
        self.assertEqual(terminal(a)["status"], "failed")
        self.assertEqual(terminal(b)["status"], "unavailable")
        self.assertFalse(service.snapshot()["healthy"])
        self.assertEqual(service.snapshot()["outstanding"], 0)
        with self.assertRaises(ServiceError):
            service.submit([{"role": "user", "content": "c"}], 1)

    def test_invalid_lengths_do_not_poison_engine(self):
        service, _ = self.start_service(max_output_tokens=200)
        bad = service.submit([{"role": "user", "content": "a"}], 100)
        self.assertEqual(terminal(bad)["status"], "invalid")
        good = service.submit([{"role": "user", "content": "b"}], 1)
        result = terminal(good)
        self.assertIsNone(result["tpot_ms"])
        self.assertTrue(service.snapshot()["healthy"])

    def test_invalid_temperature_and_max_tokens(self):
        service, _ = self.start_service()
        for kw in ({"temperature": float("nan")}, {"max_tokens": 0}, {"max_tokens": True}):
            with self.assertRaises(ServiceError):
                service.submit([{"role": "user", "content": "a"}], **kw)


class BenchmarkTests(unittest.TestCase):
    def response(self, events, done=True):
        body = ": keepalive\n\n" + "".join("data: " + json.dumps(e) + "\n\n" for e in events)
        if done:
            body += "data: [DONE]\n\n"
        obj = io.BytesIO(body.encode())
        obj.headers = {"Content-Type": "text/event-stream"}
        return obj

    def measure(self, response):
        with patch("urllib.request.urlopen", return_value=response):
            return measure_one("http://unused", "model", {"id": "r", "messages": [{"role": "user", "content": "a"}]})

    def test_role_finish_usage_are_not_tokens(self):
        events = [{"choices": [{"delta": {"role": "assistant"}}]},
                  {"choices": [{"delta": {"content": "AB"}}]},
                  {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                  {"choices": [], "usage": {"completion_tokens": 2}}]
        r = self.measure(self.response(events))
        self.assertTrue(r["success"])
        self.assertEqual(r["output_tokens"], 2)
        self.assertEqual(r["content_events"], 1)
        self.assertEqual(r["token_count_source"], "server_usage")

    def test_role_only_does_not_establish_ttft(self):
        r = self.measure(self.response([{"choices": [{"delta": {"role": "assistant"}}]}]))
        self.assertFalse(r["success"])
        self.assertIsNone(r["ttft_ms"])

    def test_unknown_count_is_not_event_count(self):
        r = self.measure(self.response([{"choices": [{"delta": {"content": "many tokens"}}]}]))
        self.assertTrue(r["success"])
        self.assertIsNone(r["output_tokens"])
        self.assertIsNone(r["tpot_ms"])
        self.assertIsNone(summarize([r], 1)["throughput_tokens_per_s"])

    def test_truncated_stream_is_failure(self):
        r = self.measure(self.response([{"choices": [{"delta": {"content": "x"}}]}], done=False))
        self.assertFalse(r["success"])

    def test_usage_disagreement_is_failure(self):
        r = self.measure(self.response([{"choices": [{"delta": {"token_ids": [1], "content": "x"}}]},
                                       {"usage": {"completion_tokens": 2}}]))
        self.assertFalse(r["success"])

    def test_whole_window_and_all_request_slo_denominator(self):
        row = {"success": True, "ttft_ms": 2, "tpot_ms": 3, "e2e_ms": 5, "output_tokens": 2}
        s = summarize([row, {"success": False}], 10)
        self.assertEqual(s["throughput_tokens_per_s"], .2)
        self.assertEqual(s["slo"]["joint_attainment_all_requests"], .5)

    def test_multiline_sse_and_comments(self):
        body = io.BytesIO(b': hi\n\ndata: {"choices":\ndata: []}\n\n')
        self.assertEqual(json.loads(next(read_events(body))), {"choices": []})

    def test_heartbeat_does_not_bypass_wall_deadline(self):
        with self.assertRaises(TimeoutError):
            list(read_events(io.BytesIO(b": keepalive\n\n"), time.perf_counter() - 1))

    def test_malformed_event_is_reported_as_failure(self):
        self.assertFalse(self.measure(self.response([[1, 2]]))["success"])


class HTTPIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.service = Service(lambda: FakeEngine(), max_outstanding=4)
        self.service.start(timeout_s=1)
        self.server = make_server(self.service, "test-model", port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = "http://127.0.0.1:" + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(1)
        self.service.close()

    def test_stream_benchmark_metrics_and_health(self):
        requests = [{"id": str(i), "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3} for i in range(4)]
        rows, summary = run(self.endpoint, "test-model", requests, concurrency=2)
        self.assertEqual(summary["success"], 4)
        self.assertTrue(all(r["output_tokens"] == 3 for r in rows))
        self.assertEqual(sum(r["content_events"] for r in rows), 12)
        with urllib.request.urlopen(self.endpoint + "/healthz") as r:
            self.assertEqual(json.load(r)["execution_mode"], "single_sequence_fifo")
        with urllib.request.urlopen(self.endpoint + "/metrics") as r:
            self.assertIn('nanokv_requests_total{status="success"} 4', r.read().decode())

    def test_nonstream_usage_and_wrong_model(self):
        payload = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}
        req = urllib.request.Request(self.endpoint + "/v1/chat/completions", data=json.dumps(payload).encode())
        with urllib.request.urlopen(req) as r:
            body = json.load(r)
        self.assertEqual(body["usage"]["completion_tokens"], 2)
        self.assertEqual(body["choices"][0]["message"]["content"], "xx")
        payload["model"] = "wrong"
        req = urllib.request.Request(self.endpoint + "/v1/chat/completions", data=json.dumps(payload).encode())
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 404)

    def test_cli_persists_measurements_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workload = root / "workload.jsonl"
            workload.write_text(json.dumps({"id": "r", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}) + "\n")
            args = ["--endpoint", self.endpoint, "--model", "test-model", "--requests", str(workload),
                    "--output", str(root / "run"), "--warmup", "1"]
            with patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(benchmark_main(args), 0)
            metadata = json.loads((root / "run/metadata.json").read_text())
            self.assertEqual(metadata["mode"], "http_measured")
            self.assertEqual(metadata["server_snapshot_before_warmup"]["execution_mode"], "single_sequence_fifo")
            self.assertEqual(json.loads((root / "run/summary.json").read_text())["success"], 1)
            self.assertEqual(json.loads((root / "run/requests.jsonl").read_text())["output_tokens"], 2)
            with self.assertRaises(FileExistsError):
                benchmark_main(args)

    def test_http_capacity_returns_429(self):
        self.service.max_outstanding = 1
        busy = self.service.submit([{"role": "user", "content": "hi"}], 50)
        self.assertEqual(busy.events.get(timeout=1)["type"], "token")
        payload = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}
        req = urllib.request.Request(self.endpoint + "/v1/chat/completions", data=json.dumps(payload).encode())
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 429)
        busy.cancelled.set()
        terminal(busy)


class EngineCancelTests(unittest.TestCase):
    def method(self):
        # Execute the actual method without importing the CUDA model runner.
        source = Path(__file__).resolve().parents[1] / "nanovllm/engine/llm_engine.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LLMEngine")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "cancel_request")
        scope = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
        return scope["cancel_request"]

    def test_sparse_cancel_resets_active_history_once(self):
        from collections import deque
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams
        seq = Sequence([1, 2], SamplingParams())
        seq.num_cached_tokens = 2
        calls = []
        owner = SimpleNamespace(scheduler=SimpleNamespace(waiting=deque(), running=deque([seq]), sparse=True),
                                model_runner=SimpleNamespace(call=calls.append))
        cancel = self.method()
        self.assertTrue(cancel(owner, seq.seq_id))
        self.assertEqual(calls, ["sparse_reset"])
        self.assertFalse(cancel(owner, seq.seq_id))

    def test_dense_cancel_releases_block_table(self):
        from collections import deque
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams
        seq = Sequence([1], SamplingParams())
        seq.block_table = [4]
        freed = []
        owner = SimpleNamespace(scheduler=SimpleNamespace(waiting=deque([seq]), running=deque(), sparse=False,
                                block_manager=SimpleNamespace(deallocate=freed.append)))
        self.assertTrue(self.method()(owner, seq.seq_id))
        self.assertEqual(freed, [seq])


if __name__ == "__main__":
    unittest.main()
