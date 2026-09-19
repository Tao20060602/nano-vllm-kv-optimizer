import unittest
import importlib.util
import sys
from pathlib import Path


METRICS_PATH = Path(__file__).parents[1] / "nanovllm" / "metrics.py"
SPEC = importlib.util.spec_from_file_location("nanovllm_metrics", METRICS_PATH)
METRICS_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = METRICS_MODULE
SPEC.loader.exec_module(METRICS_MODULE)

EngineMetrics = METRICS_MODULE.EngineMetrics
RequestMetrics = METRICS_MODULE.RequestMetrics


class RequestMetricsTest(unittest.TestCase):
    def test_latency_values(self):
        metrics = RequestMetrics(
            seq_id=1,
            prompt_tokens=10,
            started_at=1.0,
            first_token_at=1.2,
            finished_at=1.6,
            completion_tokens=5,
        )

        self.assertAlmostEqual(metrics.ttft_seconds, 0.2)
        self.assertAlmostEqual(metrics.tpot_seconds, 0.1)
        self.assertAlmostEqual(metrics.e2e_seconds, 0.6)


class EngineMetricsTest(unittest.TestCase):
    def test_summary_aggregates_phase_throughput(self):
        metrics = EngineMetrics()
        metrics.register_request(seq_id=0, prompt_tokens=8, started_at=1.0)
        metrics.record_request_progress(0, completion_tokens=1, is_finished=False, timestamp=1.1)
        metrics.record_request_progress(0, completion_tokens=3, is_finished=True, timestamp=1.3)
        metrics.record_step("prefill", num_sequences=1, num_tokens=8, elapsed_seconds=0.2)
        metrics.record_step("decode", num_sequences=1, num_tokens=3, elapsed_seconds=0.3)

        summary = metrics.summary()

        self.assertEqual(summary["requests"]["completed"], 1)
        self.assertAlmostEqual(summary["phases"]["prefill"]["throughput_tokens_per_second"], 40.0)
        self.assertAlmostEqual(summary["phases"]["decode"]["throughput_tokens_per_second"], 10.0)
        self.assertAlmostEqual(summary["latency"]["ttft"]["mean_ms"], 100.0)


if __name__ == "__main__":
    unittest.main()
