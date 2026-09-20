"""Public nano-vLLM API.

The imports are lazy so lightweight CPU-side NanoKV modules (metrics, prefix
index tests, and documentation tooling) do not require CUDA-only dependencies
just to be imported. The public ``from nanovllm import LLM, SamplingParams``
API remains unchanged.
"""

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    if name == "LLM":
        from nanovllm.llm import LLM

        return LLM
    if name == "SamplingParams":
        from nanovllm.sampling_params import SamplingParams

        return SamplingParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
