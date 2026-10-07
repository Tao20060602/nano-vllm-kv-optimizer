"""Opt-in native TTFT benchmark candidates for the frozen M12 runtime.

The ``baseline`` arm leaves every runtime instance untouched.  The
``direct_store`` arm binds a generated copy of the current ``_store_kv`` method
to each M12 runtime instance, changing only the two GPU-to-CPU KV copies so
they write directly into pageable history storage.

This module intentionally does not edit or import-patch the M12 runtime class.
The generated candidate uses the original runtime module globals and blocks on
each copy before returning, so later CPU gathers cannot observe an incomplete
host write.  Non-contiguous K/V source views are passed directly to
``Tensor.copy_``; no source contiguity assumption is introduced here.
"""

from __future__ import annotations

import hashlib
import inspect
import textwrap
from pathlib import Path
from types import MethodType
from typing import Iterable


# SHA-256 of textwrap.dedent(inspect.getsource(M12LayerRuntime._store_kv)) at
# the audited frozen runtime revision.  Fail closed if that method changes.
EXPECTED_ORIGINAL_FN_SHA256 = (
    "cbbf3820c71ab451c847ff5ab5b1e22d23dc0825e3537207033b786af110cc55"
)

_REPLACEMENTS = (
    (
        "    self.k_cpu[start_block:need].copy_(k_blocks.cpu())",
        "    self.k_cpu[start_block:need].copy_(k_blocks, non_blocking=False)",
    ),
    (
        "    self.v_cpu[start_block:need].copy_(v_blocks.cpu())",
        "    self.v_cpu[start_block:need].copy_(v_blocks, non_blocking=False)",
    ),
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_helper() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _runtime_instances(layers: Iterable[object], runtime_class):
    runtimes = []
    seen = set()
    for layer in layers:
        runtime = layer if type(layer) is runtime_class else getattr(layer, "sparse_rt", None)
        if type(runtime) is not runtime_class:
            raise TypeError(
                "layers must contain M12LayerRuntime instances or model modules "
                "whose sparse_rt is an M12LayerRuntime"
            )
        identity = id(runtime)
        if identity in seen:
            raise ValueError("layers contains the same M12 runtime more than once")
        seen.add(identity)
        runtimes.append(runtime)
    if not runtimes:
        raise ValueError("layers must contain at least one M12 runtime")
    return runtimes


def _current_source(runtime_class) -> tuple[str, str]:
    source = textwrap.dedent(inspect.getsource(runtime_class._store_kv))
    source_sha256 = _sha256_text(source)
    if source_sha256 != EXPECTED_ORIGINAL_FN_SHA256:
        raise RuntimeError(
            "M12LayerRuntime._store_kv source differs from the audited baseline: "
            f"expected {EXPECTED_ORIGINAL_FN_SHA256}, got {source_sha256}"
        )
    return source, source_sha256


def _build_direct_store(runtime_module, original_source: str):
    candidate_source = original_source
    for original, replacement in _REPLACEMENTS:
        count = candidate_source.count(original)
        if count != 1:
            raise RuntimeError(
                "expected exactly one frozen KV-copy statement before replacement; "
                f"found {count}: {original}"
            )
        candidate_source = candidate_source.replace(original, replacement, 1)

    # Prove that the candidate is exactly the original source plus the two
    # declared substitutions, with no formatting or control-flow changes.
    restored_source = candidate_source
    for original, replacement in _REPLACEMENTS:
        if restored_source.count(replacement) != 1:
            raise RuntimeError(
                "candidate source does not contain exactly one declared replacement: "
                f"{replacement}"
            )
        restored_source = restored_source.replace(replacement, original, 1)
    if restored_source != original_source:
        raise RuntimeError("candidate source differs beyond the two declared copies")

    # Resolve globals from the original module, while placing the generated
    # function in a separate locals mapping so the module namespace is untouched.
    candidate_locals = {}
    filename = inspect.getsourcefile(runtime_module.M12LayerRuntime._store_kv)
    code = compile(candidate_source, filename or "<m12-direct-store>", "exec")
    exec(code, vars(runtime_module), candidate_locals)
    function = candidate_locals.get("_store_kv")
    if not inspect.isfunction(function):
        raise RuntimeError("generated source did not define _store_kv")
    return function, candidate_source


def install_candidate(layers, arm: str) -> dict:
    """Record the baseline arm or install direct D2H storage on runtime instances.

    Args:
        layers: Iterable of M12LayerRuntime instances or model modules containing
            them in ``sparse_rt``.
        arm: ``"baseline"`` leaves methods unchanged; ``"direct_store"`` binds
            the two-copy candidate to each supplied runtime instance.

    Returns a JSON-serializable provenance dictionary for the benchmark driver.
    The direct candidate keeps ``non_blocking=False`` to ensure host history is
    complete before any later CPU gather reads it.
    """
    if arm not in ("baseline", "direct_store"):
        raise ValueError("arm must be 'baseline' or 'direct_store'")

    from nanovllm.sparse import m12_runtime as runtime_module

    runtime_class = runtime_module.M12LayerRuntime
    runtimes = _runtime_instances(layers, runtime_class)
    original_source, original_sha256 = _current_source(runtime_class)

    # Do not silently stack this patch or report a patched instance as baseline.
    if any("_store_kv" in runtime.__dict__ for runtime in runtimes):
        raise RuntimeError("an M12 runtime already has an instance _store_kv override")

    replacements = []
    for original, replacement in _REPLACEMENTS:
        replacements.append({"before": original.strip(), "after": replacement.strip()})

    generated_sha256 = original_sha256
    if arm == "direct_store":
        candidate_function, candidate_source = _build_direct_store(
            runtime_module, original_source
        )
        generated_sha256 = _sha256_text(candidate_source)
        installed = []
        try:
            for runtime in runtimes:
                runtime._store_kv = MethodType(candidate_function, runtime)
                installed.append(runtime)
        except Exception:
            for runtime in installed:
                del runtime._store_kv
            raise
        applied_replacements = replacements
    else:
        applied_replacements = []

    return {
        "arm": arm,
        "runtime_count": len(runtimes),
        "layer_ids": [int(runtime.layer_id) for runtime in runtimes],
        "original_function_sha256": original_sha256,
        "generated_function_sha256": generated_sha256,
        "replacement_count": len(applied_replacements),
        "replacements": applied_replacements,
        "helper_sha256": _sha256_helper(),
        "copy_non_blocking": False,
        "source_path": inspect.getsourcefile(runtime_class._store_kv),
        "baseline_untouched": arm == "baseline",
    }
