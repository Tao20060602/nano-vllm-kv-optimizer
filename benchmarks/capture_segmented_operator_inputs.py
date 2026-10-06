"""Capture real FA2 tensors/state for the independent segmented operator lab.

Hooks exist only in this benchmark's runtime instances. Captured engine wall
times include auditing and disk I/O and are not clean system performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
import traceback
from pathlib import Path

import torch
import triton
import flash_attn
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.sparse.m12_runtime import prefill_recent_slice
from benchmark_m14_prefill import YARN

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = {
    'archive': 'The archive records that every experiment needs a reproducible configuration and a clear measurement boundary. Each chapter records its date, the selected evidence, and the unresolved question. ',
    'code': 'A service receives a request, validates the inputs, reads its cache, performs a calculation, and writes an answer. Explain how resource ownership and synchronization preserve correctness across these steps. ',
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('capture directory exists; preserve the old run')
    args.output.mkdir(parents=True)
    manifest_path = args.output/'manifest.json'
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    source_files = sorted(set(
        [p.relative_to(ROOT).as_posix() for p in (ROOT/'nanovllm').rglob('*.py')]
        + ['benchmarks/capture_segmented_operator_inputs.py',
           'benchmarks/benchmark_m14_prefill.py', 'docs/SEGMENTED_OPERATOR_BRIDGE_PLAN.md']))
    hashes = {n: file_hash(ROOT/n) for n in source_files}
    model_manifest = args.model/'.cache/nanokv/source-manifest.json'
    model_data = json.loads(model_manifest.read_text())
    if model_data['revision'] != '1cfa9a7208912126459214e8b04321603b3df60c':
        raise RuntimeError('wrong Qwen3-4B snapshot')
    # Verify weights, not just the presence of the model directory.
    weights = {}
    for name, meta in model_data['files'].items():
        if name.endswith('.safetensors'):
            actual = file_hash(args.model/name)
            if actual != meta['sha256']:
                raise RuntimeError('weight hash mismatch: '+name)
            weights[name] = actual
    record = {
        'schema': 1, 'status': 'running', 'run_id': args.output.name,
        'capture_scope': 'instrumented existing FA2 engine; wall includes CPU copies/disk I/O, not clean performance',
        'source_hashes': hashes,
        'git_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'git_status': subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        'dirty_diff_sha256': hashlib.sha256(subprocess.check_output(['git','diff'],cwd=ROOT)).hexdigest(),
        'model': str(args.model.resolve()), 'model_revision': model_data['revision'],
        'model_manifest_sha256': file_hash(model_manifest), 'weight_sha256': weights,
        'config': {'seq_len': 16480, 'chunk_size': 4096, 'query_segments': 1,
                   'top_k': 32, 'sink': 64, 'recent': 512, 'index_select': True,
                   'rope': YARN, 'max_tokens': 1, 'temperature': 0., 'threads': 8},
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                    'cuda': torch.version.cuda, 'triton': triton.__version__,
                    'fa2': flash_attn.__version__, 'gpu': torch.cuda.get_device_name(0)},
        'requests': [], 'captures': [],
    }

    def save():
        manifest_path.write_text(json.dumps(record, indent=2)+'\n')

    save()
    llm = None
    active = {}
    pending = {}
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        llm = LLM(str(args.model), enforce_eager=True, tensor_parallel_size=1,
                  max_num_seqs=1, max_num_batched_tokens=4096, max_model_len=16512,
                  dtype='bfloat16', gpu_memory_utilization=.7,
                  enable_sparse_attention=True, use_m12_runtime=True,
                  sparse_selector='query_guided', sparse_retrieval_block_size=64,
                  sparse_num_representatives=4, sparse_recent_tokens=512,
                  sparse_first_tokens=64, sparse_top_k=32, sparse_prefill_top_k=32,
                  sparse_decode_top_k=32, sparse_gather_index_select=True,
                  sparse_prefill_chunk_size=4096, sparse_prefill_query_segments=1,
                  sparse_prefill_attention_backend='flash', rope_scaling_override=YARN)
        layers = [m.sparse_rt for m in llm.model_runner.model.modules()
                  if getattr(m, 'sparse_rt', None) is not None]
        if len(layers) != 36:
            raise RuntimeError('unexpected layer count')

        def install(rt):
            original_select = rt._gpu_select
            original_attention = rt._prefill_attention
            original_chunk = rt.prefill_chunk

            def select(*a, **kw):
                result = original_select(*a, **kw)
                pending[rt.layer_id]['ids'] = result[0].tolist()
                return result

            def attention(q, k, v, causal_from):
                info = pending[rt.layer_id]
                ids = info['ids']
                sink_len = min(64, rt.valid_len)
                _, offset, recent_len = prefill_recent_slice(rt.valid_len, 64, 512)
                lengths = (len(ids)*64, sink_len, recent_len, q.shape[0])
                if sum(lengths[:-1]) != causal_from or sum(lengths) != k.shape[0]:
                    raise RuntimeError('capture boundaries differ from packed layout')
                if set(ids) & rt.protected_blocks or len(ids) != len(set(ids)):
                    raise RuntimeError('selected/protected overlap or duplicate IDs')
                info.update(lengths=lengths, selected_ids=ids,
                            protected_ids=sorted(rt.protected_blocks),
                            recent_offset=offset, causal_from=causal_from)
                output = original_attention(q, k, v, causal_from=causal_from)
                capture = (rt.layer_id in (0,17,35)
                           and ((rt.valid_len == 4096 and q.shape[0] == 4096)
                                or q.shape[0] == 96))
                if capture:
                    segments = []
                    start = 0
                    for length in lengths:
                        segments.append((k[start:start+length].detach().cpu().contiguous(),
                                         v[start:start+length].detach().cpu().contiguous()))
                        start += length
                    payload = {'q': q.detach().cpu().contiguous(), 'segments': segments,
                               'original_output': output.detach().cpu().contiguous(),
                               'metadata': dict(info)}
                    path = args.output/f"{active['name']}-L{rt.layer_id}-P{rt.valid_len}-T{q.shape[0]}.pt"
                    torch.save(payload, path)
                    info['capture_path'] = str(path.resolve())
                    info['capture_sha256'] = file_hash(path)
                    info['expected_recent_k'] = torch.cat((segments[2][0],segments[3][0]))[-512:]
                    info['expected_recent_v'] = torch.cat((segments[2][1],segments[3][1]))[-512:]
                    info['expected_sink'] = segments[1]
                return output

            def chunk(q, k, v, device):
                info = {'request': active['name'], 'layer_id': rt.layer_id,
                        'valid_before': rt.valid_len, 'prefill_before': rt.prefill_len,
                        'blocks_before': rt.nblocks_filled,
                        'original_strides': {'q': list(q.stride()),'k': list(k.stride()),'v': list(v.stride())},
                        'original_contiguous': {'q': q.is_contiguous(),'k': k.is_contiguous(),'v': v.is_contiguous()},
                        'status': 'running'}
                pending[rt.layer_id] = info
                output = original_chunk(q, k, v, device)
                if rt.valid_len != info['valid_before']+q.shape[0] or rt.last_prefill_block_ids.tolist() != info['ids']:
                    raise RuntimeError('post-attention state/selection mismatch')
                if 'capture_path' in info:
                    for name in ('k','v'):
                        actual = getattr(rt, 'recent_'+name).detach().cpu()
                        expected = info.pop('expected_recent_'+name)
                        if not torch.equal(actual, expected):
                            raise RuntimeError('recent window changed incorrectly')
                    sk,sv = info.pop('expected_sink')
                    if not torch.equal(rt.sink_k.detach().cpu(),sk) or not torch.equal(rt.sink_v.detach().cpu(),sv):
                        raise RuntimeError('sink changed after a later chunk')
                    record['captures'].append({'path': info['capture_path'],
                                               'sha256': info['capture_sha256'],
                                               'metadata': dict(info), 'post_state_checked': True})
                info.update(status='passed', valid_after=rt.valid_len,
                            prefill_after=rt.prefill_len, blocks_after=rt.nblocks_filled)
                active['row']['layer_steps'].append(dict(info))
                return output

            rt._gpu_select = select
            rt._prefill_attention = attention
            rt.prefill_chunk = chunk

        for rt in layers:
            install(rt)
        for name, text in PROMPTS.items():
            unit = tokenizer.encode(text)
            ids = (unit*((16480+len(unit)-1)//len(unit)))[:16480]
            row = {'name': name, 'status': 'running', 'prompt_ids': ids,
                   'prompt_ids_sha256': hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                   'prompt_unit': text, 'layer_steps': [], 'instrumented_step_wall_ms': []}
            record['requests'].append(row); active.update(name=name,row=row); save()
            llm.sparse_reset()
            if any(rt.valid_len or rt.prefill_len or rt.nblocks_filled for rt in layers):
                raise RuntimeError('request state reset failed')
            row['reset_checked'] = True
            llm.add_request(ids,SamplingParams(max_tokens=1,temperature=0.,ignore_eos=True))
            while not llm.is_finished():
                torch.cuda.synchronize(); start = time.perf_counter()
                output, scheduled = llm.step(); torch.cuda.synchronize()
                row['instrumented_step_wall_ms'].append((time.perf_counter()-start)*1000)
                print('capture',name,'scheduled',scheduled,'valid',layers[0].valid_len,flush=True)
                if scheduled <= 0:
                    raise RuntimeError('capture unexpectedly entered decode')
                save()
            row.update(status='passed', generated_ids=output[0][1],
                       final_valid_lens=[rt.valid_len for rt in layers])
            if len(row['layer_steps']) != 4*36 or set(row['final_valid_lens']) != {16480}:
                raise RuntimeError('unexpected later-chunk coverage or final state')
            save()
        if len(record['captures']) != 12:
            raise RuntimeError('capture count differs from fixed plan')
        after = {n: file_hash(ROOT/n) for n in source_files}
        if after != hashes:
            raise RuntimeError('source changed during capture')
        record.update(status='passed', source_hashes_after=after)
    except Exception as exc:
        record.update(status='failed',failure=str(exc),traceback=traceback.format_exc()); raise
    finally:
        if llm is not None:
            llm.exit()
        save(); print('capture manifest',manifest_path,flush=True)


if __name__ == '__main__':
    run()
