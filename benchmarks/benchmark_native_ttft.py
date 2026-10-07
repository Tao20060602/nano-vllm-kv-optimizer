"""M23 fixed-policy warm TTFT; audit/profile instrumentation has separate phases."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import time
import traceback

import torch
import triton
import flash_attn
from nanovllm import LLM, SamplingParams
from benchmark_segmented_adapter import sha, gpu_owner_check, telemetry
from benchmark_m14_prefill import YARN
from native_ttft_candidate import install_candidate

ROOT = Path(__file__).resolve().parents[1]


def tensor_sha(value):
    value = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(value)).hexdigest()


def sources():
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT/'nanovllm').rglob('*.py')]
    names += ['benchmarks/benchmark_native_ttft.py', 'benchmarks/native_ttft_candidate.py',
              'benchmarks/benchmark_segmented_adapter.py', 'benchmarks/benchmark_m14_prefill.py',
              'docs/NATIVE_PREFILL_TTFT_OPTIMIZATION_PLAN.md']
    return {n: sha(ROOT/n) for n in sorted(names)}


def snapshot(layers, hashes=False):
    rows = []
    for rt in layers:
        row = dict(layer=rt.layer_id, valid_len=rt.valid_len, prefill_len=rt.prefill_len,
                   blocks=rt.nblocks_filled, protected=sorted(rt.protected_blocks),
                   selected_ids=None if rt.last_prefill_block_ids is None else rt.last_prefill_block_ids.tolist())
        if hashes:
            n = rt.nblocks_filled
            s = min(rt.cfg.sink_tokens, rt.valid_len)
            r = min(rt.cfg.recent_tokens, rt.valid_len)
            row['tensor_hashes'] = {name: tensor_sha(tensor) for name,tensor in {
                'k_history':rt.k_cpu[:n], 'v_history':rt.v_cpu[:n],
                'representatives':rt.reps_gpu[:,:n], 'sink_k':rt.sink_k[:s],
                'sink_v':rt.sink_v[:s], 'recent_k':rt.recent_k[:r], 'recent_v':rt.recent_v[:r]}.items()}
        rows.append(row)
    return rows


def run():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--arm', choices=('baseline','direct_store'), required=True)
    p.add_argument('--phase', choices=('audit','perf','profile'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--only-case')
    p.add_argument('--nsys-step', type=int)
    args = p.parse_args()
    if args.output.exists(): raise RuntimeError('preserve the existing output')
    if (args.nsys_step is not None) != (args.phase == 'profile'):
        raise RuntimeError('nsys-step is required only for profile phase')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = dict(schema=1, status='running', experiment='native M23 fixed-policy TTFT',
                  arm=args.arm, phase=args.phase, requests=[],
                  usable_as_clean_performance=args.phase=='perf',
                  scope='encoded single warm request enqueue through first generated token; excludes model load, explicit reset and warm requests',
                  audit_scope='16 greedy tokens, all attention-output hashes, prefill selection/state and first-token CPU KV/reps/window hashes')
    def save(): args.output.write_text(json.dumps(record,indent=2)+'\n')
    save(); llm=None; handles=[]; active={}
    try:
        fixture=json.loads(args.fixture.read_text())
        if fixture['status']!='frozen' or fixture['model_revision']!='1cfa9a7208912126459214e8b04321603b3df60c':
            raise RuntimeError('expected frozen pinned Qwen3-4B fixture')
        rows=fixture['requests']
        expected_cases={f'{prompt}-T{tokens}':tokens for prompt in ('archive','code') for tokens in (16480,32864)}
        if len(rows)!=4 or {r['name'] for r in rows}!=set(expected_cases):
            raise RuntimeError('expected exactly four frozen cases')
        for row in rows:
            if row['tokens']!=expected_cases[row['name']] or len(row['prompt_ids'])!=row['tokens']:
                raise RuntimeError('fixture length contract mismatch')
            if any(type(token) is not int or token<0 for token in row['prompt_ids']):
                raise RuntimeError('invalid fixture token IDs')
        if args.only_case: rows=[r for r in rows if r['name']==args.only_case]
        if not rows: raise RuntimeError('no requested cases')
        if args.phase=='profile' and (not args.only_case or not 0<=args.nsys_step<(rows[0]['tokens']+4095)//4096):
            raise RuntimeError('profile requires one valid case and step before model loading')
        for row in rows:
            if hashlib.sha256(json.dumps(row['prompt_ids'],separators=(',',':')).encode()).hexdigest()!=row['prompt_ids_sha256']:
                raise RuntimeError('prompt IDs hash mismatch')
        model_manifest=args.model/'.cache/nanokv/source-manifest.json'
        if sha(model_manifest)!=fixture['model_manifest_sha256']: raise RuntimeError('model manifest mismatch')
        before=sources()
        record.update(source_hashes=before,fixture_sha256=sha(args.fixture),
            fixture_path=str(args.fixture.resolve()),fixture_construction=fixture['construction'],
            fixture_source_manifest=fixture['source_manifest'],fixture_source_manifest_sha256=fixture['source_manifest_sha256'],
            model_manifest_sha256=sha(model_manifest), model_revision=fixture['model_revision'],
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            runtime=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
                         triton=triton.__version__,flash_attn=flash_attn.__version__),
            config=dict(block=64,representatives=4,top_k=32,query_summaries=1,sink=64,recent=512,
                        chunk=4096,backend='flash',index_select=True,rope=YARN,threads=8,
                        selector_graph=False,static_mask=False,decode_pipeline=False),
            gpu_owner_before=gpu_owner_check(),telemetry_before=telemetry())
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        if torch.cuda.get_device_capability()!=(8,6): raise RuntimeError('expected SM86')
        t=time.perf_counter()
        llm=LLM(str(args.model),enforce_eager=True,tensor_parallel_size=1,max_num_seqs=1,
            max_num_batched_tokens=4096,max_model_len=32928,dtype='bfloat16',gpu_memory_utilization=.7,
            enable_sparse_attention=True,use_m12_runtime=True,sparse_selector='query_guided',
            sparse_retrieval_block_size=64,sparse_num_representatives=4,sparse_recent_tokens=512,
            sparse_first_tokens=64,sparse_top_k=32,sparse_prefill_top_k=32,sparse_decode_top_k=32,
            sparse_gather_index_select=True,sparse_prefill_chunk_size=4096,sparse_prefill_query_segments=1,
            sparse_prefill_attention_backend='flash',rope_scaling_override=YARN)
        torch.cuda.synchronize();record['constructor_ms']=(time.perf_counter()-t)*1000
        modules=[m for m in llm.model_runner.model.modules() if getattr(m,'sparse_rt',None)]
        layers=[m.sparse_rt for m in modules]
        if len(layers)!=36: raise RuntimeError('expected 36 sparse layers')
        record['candidate']=install_candidate(layers,args.arm)
        def request(source, phase):
            llm.sparse_reset()
            if any(rt.valid_len or rt.nblocks_filled for rt in layers): raise RuntimeError('reset failed')
            row=dict(name=source['name'],status='running',tokens=len(source['prompt_ids']),
                     prompt_ids_sha256=source['prompt_ids_sha256'],step_ms=[],scheduled=[],states=[],
                     attention_outputs=[],measurement_instrumented=phase in ('audit','profile'))
            active['row']=row
            if phase!='warm': record['requests'].append(row)
            nsteps=(row['tokens']+4095)//4096
            if phase=='profile' and not 0<=args.nsys_step<nsteps: raise RuntimeError('profile step out of range')
            for rt in layers: rt.record_ids=phase=='audit'
            torch.cuda.synchronize();start=time.perf_counter()
            llm.add_request(source['prompt_ids'],SamplingParams(max_tokens=16 if phase=='audit' else 1,temperature=0.,ignore_eos=True))
            index=0; result=None
            while not llm.is_finished():
                active['step']=index
                t=time.perf_counter()
                capture=phase=='profile' and index==args.nsys_step
                profiler_started=False
                try:
                    if capture:
                        for rt in layers: rt.profile_nsys_stages=True
                        torch.cuda.profiler.start();profiler_started=True
                        with torch.cuda.nvtx.range('nanokv.prefill.step'):
                            result,count=llm.step();torch.cuda.synchronize()
                    else:
                        result,count=llm.step();torch.cuda.synchronize()
                finally:
                    if capture:
                        try:
                            if profiler_started: torch.cuda.profiler.stop()
                        finally:
                            for rt in layers: rt.profile_nsys_stages=False
                elapsed=(time.perf_counter()-t)*1000
                if index==nsteps-1:
                    row['ttft_ms']=(time.perf_counter()-start)*1000
                    if any(rt.valid_len!=row['tokens'] for rt in layers): raise RuntimeError('prefill lengths changed')
                    if phase=='audit': row['first_token_state']=snapshot(layers,hashes=True)
                row['step_ms'].append(elapsed);row['scheduled'].append(count)
                if phase=='audit' and count>0: row['states'].append(snapshot(layers))
                index+=1
            expected=[4096]*(row['tokens']//4096)+([row['tokens']%4096] if row['tokens']%4096 else [])
            if row['scheduled'][:nsteps]!=expected: raise RuntimeError('chunk schedule mismatch')
            if phase!='audit' and len(row['scheduled'])!=nsteps: raise RuntimeError('unexpected decode in TTFT timing')
            row.update(status='passed',generated_ids=result[0][1],prefill_step_sum_ms=sum(row['step_ms'][:nsteps]),
                       tail_ms=row['step_ms'][nsteps-1],main_ms=sum(row['step_ms'][:nsteps-1]),
                       decode_selections=[rt.ids_history for rt in layers] if phase=='audit' else [])
            if phase in ('audit','profile'): row['final_logical_state']=snapshot(layers)
            if phase=='audit':
                expected_outputs={(step,rt.layer_id) for step in range(nsteps+15) for rt in layers}
                outputs=[(item['step'],item['layer']) for item in row['attention_outputs']]
                if len(outputs)!=len(expected_outputs) or set(outputs)!=expected_outputs:
                    raise RuntimeError('attention output audit coverage incomplete or duplicated')
                if len(row['scheduled'])!=nsteps+15 or any(len(rt.ids_history)!=15 for rt in layers):
                    raise RuntimeError('decode audit coverage incomplete')
            if len(row['generated_ids'])!=(16 if phase=='audit' else 1): raise RuntimeError('generation cap mismatch')
            return row
        warm=request(max(rows,key=lambda r:len(r['prompt_ids'])),'warm')
        record['warmup']=warm;save()
        if args.phase=='audit':
            def hook(module, inputs, output):
                active['row']['attention_outputs'].append(dict(step=active['step'],layer=module.sparse_rt.layer_id,
                    shape=list(output.shape),sha256=tensor_sha(output),k_contiguous=inputs[1].is_contiguous(),
                    v_contiguous=inputs[2].is_contiguous()))
            handles=[m.register_forward_hook(hook) for m in modules]
        torch.cuda.reset_peak_memory_stats()
        for source in rows:
            gpu_owner_check();print('M23',args.phase,args.arm,source['name'],flush=True)
            request(source,args.phase);save()
        after=sources()
        if before!=after or sha(args.fixture)!=record['fixture_sha256'] or sha(model_manifest)!=record['model_manifest_sha256']:
            raise RuntimeError('source/fixture/model manifest changed during run')
        record.update(status='passed',source_hashes_after=after,telemetry_after=telemetry(),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception as exc:
        record.update(status='failed',failure=str(exc),traceback=traceback.format_exc());raise
    finally:
        for handle in handles: handle.remove()
        if llm is not None: llm.exit()
        save();print('M23 record',args.output,flush=True)


if __name__=='__main__': run()
