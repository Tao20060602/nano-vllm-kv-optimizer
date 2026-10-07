"""Native model audit and fresh-process prefill comparison, with unchanged routing."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import torch
import triton
import flash_attn

from nanovllm import LLM, SamplingParams
from nanovllm.sparse.segmented_prefill_adapter import load_operator
from benchmark_m14_prefill import YARN

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def gpu_owner_check():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    rows=[line for line in output.splitlines() if line.strip()]
    if any(not line.split(',')[0].strip().isdigit() for line in rows):
        raise RuntimeError('GPU process information unavailable; inspect GPU recovery action')
    others=[line for line in rows if int(line.split(',')[0])!=os.getpid()]
    if others:raise RuntimeError('another GPU compute process is active: '+str(others))
    return output.strip()


def telemetry():
    return subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used','--format=csv,noheader'],text=True).strip()


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--operator-root',type=Path,required=True)
    parser.add_argument('--backend',choices=('flash','flash_reuse','operator'),required=True)
    parser.add_argument('--phase',choices=('audit','perf'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError('preserve the existing output')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    manifest=json.loads(args.manifest.read_text())
    if manifest['status']!='passed' or manifest['model_revision']!='1cfa9a7208912126459214e8b04321603b3df60c':
        raise RuntimeError('expected audited pinned M17 manifest')
    model_manifest=args.model/'.cache/nanokv/source-manifest.json'
    if sha(model_manifest)!=manifest['model_manifest_sha256']:
        raise RuntimeError('model manifest mismatch')
    names=sorted([p.relative_to(ROOT).as_posix() for p in (ROOT/'nanovllm').rglob('*.py')])
    names += ['benchmarks/benchmark_segmented_adapter.py','benchmarks/benchmark_m14_prefill.py',
              'docs/NATIVE_SEGMENTED_ADAPTER_PLAN.md']
    hashes={n:sha(ROOT/n) for n in names}
    record=dict(schema=1,status='running',backend=args.backend,phase=args.phase,
                model=str(args.model.resolve()),model_revision=manifest['model_revision'],
                model_manifest_sha256=sha(model_manifest),manifest_sha256=sha(args.manifest),
                source_hashes=hashes,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                runtime=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
                             triton=triton.__version__,fa2=flash_attn.__version__),
                config=dict(seq_len=16480,chunk_size=4096,top_k=32,summary=1,sink=64,recent=512,
                            index_select=True,rope=YARN,threads=8,max_tokens=1,temperature=0),
                scope='synchronized prefill engine step sum, first generated token; excludes import/model load/precompile/warmup/disk/state collection',
                requests=[])
    def save():args.output.write_text(json.dumps(record,indent=2)+'\n')
    save();llm=None;layers=[];active={}
    try:
        record['gpu_owner_before']=gpu_owner_check()
        record['telemetry_before']=telemetry()
        if 'N/A' in record['telemetry_before'] or 'ERR' in record['telemetry_before']:
            raise RuntimeError('GPU telemetry unhealthy; CUDA validation requires recovery')
        save()
        start=time.perf_counter()
        llm=LLM(str(args.model),enforce_eager=True,tensor_parallel_size=1,max_num_seqs=1,
                max_num_batched_tokens=4096,max_model_len=16512,dtype='bfloat16',gpu_memory_utilization=.7,
                enable_sparse_attention=True,use_m12_runtime=True,sparse_selector='query_guided',
                sparse_retrieval_block_size=64,sparse_num_representatives=4,sparse_recent_tokens=512,
                sparse_first_tokens=64,sparse_top_k=32,sparse_prefill_top_k=32,sparse_decode_top_k=32,
                sparse_gather_index_select=True,sparse_prefill_chunk_size=4096,sparse_prefill_query_segments=1,
                sparse_prefill_attention_backend=args.backend,sparse_operator_root=str(args.operator_root),
                rope_scaling_override=YARN)
        torch.cuda.synchronize();record['model_constructor_synchronized_ms']=(time.perf_counter()-start)*1000
        layers=[m.sparse_rt for m in llm.model_runner.model.modules() if getattr(m,'sparse_rt',None) is not None]
        if len(layers)!=36:raise RuntimeError('expected 36 layers')
        adapter=layers[0].cfg.prefill_adapter
        if adapter and any(rt.cfg.prefill_adapter is not adapter for rt in layers):
            raise RuntimeError('scratch is not shared')
        if adapter and adapter.provenance:record['operator_provenance']=adapter.provenance
        if args.backend=='operator':
            torch.cuda.synchronize();start=time.perf_counter()
            adapter.precompile(96,(2048,64,512,96));torch.cuda.synchronize()
            record['precompile_synchronized_ms']=(time.perf_counter()-start)*1000
        else:record['precompile_synchronized_ms']=0
        save()
        if args.phase=='audit':
            if adapter is None:raise RuntimeError('audit requires an experimental adapter')
            module,policy,proof=load_operator(args.operator_root)
            record['audit_operator_provenance']=proof
            from llm_gpu_kernels.attention_segmented_benchmark import _check, _chunked_fp32_reference
            original_attend=adapter.attend
            def attend(invocation):
                actual=original_attend(invocation)
                k=torch.cat(tuple(k for k,_ in invocation.segments),dim=0)
                v=torch.cat(tuple(v for _,v in invocation.segments),dim=0)
                shadow=flash_attn.flash_attn_func(invocation.q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0),
                    softmax_scale=128**-.5,causal=True).squeeze(0)
                info=active['layer']
                info['attention_mode']=invocation.mode
                info['output_vs_same_input_fa2']=_check(actual,shadow,'adapter vs same input FA2')
                if invocation.q.shape[0]<=128:
                    info['output_vs_independent_fp32']=_check(actual,_chunked_fp32_reference(invocation.q,invocation.segments),'adapter vs FP32')
                return actual
            adapter.attend=attend
            def install(rt):
                original=rt.prefill_chunk
                def chunk(q,k,v,device):
                    before=rt.valid_len
                    old_recent_len=min(512,before)
                    old_k=rt.recent_k[:old_recent_len].clone()
                    old_v=rt.recent_v[:old_recent_len].clone()
                    sink_k=rt.sink_k[:min(64,before)].clone()
                    sink_v=rt.sink_v[:min(64,before)].clone()
                    protected=set(rt.protected_blocks)
                    info=dict(layer_id=rt.layer_id,valid_before=before,tokens=q.shape[0])
                    active['layer']=info
                    output=original(q,k,v,device)
                    ids=rt.last_prefill_block_ids.tolist()
                    if set(ids)&protected or len(ids)!=len(set(ids)):
                        raise RuntimeError('selector/protection changed')
                    length=min(512,before+k.shape[0])
                    expected_k=torch.cat((old_k,k))[-length:]
                    expected_v=torch.cat((old_v,v))[-length:]
                    if not torch.equal(rt.recent_k[:length],expected_k) or not torch.equal(rt.recent_v[:length],expected_v):
                        raise RuntimeError('recent state does not match own input window')
                    if not torch.equal(rt.sink_k[:sink_k.shape[0]],sink_k) or not torch.equal(rt.sink_v[:sink_v.shape[0]],sink_v):
                        raise RuntimeError('sink changed')
                    if rt.valid_len!=before+k.shape[0] or rt.prefill_len!=rt.valid_len:
                        raise RuntimeError('logical lengths changed')
                    info.update(selected_ids=ids,protected_before=sorted(protected),state_checked=True,
                                valid_after=rt.valid_len,blocks_after=rt.nblocks_filled)
                    active['row']['layer_audit'].append(info)
                    return output
                rt.prefill_chunk=chunk
            for rt in layers:install(rt)

        def request(source,name):
            row=dict(name=name,status='running',prompt_ids_sha256=source['prompt_ids_sha256'],
                     step_ms=[],scheduled=[],states=[],layer_audit=[])
            active['row']=row
            llm.sparse_reset()
            if any(rt.valid_len or rt.prefill_len or rt.nblocks_filled for rt in layers):
                raise RuntimeError('reset failed')
            row['reset_checked']=True
            llm.add_request(source['prompt_ids'],SamplingParams(max_tokens=1,temperature=0.,ignore_eos=True))
            before_stats=None if adapter is None else adapter.stats()
            while not llm.is_finished():
                torch.cuda.synchronize();start=time.perf_counter()
                output,count=llm.step();torch.cuda.synchronize()
                row['step_ms'].append((time.perf_counter()-start)*1000);row['scheduled'].append(count)
                if count<=0:raise RuntimeError('unexpected decode in first-token benchmark')
                row['states'].append([dict(layer_id=rt.layer_id,valid_len=rt.valid_len,prefill_len=rt.prefill_len,
                    blocks=rt.nblocks_filled,protected=sorted(rt.protected_blocks),
                    selected_ids=None if rt.last_prefill_block_ids is None else rt.last_prefill_block_ids.tolist()) for rt in layers])
            if row['scheduled']!=[4096,4096,4096,4096,96]:raise RuntimeError('chunk schedule changed')
            if any(rt.valid_len!=16480 for rt in layers):raise RuntimeError('final lengths changed')
            row.update(status='passed',generated_ids=output[0][1],prefill_wall_ms=sum(row['step_ms']),
                       first_chunk_ms=row['step_ms'][0],later_main_chunks_ms=sum(row['step_ms'][1:4]),
                       tail_chunk_ms=row['step_ms'][-1],gpu_owner_after=gpu_owner_check(),
                       adapter_before=before_stats,adapter_after=None if adapter is None else adapter.stats())
            return row
        # Complete warm request also allocates CPU history/scratch and JITs all shapes.
        warm=request(manifest['requests'][0],'warmup_archive')
        record['warmup']=warm;save()
        torch.cuda.reset_peak_memory_stats()
        for source in manifest['requests']:
            print('native adapter',args.phase,args.backend,source['name'],flush=True)
            row=request(source,source['name']);record['requests'].append(row);save()
        after={n:sha(ROOT/n) for n in names}
        if hashes!=after or sha(args.manifest)!=record['manifest_sha256']:
            raise RuntimeError('source/manifest changed during run')
        record.update(status='passed',source_hashes_after=after,
                      gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      gpu_allocated_bytes=torch.cuda.memory_allocated(),
                      adapter_stats=None if adapter is None else adapter.stats())
    except Exception as exc:
        record.update(status='failed',failure=str(exc),traceback=traceback.format_exc());raise
    finally:
        if llm is not None:llm.exit()
        try:record['telemetry_after']=telemetry()
        except Exception as exc:record['telemetry_after_error']=str(exc)
        save();print('adapter result',args.output,flush=True)


if __name__=='__main__':run()
