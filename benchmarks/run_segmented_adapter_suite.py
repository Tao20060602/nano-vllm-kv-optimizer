"""Sequential fresh-process native adapter suite; no automatic environment changes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
ORDERS=(('flash','flash_reuse','operator'),('operator','flash_reuse','flash'),('flash_reuse','operator','flash'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def model_audit(model):
    manifest=json.loads((model/'.cache/nanokv/source-manifest.json').read_text())
    if manifest['revision']!='1cfa9a7208912126459214e8b04321603b3df60c':raise RuntimeError('wrong model revision')
    rows=[]
    for name,metadata in manifest['files'].items():
        path=model/name
        digest=sha(path)
        blob=None
        if metadata.get('sha256'):
            matches=digest==metadata['sha256']
        else:
            data=path.read_bytes()
            blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            matches=blob==metadata['blob_id']
        if not matches:raise RuntimeError('model file mismatch: '+name)
        rows.append(dict(name=name,sha256=digest,git_blob_sha1=blob,matches_manifest=True,size=path.stat().st_size))
    return dict(status='passed',revision=manifest['revision'],model_path=str(model.resolve()),files=rows)


def stats(values):
    return dict(geomean=math.exp(sum(map(math.log,values))/len(values)),minimum=min(values),maximum=max(values),n=len(values))


def summarize(jobs):
    pairs=[]
    for group in range(3):
        arms={j['backend']:json.loads(Path(j['path']).read_text()) for j in jobs if j['group']==group}
        for name in ('archive','code'):
            rows={arm:next(r for r in d['requests'] if r['name']==name) for arm,d in arms.items()}
            base=rows['flash_reuse']
            original=rows['flash']
            experiment=rows['operator']
            logical=True;ordered_differences=set_differences=0;compared=0
            for ref_state,actual_state in zip(base['states'],experiment['states']):
                for ref,actual in zip(ref_state,actual_state):
                    keys=('layer_id','valid_len','prefill_len','blocks','protected')
                    logical=logical and all(ref[k]==actual[k] for k in keys)
                    if ref['selected_ids'] is not None:
                        compared+=1
                        ordered_differences+=ref['selected_ids']!=actual['selected_ids']
                        set_differences+=set(ref['selected_ids'])!=set(actual['selected_ids'])
            original_identical_states=original['states']==base['states']
            pairs.append(dict(group=group,prompt=name,
                operator_vs_reuse_prefill=experiment['prefill_wall_ms']/base['prefill_wall_ms'],
                operator_vs_reuse_tail=experiment['tail_chunk_ms']/base['tail_chunk_ms'],
                reuse_vs_flash_prefill=base['prefill_wall_ms']/original['prefill_wall_ms'],
                operator_vs_flash_prefill=experiment['prefill_wall_ms']/original['prefill_wall_ms'],
                original_generated_ids=original['generated_ids'],reuse_generated_ids=base['generated_ids'],
                operator_generated_ids=experiment['generated_ids'],
                greedy_matches_reuse=experiment['generated_ids']==base['generated_ids'],
                greedy_reuse_matches_original=base['generated_ids']==original['generated_ids'],
                logical_state_matches_reuse=logical,compared_later_layer_selections=compared,
                ordered_selection_differences=ordered_differences,selection_set_differences=set_differences,
                reuse_matches_original_states=original_identical_states))
    fields=('operator_vs_reuse_prefill','operator_vs_reuse_tail','reuse_vs_flash_prefill','operator_vs_flash_prefill')
    return dict(pairs=pairs,ratios={f:stats([p[f] for p in pairs]) for f in fields},
                per_prompt={name:{f:stats([p[f] for p in pairs if p['prompt']==name]) for f in fields} for name in ('archive','code')},
                all_greedy_matched=all(p['greedy_matches_reuse'] and p['greedy_reuse_matches_original'] for p in pairs),
                all_logical_states_matched=all(p['logical_state_matches_reuse'] and p['reuse_matches_original_states'] for p in pairs),
                ordered_selection_differences=sum(p['ordered_selection_differences'] for p in pairs),
                selection_set_differences=sum(p['selection_set_differences'] for p in pairs))


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--operator-root',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True,help='completed operator model audit')
    parser.add_argument('--output',type=Path,required=True,help='new directory, never overwritten')
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError('suite directory already exists')
    args.output.mkdir(parents=True)
    audit=json.loads(args.audit.read_text())
    if audit['status']!='passed' or audit['phase']!='audit' or audit['backend']!='operator':raise RuntimeError('completed operator audit required')
    for name,expected in audit['source_hashes'].items():
        if sha(ROOT/name)!=expected:raise RuntimeError('audited native source changed: '+name)
    names=(*audit['source_hashes'],'benchmarks/run_segmented_adapter_suite.py')
    hashes={n:sha(ROOT/n) for n in names}
    path=args.output/'suite.json'
    record=dict(status='running',orders=ORDERS,source_hashes=hashes,audit_path=str(args.audit.resolve()),
                audit_sha256=sha(args.audit),manifest_path=str(args.manifest.resolve()),manifest_sha256=sha(args.manifest),
                jobs=[],scope='three fresh-process groups, two measured prompts per process after one warm request; first generated token only')
    def save():path.write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        record['model_audit']=model_audit(args.model);save()
        env=os.environ.copy()
        for name in ('CUDA_HOME','CUDA_PATH','CUDACXX','LD_LIBRARY_PATH'):env.pop(name,None)
        env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
        for group,order in enumerate(ORDERS):
            for backend in order:
                output=args.output/f'group{group}-{backend}.json'
                log=args.output/f'group{group}-{backend}.log'
                job=dict(group=group,backend=backend,path=str(output.resolve()),log=str(log.resolve()),status='running')
                record['jobs'].append(job);save();print('fresh process',group,backend,flush=True)
                command=[sys.executable,str(ROOT/'benchmarks/benchmark_segmented_adapter.py'),
                         '--model',str(args.model),'--manifest',str(args.manifest),
                         '--operator-root',str(args.operator_root),'--backend',backend,'--phase','perf','--output',str(output)]
                start=time.perf_counter()
                with log.open('w') as stream:
                    process=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                job['process_wall_s']=time.perf_counter()-start;job['returncode']=process.returncode
                if process.returncode!=0:
                    job['status']='failed';save();raise RuntimeError('worker failed; preserved log: '+str(log))
                child=json.loads(output.read_text())
                if child['status']!='passed' or child['source_hashes']!=audit['source_hashes']:
                    raise RuntimeError('worker result/source audit mismatch')
                job.update(status='passed',result_sha256=sha(output));save()
        after={n:sha(ROOT/n) for n in names}
        if hashes!=after or sha(args.audit)!=record['audit_sha256'] or sha(args.manifest)!=record['manifest_sha256']:
            raise RuntimeError('source/audit/manifest changed during suite')
        record.update(status='passed',source_hashes_after=after,summary=summarize(record['jobs']))
    except Exception as exc:
        record.update(status='failed',failure=str(exc),traceback=traceback.format_exc());raise
    finally:
        save();print('suite result',path,flush=True)


if __name__=='__main__':run()
