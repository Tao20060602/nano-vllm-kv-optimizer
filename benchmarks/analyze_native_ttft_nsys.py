"""Read only whitelisted Nsight timelines; never export process environment data.

CPU annotations include waits. Correlation identifies submitted work, while
interval overlap shows concurrent work; neither is an exclusive critical path.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('m12.prefill_selector', 'm12.prefill_cpu_gather', 'm12.prefill_h2d_pack',
          'm12.prefill_attention', 'm12.prefill_store_kv', 'm23.store_k_copy',
          'm23.store_v_copy', 'm23.build_representatives')
RANGE_PATTERN = re.compile(r'(' + '|'.join(re.escape(s) for s in STAGES) + r')\.layer(\d+)$')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def merged(intervals):
    result = []
    for a,b in sorted(intervals):
        if b <= a: continue
        if result and a <= result[-1][1]: result[-1] = (result[-1][0], max(b,result[-1][1]))
        else: result.append((a,b))
    return result


def union(intervals):
    return sum(b-a for a,b in merged(intervals))


def overlap(left,right):
    a,b = merged(left),merged(right)
    i=j=total=0
    while i<len(a) and j<len(b):
        total += max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1] <= b[j][1]: i+=1
        else: j+=1
    return total


def stats(events, field):
    groups = defaultdict(list)
    for e in events: groups[str(e[field])].append(e)
    return sorted([dict(name=name,count=len(rows),duration_sum_ms=sum(e['end']-e['start'] for e in rows)/1e6,
                        interval_union_ms=union((e['start'],e['end']) for e in rows)/1e6,
                        bytes=sum(e.get('bytes',0) for e in rows)) for name,rows in groups.items()],
                  key=lambda r:r['duration_sum_ms'], reverse=True)


def analyze(path):
    digest = sha(path)
    con = sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    con.row_factory = sqlite3.Row
    tables = ('NVTX_EVENTS','CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_MEMCPY',
              'CUPTI_ACTIVITY_KIND_KERNEL','ENUM_CUDA_MEMCPY_OPER','ENUM_CUDA_MEM_KIND')
    schema = {t:[r['name'] for r in con.execute('PRAGMA table_info('+t+')')] for t in tables}
    if any(not cols for cols in schema.values()): raise RuntimeError('missing required Nsight tables')
    # A bounded name join reads NVTX labels only; StringIds contains sensitive
    # process metadata and must never be enumerated or exported.
    query = """SELECT n.start,n.end,n.globalTid,coalesce(n.text,s.value) AS name
               FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
               WHERE n.end>n.start AND (coalesce(n.text,s.value) LIKE 'm12.prefill_%'
               OR coalesce(n.text,s.value) LIKE 'm23.%'
               OR coalesce(n.text,s.value)='nanokv.prefill.step')"""
    annotations = [dict(r) for r in con.execute(query)]
    outer = [r for r in annotations if r['name']=='nanokv.prefill.step']
    if len(outer)!=1: raise RuntimeError('expected exactly one captured prefill step')
    window = outer[0]
    annotations = [r for r in annotations if RANGE_PATTERN.fullmatch(r['name'])]
    if any(r['start']<window['start'] or r['end']>window['end'] for r in annotations):
        raise RuntimeError('stage range outside captured step')
    runtime = [dict(r) for r in con.execute("""SELECT r.start,r.end,r.globalTid,r.correlationId,s.value AS name
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
        WHERE s.value LIKE 'cuda%' OR s.value LIKE 'cu%'""")]
    kinds = {r['id']:r['label'] for r in con.execute('SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER')}
    memory = {r['id']:r['label'] for r in con.execute('SELECT id,label FROM ENUM_CUDA_MEM_KIND')}
    copies = [dict(r) for r in con.execute('SELECT start,end,correlationId,globalPid,bytes,copyKind,srcKind,dstKind FROM CUPTI_ACTIVITY_KIND_MEMCPY')]
    for r in copies:
        r['name']=kinds[r['copyKind']];r['source_memory']=memory[r['srcKind']];r['destination_memory']=memory[r['dstKind']]
    kernels = [dict(r) for r in con.execute('''SELECT k.start,k.end,k.correlationId,k.globalPid,s.value AS name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName''')]
    con.close()
    if len({r['correlationId'] for r in runtime})!=len(runtime):
        raise RuntimeError('ambiguous runtime correlation IDs')
    activities=copies+kernels
    by_corr={r['correlationId']:r for r in runtime}
    covered=[a for a in activities if a['correlationId'] in by_corr and
             by_corr[a['correlationId']]['globalTid'] & ~((1<<24)-1)==a['globalPid']]
    if len(covered)!=len(activities): raise RuntimeError('incomplete process-qualified GPU/API correlation')
    stages={}
    for stage in STAGES:
        ranges=[r for r in annotations if r['name'].rsplit('.layer',1)[0]==stage]
        layers=[int(r['name'].rsplit('.layer',1)[1]) for r in ranges]
        if len(ranges)!=36 or set(layers)!=set(range(36)): raise RuntimeError('stage layer coverage incomplete: '+stage)
        intervals=[(r['start'],r['end']) for r in ranges]
        calls=[c for c in runtime if any(c['globalTid']==r['globalTid'] and r['start']<=c['start'] and c['end']<=r['end'] for r in ranges)]
        corr={c['correlationId'] for c in calls}
        linked_copies=[c for c in copies if c['correlationId'] in corr]
        linked_kernels=[k for k in kernels if k['correlationId'] in corr]
        cpu_ns=union(intervals)
        api_ns=overlap(intervals,[(c['start'],c['end']) for c in calls])
        stages[stage]=dict(layer_count=36,cpu_inclusive_sum_ms=sum(r['end']-r['start'] for r in ranges)/1e6,
            cpu_interval_union_ms=cpu_ns/1e6,cuda_host_api_union_ms=api_ns/1e6,
            host_time_outside_contained_cuda_api_ms=(cpu_ns-api_ns)/1e6,
            cuda_host_api=stats(calls,'name'),submitted_gpu_copies=stats(linked_copies,'name'),
            submitted_gpu_kernels=stats(linked_kernels,'name'),
            temporal_gpu_overlap_ms=overlap(intervals,[(a['start'],a['end']) for a in activities])/1e6,
            temporal_flash_kernel_overlap_ms=overlap(intervals,[(k['start'],k['end']) for k in kernels if 'flash' in k['name'].lower()])/1e6,
            temporal_gemm_kernel_overlap_ms=overlap(intervals,[(k['start'],k['end']) for k in kernels if any(s in k['name'].lower() for s in ('gemm','s168','s884'))])/1e6)
    if sha(path)!=digest: raise RuntimeError('SQLite changed while analyzing')
    return dict(path=str(path),sha256=digest,schema=schema,step_cpu_ms=(window['end']-window['start'])/1e6,
        annotation_count=len(annotations),stages=stages,cuda_host_api=stats(runtime,'name'),
        gpu_copies=stats(copies,'name'),gpu_kernels=stats(kernels,'name'),
        gpu_activity_interval_union_ms=union((a['start'],a['end']) for a in activities)/1e6,
        correlation_coverage=dict(gpu_activities=len(activities),matched=len(covered),fraction=1.0,
            method='unique runtime correlationId, process-qualified globalTid/globalPid (24 thread-id bits)'),
        memory_kinds=memory)


def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists(): raise RuntimeError('preserve existing report')
    own_hash=sha(Path(__file__));base=analyze(args.baseline.resolve());candidate=analyze(args.candidate.resolve())
    for arm in (base,candidate):
        for stage in ('m23.store_k_copy','m23.store_v_copy'):
            copies=arm['stages'][stage]['submitted_gpu_copies']
            if len(copies)!=1 or copies[0]['name']!='Device-to-Host' or copies[0]['count']!=36 or copies[0]['bytes']!=36*8*1024**2:
                raise RuntimeError('KV D2H payload/coverage changed')
    comparison={stage:{key:dict(baseline=base['stages'][stage][key],candidate=candidate['stages'][stage][key])
        for key in ('cpu_inclusive_sum_ms','cuda_host_api_union_ms','host_time_outside_contained_cuda_api_ms',
                    'temporal_gpu_overlap_ms','temporal_flash_kernel_overlap_ms','temporal_gemm_kernel_overlap_ms')}
        for stage in STAGES}
    report=dict(schema=1,status='passed',analysis_mode='CPU-only read-only Nsight SQLite, CUDA/NVTX whitelist; no GPU workload',
        source_hashes={name:sha(ROOT/name) for name in ('benchmarks/analyze_native_ttft_nsys.py','benchmarks/profile_native_ttft_nsys.py','benchmarks/benchmark_native_ttft.py','benchmarks/native_ttft_candidate.py')},
        baseline=base,candidate=candidate,comparison=comparison,usable_as_clean_performance=False,
        caveats=['One instrumented window per arm, not an independent performance experiment.',
                 'Inclusive CPU annotations contain waiting for earlier submitted GPU work. CUDA API times also include waits/staging.',
                 'Contained CUDA API union is subtracted only to describe time outside those APIs; residual is not a measured host memcpy or recoverable budget.',
                 'Submitted-work correlation and temporal GPU overlap differ; neither identifies an exclusive critical path. Do not sum overlapping stage/API/GPU times.',
                 'CPU KV temporary host copy removal is established by frozen source diff plus exact audit; Nsight does not trace PyTorch CPU memcpy.',
                 'No hardware counters, CPU instruction samples or thread context switches were captured.',
                 'Sensitive process metadata and unrelated StringIds are not exported. Raw profiler reports remain local ignored files.'])
    if sha(Path(__file__))!=own_hash: raise RuntimeError('parser changed during analysis')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print('Nsight analysis passed:',args.output)


if __name__=='__main__':run()
