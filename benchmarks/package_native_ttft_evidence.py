"""Pack CPU-only M23 source/records and verify archive hashes and paired ratios.

Raw Nsight SQLite/nsys-rep files are deliberately excluded: they can contain
process environment metadata. Model weights and virtual environments are also
excluded. Archive verification does not rehash a live model or measure a GPU.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import check_native_ttft_evidence as audit

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'benchmarks/results/native_ttft'


def digest(value):return hashlib.sha256(value).hexdigest()


def make(suite_path,output):
    if output.exists():raise RuntimeError('preserve existing archive')
    suite=json.loads(suite_path.read_text())
    if suite['status']!='passed':raise RuntimeError('suite incomplete')
    files=[suite_path]
    for job in suite['jobs']:files += [Path(job['worker_path']),Path(job['log_path'])]
    files += [RESULTS/name for name in ('fixture.json','model-audit.json','fixture-regeneration-check.json',
              'final-evidence-check.json','final-evidence-check-v2.json','nsys-analysis.json','nsys-capture-manifest.json',
              'pilot-audit-comparison.json','pilot-baseline-audit.json','pilot-direct-store-audit.json','trace-analysis.json')]
    fixture=json.loads((RESULTS/'fixture.json').read_text())
    files += [ROOT/fixture['source_manifest']]
    files += [ROOT/name for name in suite['source_hashes_before']]
    files += [ROOT/'benchmarks'/name for name in ('check_native_ttft_evidence.py','check_native_ttft_fixture.py',
              'analyze_native_ttft_nsys.py','analyze_native_prefill_trace.py','plot_native_ttft.py','package_native_ttft_evidence.py')]
    files += [ROOT/'docs'/name for name in ('NATIVE_TTFT_DIRECT_STORE_REPORT.md','NATIVE_NSIGHT_GUIDE.md','NATIVE_RETRIEVAL_TRADEOFFS.md')]
    files += [RESULTS/'ttft.png']
    paths=sorted(set(p.resolve() for p in files))
    content={}
    for path in paths:
        name=path.relative_to(ROOT).as_posix()
        if path.suffix not in ('.json','.py','.md','.log','.png'):raise RuntimeError('unexpected artifact type')
        content[name]=path.read_bytes()
    manifest=dict(schema=1,scope='CPU source and records; excludes raw Nsight/process metadata/model weights',
                  suite=suite_path.resolve().relative_to(ROOT).as_posix(),fixture='benchmarks/results/native_ttft/fixture.json',
                  files={name:dict(sha256=digest(value),bytes=len(value)) for name,value in content.items()})
    content['m23-archive-manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream,gzip.GzipFile(fileobj=stream,mode='wb',filename='',mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w') as archive:
        for name,value in sorted(content.items()):
            info=tarfile.TarInfo(name);info.size=len(value);info.mode=0o644;info.mtime=0
            archive.addfile(info,io.BytesIO(value))
    print('CPU evidence archive:',output,'SHA256',digest(output.read_bytes()))


def verify(path,output):
    if output.exists():raise RuntimeError('preserve existing verification')
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers()
        if any(not m.isfile() or m.name.startswith('/') or '..' in Path(m.name).parts for m in members):
            raise RuntimeError('unsafe archive member')
        if len({m.name for m in members})!=len(members):raise RuntimeError('duplicate archive member')
        content={m.name:archive.extractfile(m).read() for m in members}
    manifest=json.loads(content.pop('m23-archive-manifest.json'))
    if set(content)!=set(manifest['files']):raise RuntimeError('archive inventory mismatch')
    for name,value in content.items():
        if manifest['files'][name]!=dict(sha256=digest(value),bytes=len(value)):raise RuntimeError('archive hash mismatch: '+name)
    suite=json.loads(content[manifest['suite']]);fixture=json.loads(content[manifest['fixture']])
    if suite['status']!='passed' or digest(content[manifest['fixture']])!=suite['fixture_sha256']:raise RuntimeError('suite/fixture mismatch')
    workers={}
    for job in suite['jobs']:
        worker_name=str(Path(manifest['suite']).parent/Path(job['worker_path']).name)
        value=content[worker_name]
        if digest(value)!=job['worker_sha256']:raise RuntimeError('worker hash mismatch')
        workers[job['name']]=json.loads(value)
    if {r['name'] for r in fixture['requests']}!={f'{family}-T{n}' for family in ('archive','code') for n in (16480,32864)}:
        raise RuntimeError('fixture matrix mismatch')
    rows={r['name']:r for r in fixture['requests']}
    for r in rows.values():
        if len(r['prompt_ids'])!=r['tokens'] or audit.canonical_prompt_sha256(r['prompt_ids'])!=r['prompt_ids_sha256']:
            raise RuntimeError('fixture identity mismatch')
    exact=audit.compare_audits(workers['audit-baseline'],workers['audit-direct_store'],rows)
    performance=audit.performance_summary(workers,rows)
    if performance['summary']!=suite['performance_summary']:raise RuntimeError('archive recomputation differs')
    for name,expected in suite['source_hashes_before'].items():
        if digest(content[name])!=expected:raise RuntimeError('archived source mismatch: '+name)
    result=dict(schema=1,status='passed',archive_sha256=digest(path.read_bytes()),files_verified=len(content),
                verifier_source_sha256=digest(Path(__file__).read_bytes()),math_auditor_source_sha256=digest(Path(audit.__file__).read_bytes()),
                exact_audit_passed=exact['passed'],performance_summary=performance['summary'],
                scope='offline archive integrity, exact recorded audit and ratio recomputation; no live model/host/GPU validation')
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('archive verification passed:',output)


def run():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('make');a.add_argument('--suite',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('verify');a.add_argument('--archive',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.mode=='make':make(args.suite,args.output)
    else:verify(args.archive,args.output)


if __name__=='__main__':run()
