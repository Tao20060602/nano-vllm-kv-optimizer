"""CPU-only regeneration of frozen M23 token IDs from pinned prompt units."""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise RuntimeError('preserve existing output')
    frozen=sha(a.fixture);fixture=json.loads(a.fixture.read_text())
    manifest=a.model/'.cache/nanokv/source-manifest.json'
    if sha(manifest)!=fixture['model_manifest_sha256']:raise RuntimeError('model manifest changed')
    source=ROOT/fixture['source_manifest']
    if sha(source)!=fixture['source_manifest_sha256']:raise RuntimeError('original prompt manifest changed')
    legacy=json.loads(source.read_text());tokenizer=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True)
    source_rows={r['name'].split('-')[0]:r for r in legacy['requests']}
    rows=[]
    for row in fixture['requests']:
        previous=source_rows[row['prompt']]
        if row['prompt_unit']!=previous['prompt_unit']:raise RuntimeError('prompt unit changed')
        unit=tokenizer.encode(row['prompt_unit']);n=row['tokens']
        ids=(unit*((n+len(unit)-1)//len(unit)))[:n]
        if ids!=row['prompt_ids'] or ids[:16480]!=previous['prompt_ids']:raise RuntimeError('regenerated input mismatch')
        digest=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        if digest!=row['prompt_ids_sha256']:raise RuntimeError('regenerated hash mismatch')
        rows.append(dict(name=row['name'],tokens=n,unit_token_count=len(unit),prompt_ids_sha256=digest,
                         original_16480_prefix_equal=True,full_regeneration_equal=True))
    if sha(a.fixture)!=frozen:raise RuntimeError('fixture changed during regeneration')
    record=dict(schema=1,status='passed',scope='CPU tokenizer regeneration, no GPU/model inference',
                model_manifest_sha256=sha(manifest),source_manifest_sha256=sha(source),fixture_sha256=frozen,
                checker_source_sha256=sha(Path(__file__)),requests=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print('fixture regeneration passed:',a.output)


if __name__=='__main__':run()
