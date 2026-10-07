"""Nsight-only store subranges over the M23 driver; never a clean performance run."""
import hashlib
import inspect
from pathlib import Path
import sys
import textwrap
from types import MethodType

import benchmark_native_ttft as driver
from native_ttft_candidate import _REPLACEMENTS, EXPECTED_ORIGINAL_FN_SHA256
from nanovllm.sparse import m12_runtime


def run():
    if '--phase' not in sys.argv or sys.argv[sys.argv.index('--phase')+1]!='profile':
        raise RuntimeError('this wrapper requires --phase profile')
    wrapper_sha=driver.sha(Path(__file__))
    original_install=driver.install_candidate
    def install(layers,arm):
        proof=original_install(layers,arm)
        source=textwrap.dedent(inspect.getsource(m12_runtime.M12LayerRuntime._store_kv))
        if hashlib.sha256(source.encode()).hexdigest()!=EXPECTED_ORIGINAL_FN_SHA256:
            raise RuntimeError('profile baseline function changed')
        if arm=='direct_store':
            for old,new in _REPLACEMENTS:
                if source.count(old)!=1: raise RuntimeError('profile copy replacement mismatch')
                source=source.replace(old,new,1)
        statements=[(new if arm=='direct_store' else old,'store_k_copy' if index==0 else 'store_v_copy')
                    for index,(old,new) in enumerate(_REPLACEMENTS)]
        statements.append(('    self._build_reps_gpu(k_blocks, start_block, nblocks)','build_representatives'))
        for statement,label in statements:
            if source.count(statement)!=1: raise RuntimeError('profile subrange statement mismatch')
            source=source.replace(statement,
                f'    with torch.cuda.nvtx.range(f"m23.{label}.layer{{self.layer_id}}"):\n    '+statement,1)
        local={}
        exec(compile(source,str(Path(__file__)),'exec'),vars(m12_runtime),local)
        for rt in layers: rt._store_kv=MethodType(local['_store_kv'],rt)
        proof.update(profile_wrapper_sha256=wrapper_sha,profile_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
                     profiling_subranges=[label for _,label in statements],instrumented=True)
        return proof
    driver.install_candidate=install
    try:
        driver.run()
        if driver.sha(Path(__file__))!=wrapper_sha: raise RuntimeError('profile wrapper changed during capture')
    finally:
        driver.install_candidate=original_install


if __name__=='__main__': run()
