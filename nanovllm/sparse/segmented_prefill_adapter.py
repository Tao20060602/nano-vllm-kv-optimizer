"""Opt-in, single-stream inference scratch shared across sequential M12 layers."""
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_operator(root):
    root = Path(root).resolve()
    summary = json.loads((root/'results/m18-summary.json').read_text())
    path = root/summary['record_path']
    if _sha(path) != summary['record_sha256']:
        raise RuntimeError('M18 record hash mismatch')
    record = json.loads(path.read_text())
    if record['status'] != 'passed' or record['source_hashes'] != record['source_hashes_after']:
        raise RuntimeError('M18 incomplete source audit')
    for name, expected in record['source_hashes'].items():
        if _sha(root/name) != expected:
            raise RuntimeError('frozen operator source mismatch: '+name)
    src = root/'src'
    if str(src) not in sys.path:
        sys.path.insert(0,str(src))
    module = importlib.import_module('llm_gpu_kernels.attention_dynamic')
    policy_module = importlib.import_module('llm_gpu_kernels.attention_resource_policy')
    for imported in (module, policy_module):
        if not Path(imported.__file__).resolve().is_relative_to(src):
            raise RuntimeError('operator imported from another repository')
    policy = policy_module.FrozenResourcePolicy.load(root/'results/m16-resource-policy.json')
    provenance = policy.verify_provenance(root)
    return module, policy, dict(root=str(root),record_sha256=summary['record_sha256'],
                               source_files=len(record['source_hashes']),provenance=provenance)


@dataclass(frozen=True)
class PrefillInvocation:
    q: torch.Tensor
    segments: tuple
    mode: str
    packed: tuple | None


class SegmentedPrefillAdapter:
    """No KV state or selection ownership; caller finishes each layer on one stream.

    Scratch can be overwritten only after its consumer is queued on that stream.
    Outputs are consumed before the next layer, not retained across invocations.
    """
    def __init__(self, backend, scale, geometry, operator_root=None):
        if backend not in ('flash_reuse','operator'):
            raise ValueError('expected explicit experimental backend')
        self.backend, self.scale, self.geometry = backend, scale, geometry
        self.stream = torch.cuda.current_stream()
        self.device = self.stream.device
        self.packed_k = self.packed_v = self.selected_k = self.selected_v = None
        self.pack_capacity = self.selected_capacity = 0
        self.workspace = None
        self.workspace_builds = self.direct_calls = self.fa2_calls = 0
        self.selection_cache = {}
        self.module = self.policy = self.compile_cache = None
        self.provenance = None
        if backend == 'operator':
            if geometry != (32,8,128,torch.bfloat16) or scale != 128**-.5:
                raise ValueError('operator experiment requires BF16 Q32/KV8/D128 and default scale')
            self.module,self.policy,self.provenance = load_operator(operator_root)
            self.compile_cache = self.module.AttentionCompileCache(max_entries=16)

    def _stream_check(self, device):
        if device != self.device or torch.cuda.current_stream(device).cuda_stream != self.stream.cuda_stream:
            raise ValueError('shared prefill scratch belongs to one CUDA device/stream')

    def _choice(self,tokens,lengths):
        if self.backend != 'operator':
            return 'fa2',None,None
        key=(tokens,lengths)
        if key not in self.selection_cache:
            if len(self.selection_cache)>=16:
                self.selection_cache.clear()
            self.selection_cache[key]=self.policy.select(tokens,lengths)
        return self.selection_cache[key]

    def precompile(self,tokens,lengths):
        mode,config,_=self._choice(tokens,tuple(lengths))
        if mode=='direct':
            self.compile_cache.get(self.device,tokens,tuple(lengths),config)
        return mode

    def prepare(self,q,stage_k,stage_v,sink,recent,current):
        self._stream_check(q.device)
        lengths=(stage_k.shape[0],sink[0].shape[0],recent[0].shape[0],current[0].shape[0])
        if stage_k.shape!=stage_v.shape or current[0].shape[0]!=q.shape[0]:
            raise ValueError('inconsistent selected/current lengths')
        if not stage_k.is_pinned() or not stage_v.is_pinned():
            raise ValueError('selected CPU staging must be pinned')
        mode,_,_=self._choice(q.shape[0],lengths)
        _,heads,dim,dtype=self.geometry
        if mode=='direct':
            if lengths[0]>self.selected_capacity:
                self.selected_capacity=lengths[0]
                self.selected_k=torch.empty((lengths[0],heads,dim),dtype=dtype,device=q.device)
                self.selected_v=torch.empty_like(self.selected_k)
                self.selected_k.record_stream(self.stream);self.selected_v.record_stream(self.stream)
            # The zero-selected case owns a valid empty view, without a DMA.
            if self.selected_k is None:
                self.selected_k=torch.empty((0,heads,dim),dtype=dtype,device=q.device)
                self.selected_v=torch.empty_like(self.selected_k)
            selected=(self.selected_k[:lengths[0]],self.selected_v[:lengths[0]])
            selected[0].copy_(stage_k,non_blocking=False)
            selected[1].copy_(stage_v,non_blocking=False)
            return PrefillInvocation(q,(selected,sink,recent,current),'direct',None)
        total=sum(lengths)
        if total>self.pack_capacity:
            self.pack_capacity=total
            self.packed_k=torch.empty((total,heads,dim),dtype=dtype,device=q.device)
            self.packed_v=torch.empty_like(self.packed_k)
            self.packed_k.record_stream(self.stream);self.packed_v.record_stream(self.stream)
        pk,pv=self.packed_k[:total],self.packed_v[:total]
        offset=0;segments=[]
        for length,(k,v) in zip(lengths,((stage_k,stage_v),sink,recent,current)):
            dk,dv=pk[offset:offset+length],pv[offset:offset+length]
            dk.copy_(k,non_blocking=False);dv.copy_(v,non_blocking=False)
            segments.append((dk,dv));offset+=length
        return PrefillInvocation(q,tuple(segments),'fa2',(pk,pv))

    def attend(self,invocation):
        self._stream_check(invocation.q.device)
        if invocation.mode=='direct':
            flat=(invocation.q,*(t for pair in invocation.segments for t in pair))
            metadata=self.module._metadata(flat)
            if self.workspace is None or self.workspace.metadata!=metadata:
                self.workspace=self.module.DynamicAttentionWorkspace(
                    invocation.q,invocation.segments,self.policy,self.compile_cache)
                self.workspace_builds+=1
            self.direct_calls+=1
            return self.workspace.run(invocation.q,invocation.segments)
        self.fa2_calls+=1
        from flash_attn import flash_attn_func
        k,v=invocation.packed
        return flash_attn_func(invocation.q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0),
                               softmax_scale=self.scale,causal=True).squeeze(0)

    def stats(self):
        tensors=(self.packed_k,self.packed_v,self.selected_k,self.selected_v)
        scratch=sum(t.numel()*t.element_size() for t in tensors if t is not None)
        direct_bytes=0 if self.workspace is None else (
            self.workspace.output.numel()*self.workspace.output.element_size()+
            self.workspace.conversion_workspace_bytes)
        return dict(direct_calls=self.direct_calls,fa2_calls=self.fa2_calls,
                    workspace_builds=self.workspace_builds,shared_scratch_bytes=scratch,
                    direct_workspace_bytes=direct_bytes,pack_capacity=self.pack_capacity,
                    selected_capacity=self.selected_capacity,
                    cache=None if self.compile_cache is None else self.compile_cache.stats())
