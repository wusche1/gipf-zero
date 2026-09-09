"""Compare native PUCT selection with its NumPy fallback on fresh identical roots."""
import argparse,hashlib,json,random,time,types
from pathlib import Path
import numpy as np
import torch
import gipf_engine as ge
from training.model import load_model
import training.search as search

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--roots',type=int,default=128);p.add_argument('--simulations',type=int,default=128);p.add_argument('--repeats',type=int,default=3);p.add_argument('--device',default='cuda');p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2);model,_=load_model(a.checkpoint,a.device);rng=random.Random(1981)
    states=[]
    for _ in range(a.roots):
        state=ge.State()
        while state.ply<8 or state.phase=='capture':state.apply(rng.choice(state.legal_actions()))
        states.append(state)
    search.BatchedMCTS(model,a.device).search([search.Node(s.clone()) for s in states[:16]],8)
    results={'numpy':[],'native':[]};policy_sets={}
    try:
        for _ in range(a.repeats):
            for backend in ['numpy','native']:
                search.ge=types.SimpleNamespace(encode_batch=ge.encode_batch) if backend=='numpy' else ge
                roots=[search.Node(s.clone()) for s in states];start=time.perf_counter()
                policies=search.BatchedMCTS(model,a.device).search(roots,a.simulations)
                results[backend].append(time.perf_counter()-start);policy_sets[backend]=policies
        assert all(np.array_equal(x,y) for x,y in zip(policy_sets['numpy'],policy_sets['native']))
    finally:search.ge=ge
    result={'config':vars(a),'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),'seconds':results,'median_seconds':{k:float(np.median(v)) for k,v in results.items()},'identical_policies':True,'torch_version':torch.__version__,'device_name':torch.cuda.get_device_name() if a.device.startswith('cuda') else 'CPU'}
    result['native_speedup']=result['median_seconds']['numpy']/result['median_seconds']['native']
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
