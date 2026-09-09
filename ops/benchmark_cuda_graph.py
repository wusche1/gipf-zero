import argparse,json,time,hashlib,statistics
from pathlib import Path
import numpy as np,torch
import gipf_engine as ge
from training.model import load_model
from training.search import Node,BatchedMCTS
from training.evaluate import settle_opening
import random
parser=argparse.ArgumentParser();parser.add_argument('checkpoint');parser.add_argument('--output',required=True);args=parser.parse_args()
torch.set_num_threads(2)
path=Path(args.checkpoint);model,_=load_model(path,'cuda')
states=[settle_opening(ge.State(),8,random.Random(1981+i)) for i in range(128)]
searches={'eager':BatchedMCTS(model,'cuda'),'cuda_graph':BatchedMCTS(model,'cuda',cuda_graph_batch=128)}
for search in searches.values():search.search([Node(s.clone()) for s in states[:16]],8)
values={k:[] for k in searches};policies={}
for _ in range(3):
 for name,search in searches.items():
  roots=[Node(s.clone()) for s in states];start=time.perf_counter();policies[name]=search.search(roots,128);values[name].append(time.perf_counter()-start)
result={'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'roots':128,'simulations':128,'seconds':values,'median_seconds':{k:statistics.median(v) for k,v in values.items()},'argmax_agreement':sum(int(np.argmax(x))==int(np.argmax(y)) for x,y in zip(policies['eager'],policies['cuda_graph']))/128,'exact_policy_agreement':sum(np.array_equal(x,y) for x,y in zip(policies['eager'],policies['cuda_graph']))/128,'graph_active':searches['cuda_graph'].inference.graph is not None}
result['speedup']=result['median_seconds']['eager']/result['median_seconds']['cuda_graph']
Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
