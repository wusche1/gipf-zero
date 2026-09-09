"""Atomically publish a verified local checkpoint; original training file is kept."""
import argparse,datetime,hashlib,io,json,os
from pathlib import Path
import torch
from training.model import load_model,encode
import gipf_engine as ge
ROOT=Path(__file__).resolve().parents[1]

def promote(checkpoint,name,reports):
    raw=Path(checkpoint).read_bytes();source_hash=hashlib.sha256(raw).hexdigest()
    model,info=load_model(io.BytesIO(raw),'cpu');torch.set_num_threads(1)
    with torch.inference_mode():
        logits,value=model(torch.tensor(encode(ge.State())[None]))
        if not torch.isfinite(logits).all() or not torch.isfinite(value).all():raise ValueError('Nonfinite checkpoint')
    for tensor in model.state_dict().values():
        if not torch.isfinite(tensor).all():raise ValueError('Nonfinite weights')
    evaluations=[]
    for report in reports:
        data=json.loads(Path(report).read_text())
        # Promotion evidence must refer to this exact source file, never a moving latest.
        actual=data.get('checkpoint_sha256',(data.get('hashes') or [None])[0])
        if actual!=source_hash:raise ValueError(f'Evaluation checkpoint mismatch: {report}')
        evaluations.append({'report':str(Path(report).relative_to(ROOT)) if Path(report).is_absolute() else str(report),'opponent':data.get('opponent',data.get('champion')),'wins':data['wins'],'losses':data['losses'],'cutoffs':data['cutoffs'],'unfinished':data.get('unfinished',0)})
    data={'name':name,'model':model.state_dict(),'config':info['config'],'games':info.get('games',0),'iteration':info.get('iteration',0),'source_checkpoint_sha256':source_hash,'timestamp':info.get('timestamp')}
    folder=ROOT/'checkpoints';folder.mkdir(exist_ok=True)
    temp=folder/'champion.tmp';torch.save(data,temp);os.replace(temp,folder/'champion.pt')
    metadata={'name':name,'kind':'rl','games_trained':data['games'],'iteration':data['iteration'],'architecture':data['config'],'source_checkpoint_sha256':source_hash,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'evaluations':evaluations,'evaluation_settings':'See linked reports for fixed simulation counts, baseline time budgets, paired opening seeds, and cutoffs.'}
    out=ROOT/'reports/champion.json';tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(metadata,indent=2)+'\n');tmp.replace(out)
    with (ROOT/'reports/promotions.jsonl').open('a') as f:f.write(json.dumps(metadata)+'\n')
    return metadata

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--name',required=True);p.add_argument('--reports',nargs='+',required=True);args=p.parse_args()
    print(json.dumps(promote(args.checkpoint,args.name,args.reports)),flush=True)
