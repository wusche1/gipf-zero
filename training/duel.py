"""Equal-simulation neural checkpoint matches on paired seeded openings."""
import argparse,hashlib,io,json,random,time
from pathlib import Path
import numpy as np
import torch
import gipf_engine as ge
from baselines.evaluate import wilson_interval
from .model import load_model
from .search import Node,BatchedMCTS
from .evaluate import settle_opening,is_insertion_cutoff

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--champion',required=True)
    p.add_argument('--games',type=int,default=80);p.add_argument('--batch',type=int,default=64);p.add_argument('--simulations',type=int,default=128)
    p.add_argument('--seed',type=int,default=84113);p.add_argument('--max-ply',type=int,default=300);p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--output',required=True);args=p.parse_args();torch.set_num_threads(2)
    models=[];hashes=[];metadata=[]
    for name in (args.candidate,args.champion):
        data=Path(name).read_bytes();hashes.append(hashlib.sha256(data).hexdigest())
        model,info=load_model(io.BytesIO(data),'cuda');models.append(model);metadata.append({'games':info.get('games'),'iteration':info.get('iteration'),'config':info.get('config')})
    search=[BatchedMCTS(m,'cuda') for m in models]
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() or out.with_suffix('.jsonl').exists():raise ValueError('Use a new output path for each duel')
    start=time.monotonic();deadline=start+args.seconds;records=[];active=[];next_id=0
    while len(records)<args.games and time.monotonic()<deadline:
        while len(active)<args.batch and next_id<args.games:
            s=ge.State();settle_opening(s,4,random.Random(args.seed+next_id//2))
            active.append({'id':next_id,'state':s,'candidate_color':1 if next_id%2==0 else -1,'decisions':0});next_id+=1
        groups=[[],[]]
        for game in active:
            who=0 if game['state'].current_player==game['candidate_color'] else 1
            groups[who].append(game)
        for who,games in enumerate(groups):
            if not games:continue
            roots=[Node(g['state'].clone()) for g in games]
            policies=search[who].search(roots,args.simulations,deadline=deadline)
            for game,root,pi in zip(games,roots,policies):
                game['state'].apply(int(root.actions[int(np.argmax(pi))]));game['decisions']+=1
        remaining=[]
        for game in active:
            s=game['state'];cutoff=is_insertion_cutoff(s,args.max_ply) or game['decisions']>=2000
            if s.winner or cutoff:
                record={'game':game['id'],'candidate_color':game['candidate_color'],'winner':s.winner,'outcome':'win' if s.winner==game['candidate_color'] else 'loss' if s.winner else 'cutoff','ply':s.ply}
                records.append(record)
                with out.with_suffix('.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps({'finished':len(records),'last':record,'elapsed':round(time.monotonic()-start,1)}),flush=True)
            else:remaining.append(game)
        active=remaining
    wins=sum(x['outcome']=='win' for x in records);losses=sum(x['outcome']=='loss' for x in records);cutoffs=len(records)-wins-losses
    result={'candidate':args.candidate,'champion':args.champion,'hashes':hashes,'models':metadata,'wins':wins,'losses':losses,'cutoffs':cutoffs,'unfinished':args.games-len(records),'win_rate_decisive':wins/max(1,wins+losses),'wilson_95_decisive':wilson_interval(wins,wins+losses) if wins+losses else [0,1],'elapsed_seconds':time.monotonic()-start,'config':vars(args),'records':records}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('records','config')}),flush=True)
if __name__=='__main__':main()
