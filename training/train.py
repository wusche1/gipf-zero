"""Bounded, checkpointed AlphaZero-style self-play from random weights."""
import argparse,json,time,os,signal,random
from pathlib import Path
from collections import deque
from dataclasses import asdict
import numpy as np
import torch
import gipf_engine as ge
from .model import PolicyValue,ModelConfig,encode,augment,ACTIONS,load_model
from .search import Node,BatchedMCTS

STOP=False
def stop(*_):
    global STOP;STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

def atomic_save(data,path):
    temp=path.with_suffix('.tmp');torch.save(data,temp);os.replace(temp,path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True);p.add_argument('--kind',default='resnet',choices=['resnet','mlp'])
    p.add_argument('--head',default='flat',choices=['flat','factorized']);p.add_argument('--width',type=int,default=48);p.add_argument('--blocks',type=int,default=3)
    p.add_argument('--seconds',type=int,default=600);p.add_argument('--deadline',type=float,default=0)
    p.add_argument('--games',type=int,default=64);p.add_argument('--games-per-iteration',type=int,default=32)
    p.add_argument('--simulations',type=int,default=64);p.add_argument('--updates',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=256);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--max-ply',type=int,default=240);p.add_argument('--replay-size',type=int,default=100000)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--resume');p.add_argument('--device',default='cuda')
    args=p.parse_args();out=Path(args.run);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    if args.device.startswith('cuda'):torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    rng=np.random.default_rng(args.seed)
    config=ModelConfig(args.kind,args.width,args.blocks,args.head)
    model=PolicyValue(config).to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    iteration=updates=finished=decisions=cutoffs=0;replay=deque(maxlen=args.replay_size);resume_data=None
    if args.resume:
        model,data=load_model(args.resume,args.device);config=model.config
        resume_data=data
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
        if 'optimizer' in data:optimizer.load_state_dict(data['optimizer'])
        for g in optimizer.param_groups:g['lr']=args.lr
        iteration=data.get('iteration',0);updates=data.get('updates',0);finished=data.get('games',0)
        decisions=data.get('decisions',0);cutoffs=data.get('cutoffs',0)
        if 'torch_rng' in data:torch.set_rng_state(data['torch_rng'])
        if 'numpy_rng' in data:rng.bit_generator.state=data['numpy_rng']
        if 'python_rng' in data:random.setstate(data['python_rng'])
        rp=Path(args.resume).parent/'replay.pt'
        if rp.exists():replay.extend(torch.load(rp,map_location='cpu',weights_only=False))
    (out/'config.json').write_text(json.dumps(vars(args),indent=2)+'\n')
    start=time.monotonic();end=start+args.seconds
    if args.deadline:end=min(end,start+args.deadline-time.time())
    search=BatchedMCTS(model,args.device,seed=args.seed)
    if resume_data is not None and 'search_rng' in resume_data:search.rng.bit_generator.state=resume_data['search_rng']
    roots=[Node(ge.State()) for _ in range(args.games)];histories=[[] for _ in roots]
    last_heartbeat=0;last_checkpoint=start;last_replay_save=0;session_games=0
    def log(event,**kw):
        record={'event':event,'time':time.time(),'elapsed':round(time.monotonic()-start,2),'iteration':iteration,'games':finished,'session_games':session_games,'decisions':decisions,'updates':updates,'replay':len(replay),'cutoffs':cutoffs,**kw}
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        (out/'heartbeat.json').write_text(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    def checkpoint(force_replay=False):
        nonlocal last_replay_save
        data={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':asdict(config),'iteration':iteration,'updates':updates,'games':finished,'decisions':decisions,'cutoffs':cutoffs,'timestamp':time.time(),'args':vars(args),'torch_rng':torch.get_rng_state(),'numpy_rng':rng.bit_generator.state,'python_rng':random.getstate(),'search_rng':search.rng.bit_generator.state}
        atomic_save(data,out/'latest.pt')
        # Replay saved too: a restarted run does not silently discard experience.
        if force_replay or time.monotonic()-last_replay_save>=120:
            atomic_save(list(replay),out/'replay.pt');last_replay_save=time.monotonic()
        if iteration%100==0:
            atomic_save({k:v for k,v in data.items() if k!='optimizer'},out/f'checkpoint-{iteration:04d}.pt')
            snapshots=sorted(out.glob('checkpoint-*.pt'))
            for old in snapshots[1:-6]:old.unlink()
    model.eval();checkpoint();log('start',parameters=sum(x.numel() for x in model.parameters()),device=args.device)
    last_finished=finished
    try:
        while time.monotonic()<end and not STOP:
            policies=search.search(roots,args.simulations,noise=True,deadline=end)
            for i,(root,pi) in enumerate(zip(roots,policies)):
                if root.state.ply<16:
                    selected=int(rng.choice(len(pi),p=pi))
                else:
                    # Low but nonzero temperature prevents deterministic self-play collapse.
                    weights=np.power(pi,4);weights/=weights.sum()
                    selected=int(rng.choice(len(pi),p=weights))
                histories[i].append((encode(root.state).astype(np.float16),root.actions.astype(np.int16),pi.astype(np.float16),root.actor))
                child=root.children.get(selected)
                if child is None:
                    state=root.state.clone();state.apply(int(root.actions[selected]));child=Node(state)
                roots[i]=child;decisions+=1
                terminal=child.state.winner
                # Do not label a position zero merely because the final allowed
                # insertion has just created a mandatory capture sequence.
                cutoff=(child.state.ply>=args.max_ply and child.state.phase!='capture') or len(histories[i])>=2000
                if terminal or cutoff:
                    finished+=1;session_games+=1;cutoffs+=int(not terminal)
                    for x,actions,policy,actor in histories[i]:replay.append((x,actions,policy,float(terminal*actor)))
                    roots[i]=Node(ge.State());histories[i]=[]
            now=time.monotonic()
            if now-last_heartbeat>=15:
                log('selfplay',evaluations=search.evaluations,decisions_per_second=round(decisions/max(.01,now-start),2));last_heartbeat=now
            if finished-last_finished>=args.games_per_iteration and len(replay)>=args.batch_size:
                model.train();losses=[];gradnorms=[]
                # Indexable snapshot; avoid rebuilding it for every SGD step.
                samples=list(replay)
                for _ in range(args.updates):
                    if STOP or time.monotonic()>=end:break
                    batch=[samples[int(j)] for j in rng.integers(len(samples),size=args.batch_size)]
                    features=[];targets=np.zeros((len(batch),ACTIONS),np.float32);legal=np.zeros_like(targets,dtype=bool);values=[]
                    for k,(x,a,pi,z) in enumerate(batch):
                        x,a=augment(x,a,rng);features.append(x);targets[k,a]=pi;legal[k,a]=True;values.append(z)
                    x=torch.from_numpy(np.stack(features)).float().to(args.device)
                    target=torch.from_numpy(targets).to(args.device);mask=torch.from_numpy(legal).to(args.device)
                    z=torch.tensor(values,device=args.device)
                    logits,value=model(x);logits=logits.masked_fill(~mask,-1e9)
                    ploss=-(target*torch.log_softmax(logits,dim=1)).sum(1).mean();vloss=(value-z).square().mean()
                    loss=ploss+vloss
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                    if not torch.isfinite(gn):raise RuntimeError('Nonfinite gradient')
                    optimizer.step();updates+=1;losses.append((ploss.item(),vloss.item()));gradnorms.append(float(gn))
                model.eval();iteration+=1;last_finished=finished
                # Stale priors/values from old weights must not persist across updates.
                roots=[Node(r.state) for r in roots]
                means=np.mean(losses,axis=0) if losses else [0,0]
                log('train',policy_loss=float(means[0]),value_loss=float(means[1]),gradient_norm=float(np.mean(gradnorms)) if gradnorms else 0)
                checkpoint();last_checkpoint=time.monotonic()
            elif now-last_checkpoint>=120:
                checkpoint();last_checkpoint=now
    finally:
        checkpoint(force_replay=True);log('stopped',reason='signal' if STOP else 'deadline')

if __name__=='__main__':main()
