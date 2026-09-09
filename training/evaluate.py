"""Batched GPU evaluation with fixed, seeded opening diversity and separate cutoffs."""
import argparse,json,time,hashlib
from pathlib import Path
import random
import numpy as np
import torch
import gipf_engine as ge
from baselines import make_player
from baselines.evaluate import wilson_interval
from .model import load_model
from .search import Node,BatchedMCTS


def evaluation_key(args, checkpoint_sha256):
    """Fields that affect game work/results and must agree for JSONL resume."""
    payload = {
        'checkpoint_sha256': checkpoint_sha256,
        'opponent': args.opponent,
        'simulations': args.simulations,
        'move_seconds': args.move_seconds,
        'max_ply': args.max_ply,
        'seed': args.seed,
        'opening_plies': args.opening_plies,
        'neural_budget_ms': args.neural_budget_ms,
        'threads': args.threads,
        'device': args.device,
        'batch': args.batch,
        'baseline_simulations': args.baseline_simulations,
        'baseline_depth': args.baseline_depth,
    }
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def settle_opening(state, opening_plies, rng):
    """Produce a settled push-phase opening, never an arbitrary pending capture."""
    while (state.ply < opening_plies or state.phase == 'capture') and not state.winner:
        state.apply(rng.choice(state.legal_actions()))
    return state


def is_insertion_cutoff(state, max_ply):
    """An insertion cap becomes final only after mandatory captures settle."""
    return state.ply >= max_ply and state.phase != 'capture'


def load_resume_records(path, key, games):
    if not path.exists():
        raise FileNotFoundError(f'no heartbeat to resume: {path}')
    records=[];ids=set()
    for line_number,line in enumerate(path.read_text().splitlines(),1):
        if not line.strip():continue
        record=json.loads(line)
        if record.get('evaluation_key') != key:
            raise ValueError(f'heartbeat configuration/checkpoint mismatch at {path}:{line_number}')
        game_id=record.get('game')
        if not isinstance(game_id,int) or not 0 <= game_id < games or game_id in ids:
            raise ValueError(f'invalid or duplicate game id at {path}:{line_number}')
        ids.add(game_id);records.append(record)
    return records,ids


def measurement_stats(values):
    if not values:return {'count':0,'mean':None,'min':None,'max':None}
    return {'count':len(values),'mean':sum(values)/len(values),'min':min(values),'max':max(values)}


def outcome_summary(records):
    wins=sum(record['outcome']=='win' for record in records)
    losses=sum(record['outcome']=='loss' for record in records)
    cutoffs=sum(record['outcome']=='cutoff' for record in records)
    decisive=wins+losses
    return {
        'games':len(records),'wins':wins,'losses':losses,'cutoffs':cutoffs,
        'insertion_cutoffs':sum(record.get('cutoff_reason')=='insertion' for record in records),
        'decision_cutoffs':sum(record.get('cutoff_reason')=='decision' for record in records),
        'decisive_games':decisive,
        'win_rate_decisive':wins/decisive if decisive else None,
        'wilson_95_decisive':wilson_interval(wins,decisive) if decisive else [0,1],
    }


def paired_opening_summary(records):
    """Summarize only pairs in which neural played both colours to completion."""
    by_pair={}
    for record in records:
        by_pair.setdefault(record['game']//2,{})[record['neural_color']]=record
    pairs=[]
    counts={'neural_sweeps':0,'splits':0,'opponent_sweeps':0,'pairs_with_cutoff':0}
    for pair_id,sides in sorted(by_pair.items()):
        if 1 not in sides or -1 not in sides:
            continue
        white,black=sides[1],sides[-1]
        outcomes=[white['outcome'],black['outcome']]
        if 'cutoff' in outcomes:
            outcome='cutoff';counts['pairs_with_cutoff']+=1
        elif outcomes==['win','win']:
            outcome='neural_sweep';counts['neural_sweeps']+=1
        elif outcomes==['loss','loss']:
            outcome='opponent_sweep';counts['opponent_sweeps']+=1
        else:
            outcome='split';counts['splits']+=1
        pairs.append({'pair':pair_id,'white_outcome':white['outcome'],'black_outcome':black['outcome'],'outcome':outcome})
    return {'complete_pairs':len(pairs),**counts,'records':pairs}


def baseline_settings(args):
    settings={'name':args.opponent}
    if args.opponent=='mcts':settings['simulations']=args.baseline_simulations or 128
    if args.opponent=='minimax':settings['depth']=args.baseline_depth or 2
    return settings

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--opponent',default='random')
    p.add_argument('--games',type=int,default=40);p.add_argument('--batch',type=int,default=32);p.add_argument('--simulations',type=int,default=128)
    p.add_argument('--move-seconds',type=float,default=.05);p.add_argument('--max-ply',type=int,default=240);p.add_argument('--seed',type=int,default=91823)
    p.add_argument('--opening-plies',type=int,default=4);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--seconds',type=int,default=600);p.add_argument('--resume',action='store_true');p.add_argument('--neural-budget-ms',type=float,default=0,help='optional cap per batched neural search; 0 keeps the fixed simulation count');p.add_argument('--threads',type=int,default=2);p.add_argument('--baseline-simulations',type=int);p.add_argument('--baseline-depth',type=int)
    args=p.parse_args()
    if args.neural_budget_ms<0:p.error('--neural-budget-ms must be non-negative')
    if args.threads<1:p.error('--threads must be positive')
    if args.baseline_simulations is not None and args.opponent!='mcts':p.error('--baseline-simulations is only valid with --opponent mcts')
    if args.baseline_depth is not None and args.opponent!='minimax':p.error('--baseline-depth is only valid with --opponent minimax')
    if args.baseline_simulations is not None and args.baseline_simulations<1:p.error('--baseline-simulations must be positive')
    if args.baseline_depth is not None and args.baseline_depth<1:p.error('--baseline-depth must be positive')
    torch.set_num_threads(args.threads)
    checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    model,info=load_model(args.checkpoint,args.device);search=BatchedMCTS(model,args.device,seed=args.seed)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);heartbeat=out.with_suffix('.jsonl');key=evaluation_key(args,checkpoint_sha256)
    if args.resume:records,completed=load_resume_records(heartbeat,key,args.games)
    else:
        records=[];completed=set();heartbeat.write_text('')
    start=time.monotonic();deadline=start+args.seconds
    active=[];next_id=0;root_simulations=[];batch_search_seconds=[]
    def newgame(gid):
        s=ge.State();rng=random.Random(args.seed+gid//2)
        # Identical opening for a colour-swapped pair, independent of training.
        settle_opening(s,args.opening_plies,rng)
        overrides={}
        if args.opponent=='mcts' and args.baseline_simulations is not None:overrides['simulations']=args.baseline_simulations
        if args.opponent=='minimax' and args.baseline_depth is not None:overrides['depth']=args.baseline_depth
        return {'id':gid,'state':s,'side':1 if gid%2==0 else -1,'decisions':0,'opponent':make_player(args.opponent,seed=args.seed+gid,**overrides)}
    while len(records)<args.games and time.monotonic()<deadline:
        while len(active)<args.batch:
            while next_id in completed and next_id<args.games:next_id+=1
            if next_id>=args.games:break
            active.append(newgame(next_id));next_id+=1
        neural=[]
        for game in active:
            s=game['state']
            if s.current_player==game['side']:neural.append(game)
            else:
                action=game['opponent'].choose_action(s,time_limit=args.move_seconds)
                s.apply(action);game['decisions']+=1
        if neural:
            roots=[Node(g['state'].clone()) for g in neural]
            search_started=time.monotonic()
            if args.simulations:
                search_deadline=deadline if args.neural_budget_ms<=0 else min(deadline,time.monotonic()+args.neural_budget_ms/1000)
                policies=search.search(roots,args.simulations,deadline=search_deadline)
            else:
                search.evaluate(roots);policies=[r.p for r in roots]
            batch_search_seconds.append(time.monotonic()-search_started)
            root_simulations.extend(int(root.n.sum()) for root in roots)
            for game,root,pi in zip(neural,roots,policies):
                action=int(root.actions[int(np.argmax(pi))]);game['state'].apply(action);game['decisions']+=1
        remaining=[]
        for game in active:
            s=game['state']
            insertion_cutoff=is_insertion_cutoff(s,args.max_ply)
            decision_cutoff=game['decisions']>=2000
            if s.winner or insertion_cutoff or decision_cutoff:
                r={'game':game['id'],'neural_color':game['side'],'winner':s.winner,'outcome':'win' if s.winner==game['side'] else 'loss' if s.winner else 'cutoff','cutoff_reason':'insertion' if not s.winner and insertion_cutoff else 'decision' if not s.winner and decision_cutoff else None,'ply':s.ply,'decisions':game['decisions'],'evaluation_key':key}
                records.append(r)
                with heartbeat.open('a') as f:f.write(json.dumps(r)+'\n');f.flush()
                print(json.dumps({'finished':len(records),'games':args.games,'last':r,'elapsed':round(time.monotonic()-start,1)}),flush=True)
            else:remaining.append(game)
        active=remaining
    summary=outcome_summary(records);by_colour={'white':outcome_summary([r for r in records if r['neural_color']==1]),'black':outcome_summary([r for r in records if r['neural_color']==-1])}
    result={'checkpoint':args.checkpoint,'checkpoint_sha256':checkpoint_sha256,'evaluation_key':key,'model_games':info.get('games',0),'model_iteration':info.get('iteration',0),'opponent':args.opponent,'baseline_settings':baseline_settings(args),'requested_games':args.games,'finished_games':len(records),'wins':summary['wins'],'losses':summary['losses'],'cutoffs':summary['cutoffs'],'unfinished':args.games-len(records),'win_rate_all_finished':summary['wins']/len(records) if records else None,'win_rate_decisive':summary['win_rate_decisive'],'wilson_95_all_finished':wilson_interval(summary['wins'],len(records)) if records else [0,1],'wilson_95_decisive':summary['wilson_95_decisive'],'by_neural_colour':by_colour,'paired_openings':paired_opening_summary(records),'neural_root_simulations':measurement_stats(root_simulations),'neural_batch_search_seconds':measurement_stats(batch_search_seconds),'elapsed_seconds':time.monotonic()-start,'config':vars(args),'records':records}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('records','config')}),flush=True)
if __name__=='__main__':main()
