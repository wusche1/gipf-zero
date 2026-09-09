"""Inert, deadline-bounded five-model selection, continuation, and publication."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, shutil, signal, subprocess, sys, tempfile, time
from itertools import combinations
from pathlib import Path
from typing import Any
import torch
import gipf_engine as ge
from baselines.evaluate import wilson_interval
from ops.promote import promote

ROOT=Path(__file__).resolve().parents[1]; STOP=False
def epoch(value:str)->float:return dt.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
def digest(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def tag()->str:return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def relative(path:Path)->str:
 try:return str(path.relative_to(ROOT))
 except ValueError:return str(path)

class Runner:
 def __init__(self,c:dict[str,Any]):
  self.c=c;self.deadline=epoch(c['deadline']);self.stop_at=epoch(c['training_stop']);self.tag=tag()
  self.base=ROOT/'runs'/'overnight'/self.tag;self.reports=ROOT/'reports'/'overnight'/self.tag;self.archive=ROOT/'checkpoints'/'overnight'/self.tag
  for p in (self.base,self.reports,self.archive):p.mkdir(parents=True,exist_ok=False)
  self.heart=self.base/'heartbeat.jsonl';self.summary_path=self.reports/'summary.json';self.backend={};self.summary={'version':c['version'],'tag':self.tag,'deadline':self.deadline,'training_stop':self.stop_at,'status':'running','errors':[],'training':[],'duels':[],'ranking':None,'continuation':None,'promotion':None,'publication':None}
  (self.reports/'config.json').write_text(json.dumps(c,indent=2)+'\n')
 def remaining(self):return self.deadline-time.time()
 def emit(self,event,**kw):
  row={'time':time.time(),'remaining_seconds':max(0,self.remaining()),'event':event,**kw}
  with self.heart.open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush()
  t=self.summary_path.with_suffix('.tmp');t.write_text(json.dumps(self.summary,indent=2,sort_keys=True)+'\n');os.replace(t,self.summary_path);print(json.dumps(row,sort_keys=True),flush=True)
 def error(self,label,e):self.summary['errors'].append({'label':label,'type':type(e).__name__,'message':str(e)});self.emit('error',label=label,error=repr(e))
 def revision(self):
  native=Path(ge.__file__).resolve(); files=[native,ROOT/'training/model.py',ROOT/'training/query_transformer.py',ROOT/'training/train.py',ROOT/'training/search.py',ROOT/'training/native_search.py']
  return {str(x.relative_to(ROOT)) if x.is_relative_to(ROOT) else str(x):digest(x) for x in files}
 def check_backend(self):
  if self.revision()!=self.backend:raise RuntimeError('backend/model/train files changed after overnight freeze')
 def child(self,label,cmd,log,seconds,need=None):
  if STOP or self.remaining()<=25:return False
  allowed=min(seconds,max(1,int(self.remaining()-20)));log.parent.mkdir(parents=True,exist_ok=True);p=None;self.emit('child_start',label=label,command=cmd,outer_seconds=allowed)
  try:
   with log.open('w') as f:
    p=subprocess.Popen([sys.executable,'-u',*cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,text=True,start_new_session=True);start=time.monotonic();beat=start+30
    while p.poll() is None:
     if STOP or time.monotonic()-start>=allowed or self.remaining()<=0:raise TimeoutError(f'{label} timed out')
     if time.monotonic()>=beat:self.emit('child_running',label=label,elapsed_seconds=round(time.monotonic()-start,1));beat+=30
     time.sleep(2)
   if p.returncode:raise RuntimeError(f'{label} exited {p.returncode}')
   if need and not need.exists():raise RuntimeError(f'{label} missing {need}')
   self.emit('child_complete',label=label);return True
  except Exception as e:self.error(label,e);return False
  finally:
   if p and p.poll() is None:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=12)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=8)
 def wait_service(self):
  end=time.monotonic()+self.c['wait_seconds'];name=self.c['wait_service']
  while time.monotonic()<end and self.remaining()>25 and not STOP:
   r=subprocess.run(['supervisorctl','status',name],text=True,capture_output=True,timeout=10);line=(r.stdout+r.stderr).strip();state=line.split()[1] if len(line.split())>1 else 'UNKNOWN'
   self.emit('wait_service',service=name,state=state)
   if state in ('EXITED','STOPPED','FATAL'):return True
   time.sleep(30)
  self.error('wait_service',TimeoutError(name));return False
 def preflight(self):
  from training.model import ModelConfig,PolicyValue,encode
  m=PolicyValue(ModelConfig('transformer',164,5,'query')).cpu().eval()
  with torch.inference_mode(): logits,value=m(torch.tensor(encode(ge.State())[None]))
  if logits.shape!=(1,2730) or value.shape!=(1,) or not torch.isfinite(logits).all():raise RuntimeError('query integration preflight failed')
  self.backend=self.revision();self.summary['backend_revision']=self.backend;self.emit('backend_frozen')
 def freeze(self,name,source):
  raw=source.read_bytes();h=hashlib.sha256(raw).hexdigest();target=self.archive/f'{name}-{h[:12]}.pt';target.write_bytes(raw);self.emit('checkpoint_frozen',name=name,path=str(target.relative_to(ROOT)),sha256=h);return target
 def train(self,a,seed):
  self.check_backend();run=self.base/f"{a['name']}-seed{seed}";s=self.c['selfplay'];started=time.monotonic();ok=False
  for attempt in range(2):
   seconds=int(self.c['training_seconds']-(time.monotonic()-started))
   if seconds<30:break
   cmd=['-m','training.train','--run',str(run),'--kind',a['kind'],'--head',a['head'],'--width',str(a['width']),'--blocks',str(a['blocks']),'--seed',str(seed),'--seconds',str(seconds),'--games',str(s['games']),'--games-per-iteration',str(s['games_per_iteration']),'--updates',str(s['updates']),'--simulations',str(s['simulations']),'--batch-size',str(s['batch_size']),'--lr',str(s['lr']),'--max-ply',str(s['max_ply']),'--replay-size',str(s['replay_size']),'--device',s['device'],'--native-forest']
   if attempt:
    if not (run/'latest.pt').exists():break
    cmd.extend(['--resume',str(run/'latest.pt')]);self.emit('training_recovery',run=run.name,seconds=seconds)
   ok=self.child(f'train-{run.name}-attempt{attempt+1}',cmd,run/f'train-{attempt+1}.log',seconds+30,run/'latest.pt')
   if ok:break
  entry={'name':run.name,'run':relative(run),'completed':ok}
  if ok:entry['frozen']=relative(self.freeze(run.name,run/'latest.pt'))
  self.summary['training'].append(entry);return run if ok else None
 def duel(self,left,right,mode,seed,attempt=1,games=None,seconds=None):
  self.check_backend();d=self.c['duels'];spec=d[mode];games=d['games'] if games is None else games;seconds=(d['seconds_cpu'] if mode=='equal_cpu_time' else d['seconds_sim']) if seconds is None else seconds;out=self.reports/f'{left.name}-vs-{right.name}-{mode}-seed{seed}-attempt{attempt}.json';cmd=['-m','training.duel','--candidate',str(left/'latest.pt'),'--champion',str(right/'latest.pt'),'--games',str(games),'--batch',str(spec['batch']),'--device',spec['device'],'--threads',str(spec['threads']),'--simulations',str(spec['simulations']),'--budget-ms',str(spec['budget_ms']),'--seed',str(seed),'--max-ply','300','--seconds',str(seconds),'--output',str(out)]
  ok=self.child('duel-'+out.stem,cmd,out.with_suffix('.log'),seconds+10,out);data=json.loads(out.read_text()) if ok else None;rec={'left':left.name,'right':right.name,'mode':mode,'seed':seed,'attempt':attempt,'report':relative(out),'complete':bool(data and data.get('unfinished')==0),'result':None if not data else {k:data.get(k) for k in ('wins','losses','cutoffs','unfinished','candidate_root_simulations','champion_root_simulations')}};self.summary['duels'].append(rec);return data
 def rank(self,models,reports):
  score={n:{'wins':0,'total':0,'roots':[]} for n in models};pair={}
  for left,right,d in reports:
   if not d or d.get('unfinished'):continue
   w,lost,cut=int(d['wins']),int(d['losses']),int(d['cutoffs'])
   for n,x,t,root in ((left,w,w+lost+cut,d.get('candidate_root_simulations',{}).get('mean')),(right,lost,w+lost+cut,d.get('champion_root_simulations',{}).get('mean'))):score[n]['wins']+=x;score[n]['total']+=t;score[n]['roots']+=[] if root is None else [root]
   key=tuple(sorted((left,right)));p=pair.setdefault(key,[0,0,0])
   if key[0]==left:p[0]+=w;p[1]+=lost
   else:p[0]+=lost;p[1]+=w
   p[2]+=w+lost+cut
  rows=[]
  for n,x in score.items():
   rate=x['wins']/x['total'] if x['total'] else -1;expected=self.c['duels']['games']*(len(models)-1)*len(self.c['seeds']);rows.append({'name':n,'wins':x['wins'],'total':x['total'],'unplayed':expected-x['total'],'rate':rate,'wilson_95':wilson_interval(x['wins'],x['total']) if x['total'] else [0,1],'mean_root_simulations':sum(x['roots'])/len(x['roots']) if x['roots'] else 0})
  minimum=self.c['duels']['min_cpu_games_per_model']
  if any(row['total']<minimum for row in rows):
   self.summary['ranking']={'rows':rows,'winner':None,'minimum_cpu_games_per_model':minimum,'reason':'insufficient complete equal-CPU coverage'};self.emit('ranking_blocked',minimum_cpu_games_per_model=minimum);return None
  rows.sort(key=lambda x:(x['rate'],x['mean_root_simulations']),reverse=True);best,alt=rows[:2];p=pair.get(tuple(sorted((best['name'],alt['name']))),[0,0,0]);bw=p[0] if best['name']<alt['name'] else p[1];lo,hi=wilson_interval(bw,p[2]) if p[2] else (0,1);inconclusive=lo<=.5<=hi
  self.summary['ranking']={'rows':rows,'winner':best['name'],'alternative':alt['name'],'direct_wilson_95':[lo,hi],'inconclusive':inconclusive,'tie_rule':'higher achieved CPU root simulations within the fixed 50 ms cap only after an extra head-to-head remains inconclusive'};self.emit('ranking',winner=best['name'],alternative=alt['name'],inconclusive=inconclusive);return best['name']
 def best_seed(self,name,evidence):
  scores={seed:[0,0] for seed in self.c['seeds']}
  for left,right,data in evidence:
   if not data or data.get('unfinished'):continue
   for run,wins in ((left,int(data['wins'])),(right,int(data['losses']))):
    if run.name.startswith(name+'-seed'):
     seed=int(run.name.rsplit('seed',1)[1]);scores[seed][0]+=wins;scores[seed][1]+=int(data['wins'])+int(data['losses'])+int(data['cutoffs'])
  seed=max(self.c['seeds'],key=lambda s:(scores[s][0]/scores[s][1] if scores[s][1] else -1,scores[s][0]))
  return seed,scores
 def start_league(self,run):
  deadline=min(self.stop_at-1200,self.deadline-60)
  if deadline<=time.time()+30:
   self.emit('league_skipped',reason='insufficient time before 05:10 UTC');return None,None
  log=(run/'overnight-league.log').open('w')
  command=[sys.executable,'-u','-m','ops.league','--run',relative(run),'--interval','1800','--deadline',str(deadline)]
  process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,text=True,start_new_session=True)
  self.emit('league_started',run=relative(run),deadline=deadline,pid=process.pid)
  return process,log
 def stop_league(self,process,log):
  if process is None:return
  try:
   if process.poll() is None:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=8)
   self.emit('league_stopped',returncode=process.returncode)
  finally:log.close()
 def continue_train(self,a,run):
  self.check_backend();s=self.c['continuation'];started=time.monotonic();allocation=max(0,int(min(self.stop_at-time.time(),self.remaining()-900)));ok=False
  self.summary['continuation']={'run':relative(run),'completed':None,'seconds':allocation,'league_deadline':min(self.stop_at-1200,self.deadline-60)};self.emit('continuation_start',run=relative(run),seconds=allocation)
  league,league_log=self.start_league(run)
  try:
   for attempt in range(2):
    seconds=int(allocation-(time.monotonic()-started))
    if seconds<30:break
    cmd=['-m','training.train','--run',str(run),'--resume',str(run/'latest.pt'),'--kind',a['kind'],'--head',a['head'],'--width',str(a['width']),'--blocks',str(a['blocks']),'--seconds',str(seconds),'--games',str(s['games']),'--games-per-iteration',str(s['games_per_iteration']),'--updates',str(s['updates']),'--simulations',str(s['simulations']),'--batch-size',str(s['batch_size']),'--lr',str(s['lr']),'--max-ply',str(s['max_ply']),'--replay-size',str(s['replay_size']),'--device',s['device'],'--native-forest']
    if attempt:self.emit('continuation_recovery',run=run.name,seconds=seconds)
    ok=self.child(f'continuation-{run.name}-attempt{attempt+1}',cmd,run/f'continuation-{attempt+1}.log',seconds+30,run/'latest.pt')
    if ok:break
  finally:self.stop_league(league,league_log)
  self.summary['continuation']['completed']=ok;return self.freeze('selected-final',run/'latest.pt') if ok else None
 def gate(self,candidate):
  champion=self.freeze('served-champion',ROOT/'checkpoints/champion.pt');duel=self.reports/'final-vs-served.json';cmd=['-m','training.duel','--candidate',str(candidate),'--champion',str(champion),'--games','80','--batch','1','--device','cpu','--threads','1','--budget-ms','50','--simulations','10000','--seed','950001','--max-ply','300','--seconds','600','--output',str(duel)];ok=self.child('final-duel',cmd,duel.with_suffix('.log'),min(630,max(1,int(self.remaining()-60))),duel);d=json.loads(duel.read_text()) if ok else {};total=sum(int(d.get(k,0)) for k in ('wins','losses','cutoffs'));passed=ok and d.get('unfinished')==0 and total==80 and wilson_interval(int(d.get('wins',0)),total)[0]>.5
  greedy=self.reports/'final-vs-greedy.json'
  if passed:
   cmd=['-m','training.evaluate','--checkpoint',str(candidate),'--opponent','greedy','--games','40','--batch','64','--device','cuda','--threads','2','--simulations','64','--seed','950002','--max-ply','300','--seconds','300','--output',str(greedy)];ok=self.child('final-greedy',cmd,greedy.with_suffix('.log'),min(330,max(1,int(self.remaining()-30))),greedy);g=json.loads(greedy.read_text()) if ok else {};passed=ok and g.get('unfinished')==0 and g.get('wins',0)>=32
  self.summary['promotion']={'candidate':relative(candidate),'duel_report':relative(duel),'greedy_report':relative(greedy) if greedy.exists() else None,'passed':passed}
  if passed:
   meta=promote(candidate,'GIPF Zero overnight winner',[duel,greedy]);self.summary['promotion']['metadata']=meta
  self.emit('promotion_gate',passed=passed);return ROOT/'checkpoints/champion.pt'
 def publish(self,champion):
  meta=json.loads((ROOT/'reports/champion.json').read_text());stage=Path(tempfile.mkdtemp(prefix='gipf-hf-'));(stage/'champion.pt').write_bytes(champion.read_bytes());(stage/'config.json').write_text(json.dumps(meta['architecture'],indent=2)+'\n');card=f"# GIPF Zero checkpoint\n\nCustom PyTorch `state_dict` checkpoint for the [GIPF Zero source](https://github.com/wusche1/gipf-zero). Load with `training.model.load_model`.\n\nSource SHA-256: `{meta['source_checkpoint_sha256']}`.\n\nEvaluation evidence: {json.dumps(meta['evaluations'])}. These are engine-baseline results, not a human or expert rating.\n";(stage/'README.md').write_text(card);(stage/'results.json').write_text(json.dumps(self.summary,indent=2)+'\n');(self.reports/'MODEL_CARD.md').write_text(card)
  repo=self.c['publication']['hf_repo'];code="from huggingface_hub import HfApi;import sys; a=HfApi();a.create_repo(repo_id=sys.argv[1],repo_type='model',private=False,exist_ok=True);a.upload_folder(repo_id=sys.argv[1],repo_type='model',folder_path=sys.argv[2],commit_message='Update verified GIPF Zero checkpoint')";hf=self.child('huggingface-upload',['-c',code,repo,str(stage)],self.reports/'huggingface-upload.log',180);shutil.rmtree(stage,ignore_errors=True)
  gitcode="import subprocess,sys;raise SystemExit(subprocess.call(sys.argv[1:]))";rel=relative(self.reports);committed=self.child('github-commit',['-c',gitcode,'git','add','--',rel],self.reports/'github-commit.log',60) and self.child('github-commit',['-c',gitcode,'git','commit','-m',f'Publish overnight evaluation {self.tag}'],self.reports/'github-commit.log',90)
  pushed=committed and self.child('github-push',['-c',gitcode,'git','push'],self.reports/'github-push.log',180)
  release=False
  if pushed:
   release=self.child('github-release',['-c',gitcode,'gh','release','create',f'overnight-{self.tag}',str(champion),'--title',f'GIPF Zero overnight {self.tag}','--notes-file',str(self.reports/'MODEL_CARD.md')],self.reports/'github-release.log',180)
  self.summary['publication']={'hf_repo':repo,'uploaded':hf,'source_hash':meta['source_checkpoint_sha256'],'github_commit':committed,'github_push':pushed,'github_release':release};self.emit('publication',uploaded=hf,github_release=release)
 def run(self):
  self.emit('runner_created')
  if not self.wait_service():self.summary['status']='blocked';self.emit('blocked');return 1
  try:self.preflight()
  except Exception as e:self.error('preflight',e);self.summary['status']='blocked';self.emit('blocked');return 1
  runs={};by={a['name']:a for a in self.c['architectures']}
  for a in self.c['architectures']:
   for seed in self.c['seeds']:
    try:r=self.train(a,seed);runs[(a['name'],seed)]=r
    except Exception as e:self.error('train',e)
  evidence=[];cpu_evidence=[]
  # Finish primary equal-CPU evidence across every available pairing before
  # spending time on the secondary fixed-simulation comparison.
  for mode in ('equal_cpu_time','equal_simulations'):
   for a,b in combinations(self.c['architectures'],2):
    for seed in self.c['seeds']:
     left,right=runs.get((a['name'],seed)),runs.get((b['name'],seed))
     if not left or not right:continue
     try:
      data=self.duel(left,right,mode,970000+seed)
      # An unfinished primary report is not evidence. One fresh, bounded retry
      # supplies the required coverage without ever overwriting raw output.
      if mode=='equal_cpu_time' and data and data.get('unfinished'):
       data=self.duel(left,right,mode,970000+seed,attempt=2)
      item=(left,right,data);evidence.append(item)
      if mode=='equal_cpu_time':cpu_evidence.append(item)
     except Exception as e:self.error('duel',e)
  named_evidence=[(next(a['name'] for a in self.c['architectures'] if l.name.startswith(a['name']+'-seed')),next(a['name'] for a in self.c['architectures'] if r.name.startswith(a['name']+'-seed')),d) for l,r,d in cpu_evidence]
  # Select the better of the two independently trained seeds using only its
  # equal-CPU-time matches; never mix secondary fixed-simulation evidence.
  names=[a['name'] for a in self.c['architectures']];winner=self.rank(names,named_evidence)
  if winner is None:
   # Do not waste the remaining training window if fresh comparisons fail.
   # Preserve that failure explicitly and continue the already-validated model.
   meta=json.loads((ROOT/'reports/champion.json').read_text());config=meta['architecture']
   fallback=self.base/'validated-incumbent-fallback';fallback.mkdir(exist_ok=True)
   (fallback/'latest.pt').write_bytes((ROOT/'checkpoints/champion.pt').read_bytes())
   architecture={'name':'validated-incumbent-fallback',**config}
   self.summary['ranking']['fallback']='Continue served model weights with fresh optimizer/replay; no architecture winner claimed'
   final=self.continue_train(architecture,fallback)
   champion=self.gate(final) if final and self.remaining()>60 else ROOT/'checkpoints/champion.pt'
   if self.remaining()>190:self.publish(champion)
   self.summary['status']='partial';self.emit('runner_finished',status='partial');return 1
  winner_seed,seed_score=self.best_seed(winner,cpu_evidence);self.summary['ranking']['winner_seed']=winner_seed;self.summary['ranking']['seed_scores']=seed_score
  # The initial 24-game direct comparison is intentionally labelled
  # inconclusive when its interval crosses 0.5. Spend a final bounded 80-game
  # match between the strongest seeds before applying the cost tie rule.
  alternative=self.summary['ranking']['alternative']
  if self.summary['ranking']['inconclusive'] and self.remaining()>900:
   alt_seed,alt_scores=self.best_seed(alternative,cpu_evidence);left,right=runs[(winner,winner_seed)],runs[(alternative,alt_seed)]
   extra=self.duel(left,right,'equal_cpu_time',989001,attempt=3,games=80,seconds=600)
   if extra and not extra.get('unfinished'):
    total=int(extra['wins'])+int(extra['losses'])+int(extra['cutoffs']);lo,hi=wilson_interval(int(extra['wins']),total)
    if hi<.5:winner,winner_seed=alternative,alt_seed
    elif lo<=.5:
     left_roots=extra.get('candidate_root_simulations',{}).get('mean',0);right_roots=extra.get('champion_root_simulations',{}).get('mean',0)
     if right_roots>left_roots:winner,winner_seed=alternative,alt_seed
    self.summary['ranking']['tie_break']={'report':self.summary['duels'][-1]['report'],'wilson_95':[lo,hi],'alternative_seed_scores':alt_scores,'selected':winner}
   else:self.summary['ranking']['tie_break']={'reason':'extra head-to-head unfinished'}
  self.summary['ranking']['winner']=winner;self.summary['ranking']['winner_seed']=winner_seed;run=runs.get((winner,winner_seed))
  final=self.continue_train(by[winner],run) if run else None
  champion=self.gate(final) if final and self.remaining()>60 else ROOT/'checkpoints/champion.pt'
  if champion.exists() and self.remaining()>190:self.publish(champion)
  self.summary['status']='complete' if not self.summary['errors'] and self.remaining()>0 else 'partial';self.emit('runner_finished',status=self.summary['status']);return 0
def main():
 global STOP
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'ops/overnight_config.json');a=p.parse_args();c=json.loads(a.config.read_text());signal.signal(signal.SIGTERM,lambda *_:globals().__setitem__('STOP',True));return Runner(c).run()
if __name__=='__main__':raise SystemExit(main())
