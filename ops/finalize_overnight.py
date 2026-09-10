"""Publish the completed overnight report after the managed runner exits."""
import datetime,hashlib,json,os,signal,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEADLINE=datetime.datetime.fromisoformat('2026-09-10T06:00:00+00:00').timestamp()
EARLIEST=DEADLINE-1800
STATUS=ROOT/'runs/overnight-publication.json'
def emit(event,**kw):
 row={'time':time.time(),'event':event,**kw};p=STATUS.with_suffix('.tmp');p.write_text(json.dumps(row)+'\n');p.replace(STATUS);print(json.dumps(row),flush=True)
def run(label,command,seconds=90):
 allowed=min(seconds,int(DEADLINE-time.time()-5))
 if allowed<5:raise TimeoutError('publication deadline')
 emit('running',step=label)
 p=subprocess.Popen(command,cwd=ROOT,start_new_session=True)
 try:
  result=p.wait(timeout=allowed)
  if result:raise RuntimeError(f'{label} exited {result}')
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def continuation_report_sources(data):
 continuation=data.get('continuation') or {};run_value=continuation.get('run') if isinstance(continuation,dict) else None
 if not isinstance(run_value,str):return []
 run=(ROOT/run_value).resolve()
 try:run.relative_to(ROOT.resolve())
 except ValueError:return []
 if not run.is_dir():return []
 result=[]
 for log in sorted(list(run.glob('league-*-duel.log'))+list(run.glob('league-*-greedy.log'))):
  source=ROOT/'reports'/f'{log.stem}.json'
  if source.is_file():result.append(source)
 return result
def collect_continuation_reports(data,report_dir):
 """Copy only league reports named by this continuation's own durable logs."""
 copied=[]
 for source in continuation_report_sources(data):
  target=report_dir/source.name
  raw=source.read_bytes()
  if target.exists() and target.read_bytes()!=raw:raise RuntimeError(f'conflicting continuation report: {target}')
  if not target.exists():target.write_bytes(raw)
  copied.append(target)
 return copied
def served_metadata():
 path=ROOT/'checkpoints/champion.pt';metadata_path=ROOT/'reports/champion.json';metadata=json.loads(metadata_path.read_text())
 metadata['served_artifact_sha256']=digest(path)
 tmp=metadata_path.with_suffix('.tmp');tmp.write_text(json.dumps(metadata,indent=2)+'\n');tmp.replace(metadata_path)
 return metadata
def model_card(metadata):
 return ("# GIPF Zero checkpoint\n\n"
         "Custom PyTorch `state_dict` checkpoint for the [GIPF Zero source](https://github.com/wusche1/gipf-zero). Load it with `training.model.load_model`.\n\n"
         f"Published public opponent: **{metadata.get('name','GIPF Zero')}**, trained for **{metadata.get('games_trained','unknown')} self-play games**. Architecture: `{json.dumps(metadata.get('architecture',{}),sort_keys=True)}`.\n\n"
         f"Source training checkpoint SHA-256: `{metadata.get('source_checkpoint_sha256','unknown')}`.\n\n"
         f"Served `champion.pt` artifact SHA-256: `{metadata.get('served_artifact_sha256','unknown')}`.\n\n"
         "`RESULTS.md` documents the controlled overnight comparison of MLP, square-CNN, hex-CNN, and query-Transformer policies, plus its separately labelled continuation evaluations. These automated results are not a human or expert rating.\n")
def main():
 signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
 while time.time()<DEADLINE-30:
  p=subprocess.run(['supervisorctl','status','gipf_overnight'],capture_output=True,text=True,timeout=10)
  words=p.stdout.split();state=words[1] if len(words)>1 else 'UNKNOWN'
  if time.time()>=EARLIEST and state in ('EXITED','STOPPED','FATAL'):break
  emit('waiting',service_state=state);time.sleep(30)
 else:raise TimeoutError('runner has not exited before publication deadline')
 from ops.render_overnight_report import latest_summary,render
 summary=latest_summary();data=json.loads(summary.read_text());copied=collect_continuation_reports(data,summary.parent);report=render(summary)
 stage=Path(tempfile.mkdtemp(prefix='gipf-final-report-'))
 for path in summary.parent.iterdir():
  if path.suffix in ('.json','.jsonl','.md'):(stage/path.name).write_bytes(path.read_bytes())
 (stage/'results.json').write_bytes(summary.read_bytes())
 metadata=served_metadata();(stage/'champion-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
 (stage/'README.md').write_text('---\nlibrary_name: pytorch\ntags:\n- reinforcement-learning\n- board-game\n---\n\n'+model_card(metadata)+'\n[Full overnight comparison and results](RESULTS.md).\n')
 # Publish the artifact whose digest is named in the card, even after a prior
 # partial upload, so the repository stays a self-contained served checkpoint.
 (stage/'champion.pt').write_bytes((ROOT/'checkpoints/champion.pt').read_bytes())
 (stage/'config.json').write_text(json.dumps(metadata['architecture'],indent=2)+'\n')
 hf="from huggingface_hub import HfApi;import sys;a=HfApi();a.create_repo('wuschelschulz/gipf-zero',exist_ok=True);a.upload_folder(repo_id='wuschelschulz/gipf-zero',folder_path=sys.argv[1],commit_message='Publish completed overnight evaluation report')"
 run('huggingface-report',[sys.executable,'-c',hf,str(stage)],120)
 rel=str(summary.parent.relative_to(ROOT))
 run('git-stage',['git','add','--',rel,'reports/champion.json','reports/promotions.jsonl'],15)
 changed=subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT,timeout=10).returncode
 if changed:run('git-commit',['git','commit','-m','Publish completed overnight report and champion metadata'],30)
 run('git-push',['git','push','origin','main'],90)
 emit('complete',summary=str(summary.relative_to(ROOT)),hf_repo='wuschelschulz/gipf-zero',report=str(report.relative_to(ROOT)),continuation_reports=[str(path.relative_to(ROOT)) for path in copied])
if __name__=='__main__':
 try:main()
 except Exception as e:emit('failed',error=repr(e));raise
