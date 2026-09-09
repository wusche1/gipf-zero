"""Small read-only status snapshot for the managed overnight experiment."""
import datetime,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def snapshot():
 p=sorted((ROOT/'reports/overnight').glob('*/summary.json'))[-1];s=json.loads(p.read_text());run=ROOT/'runs/overnight'/s['tag']
 heart=run/'heartbeat.jsonl'
 with heart.open('rb') as f:
  f.seek(max(0,heart.stat().st_size-16000)); lines=f.read().decode().splitlines()
 event=json.loads(lines[-1]); hs=list(run.glob('*/heartbeat.json')); active=max(hs,key=lambda p:p.stat().st_mtime) if hs else None
 data=json.loads(active.read_text()) if active else {}
 gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5).stdout.strip()
 return {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),'status':s['status'],'stage':event.get('label',event['event']),'trained':len(s['training']),'duels':len(s['duels']),'errors':len(s['errors']),'active':active.parent.name if active else None,'games':data.get('games'),'updates':data.get('updates'),'cutoffs':data.get('cutoffs'),'heartbeat_age_s':round(time.time()-data.get('time',time.time())),'gpu_util_memory_mb':gpu,'winner':(s.get('ranking') or {}).get('winner'),'champion':json.loads((ROOT/'reports/champion.json').read_text())['name']}
if __name__=='__main__':print(json.dumps(snapshot()),flush=True)
