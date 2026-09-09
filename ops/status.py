"""Compact read-only experiment status for coordinator heartbeats."""
import datetime,json,pathlib,subprocess,time,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),'runs':{},'evaluations':{}}
for p in (ROOT/'runs').glob('*/heartbeat.json'):
    if p.parent.name=='smoke':continue
    if '--brief' in sys.argv and not p.parent.name.startswith('final'):continue
    try:
        d=json.loads(p.read_text());report['runs'][p.parent.name]={k:d.get(k) for k in ('event','games','updates','cutoffs','elapsed','policy_loss','value_loss')};report['runs'][p.parent.name]['heartbeat_age']=round(time.time()-d['time'])
    except (ValueError,OSError):pass
    if '--brief' in sys.argv:
        try:
            for line in reversed((p.parent/'metrics.jsonl').read_text()[-65536:].splitlines()):
                try: latest=json.loads(line)
                except ValueError: continue
                if latest.get('event')=='train':
                    report['runs'][p.parent.name].update({k:round(latest[k],4) for k in ('policy_loss','value_loss')})
                    break
        except (OSError,KeyError):pass
for p in ([] if '--brief' in sys.argv else (ROOT/'reports').glob('pilot_*.json')):
    try:
        d=json.loads(p.read_text());report['evaluations'][p.stem]=[d.get('wins'),d.get('losses'),d.get('cutoffs')]
    except (ValueError,OSError):pass
try:
    result=subprocess.run(['supervisorctl','status'],capture_output=True,text=True,timeout=5)
    report['services']={parts[0]:parts[1] for line in result.stdout.splitlines() if (parts:=line.split()) and parts[0].startswith('gipf_')}
except Exception as e:report['service_error']=type(e).__name__
try:
    result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5);report['gpu_util_percent_memory_mb']=result.stdout.strip()
except Exception:pass
try:
    champion=json.loads((ROOT/'reports/champion.json').read_text());report['champion']={'name':champion['name'],'games':champion['games_trained']}
    league=ROOT/'runs/final/league-heartbeat.json'
    if league.exists():report['league']=json.loads(league.read_text())
except (OSError,ValueError,KeyError):pass
print(json.dumps(report))
