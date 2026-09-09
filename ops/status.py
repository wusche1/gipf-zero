"""Compact read-only experiment status for coordinator heartbeats."""
import datetime,json,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),'runs':{},'evaluations':{}}
for p in (ROOT/'runs').glob('*/heartbeat.json'):
    if p.parent.name=='smoke':continue
    try:
        d=json.loads(p.read_text());report['runs'][p.parent.name]={k:d.get(k) for k in ('event','games','updates','cutoffs','elapsed','policy_loss','value_loss')};report['runs'][p.parent.name]['heartbeat_age']=round(time.time()-d['time'])
    except (ValueError,OSError):pass
for p in (ROOT/'reports').glob('pilot_*.json'):
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
print(json.dumps(report))
