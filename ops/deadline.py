"""Independent training cutoff; keeps public inference and tunnel alive."""
import datetime,time,subprocess,json
from pathlib import Path
DEADLINE=datetime.datetime(2026,9,9,22,0,tzinfo=datetime.timezone.utc).timestamp()
while time.time()<DEADLINE:
    time.sleep(min(30,max(0,DEADLINE-time.time())))
for name in ('gipf_pilots','gipf_factorized_pilot','gipf_final_training'):
    result=subprocess.run(['supervisorctl','stop',name],capture_output=True,text=True,timeout=55)
    print(json.dumps({'event':'deadline_stop','service':name,'result':result.stdout.strip(),'time':time.time()}),flush=True)
Path('/workspace/gipf/ops/deadline-reached.json').write_text(json.dumps({'stopped_at':time.time(),'inference_preserved':True})+'\n')
