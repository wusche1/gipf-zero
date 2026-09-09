"""Keep the Pages endpoint current after a managed quick-tunnel restart.

Only updates web/config.json. Does not reset, rebase, or overwrite remote work.
"""
import json,re,time,subprocess,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LOG=Path('/var/log/portal/gipf_tunnel.log')
CONFIG=ROOT/'web/config.json'

def tick():
    if not LOG.exists() or not CONFIG.exists():return
    urls=re.findall(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com',LOG.read_text()[-100000:])
    if not urls:return
    endpoint=urls[-1];config=json.loads(CONFIG.read_text())
    if config.get('aiEndpoint')==endpoint:return
    with urllib.request.urlopen(endpoint+'/api/status',timeout=10) as r:
        if not json.load(r).get('ok'):return
    # Do not operate until the repository has its first reviewed commit.
    if subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,timeout=10).returncode:return
    config['aiEndpoint']=endpoint
    temp=CONFIG.with_suffix('.tmp');temp.write_text(json.dumps(config,indent=2)+'\n');temp.replace(CONFIG)
    subprocess.run(['git','add','--','web/config.json'],cwd=ROOT,check=True,timeout=15)
    subprocess.run(['git','commit','--only','web/config.json','-m','Update public inference endpoint after tunnel restart'],cwd=ROOT,check=True,timeout=20)
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True,timeout=30)
    print(json.dumps({'event':'endpoint_updated','endpoint':endpoint,'time':time.time()}),flush=True)

if __name__=='__main__':
    while True:
        try:tick()
        except Exception as e:print(json.dumps({'event':'endpoint_sync_error','error':str(e),'time':time.time()}),flush=True)
        time.sleep(30)
