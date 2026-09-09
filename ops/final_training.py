"""Restart bounded training after crashes, preserving checkpoints and replay."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--run',default='runs/final');p.add_argument('--deadline',type=float,required=True);p.add_argument('--simulations',type=int,default=128);p.add_argument('--lr',type=float,default=.001);p.add_argument('--games',type=int,default=128);p.add_argument('--cuda-graph-batch',type=int,default=0);a=p.parse_args()
    run=ROOT/a.run;run.mkdir(parents=True,exist_ok=True);attempt=0
    while time.time()<a.deadline:
        resume=run/'latest.pt'
        if not resume.exists():resume=ROOT/a.source
        attempt+=1
        command=[sys.executable,'-u','-m','training.train','--run',str(run),'--resume',str(resume),'--games',str(a.games),'--games-per-iteration','64','--updates','64','--simulations',str(a.simulations),'--batch-size','256','--lr',str(a.lr),'--seconds',str(max(1,int(a.deadline-time.time()))),'--deadline',str(a.deadline),'--max-ply','300','--cuda-graph-batch',str(a.cuda_graph_batch)]
        print(json.dumps({'event':'launch','attempt':attempt,'time':time.time(),'command':command}),flush=True)
        process=subprocess.Popen(command,cwd=ROOT)
        stale=False;launched=time.time()
        while process.poll() is None:
            heartbeat=run/'heartbeat.json'
            # Allow initial replay loading; then recover if progress stops for 3 min.
            last_progress=max(launched,heartbeat.stat().st_mtime if heartbeat.exists() else launched)
            if time.time()-last_progress>180:
                stale=True;process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
                break
            if time.time()>a.deadline+30:
                process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
                break
            time.sleep(10)
        print(json.dumps({'event':'child_exit','code':process.returncode,'stale':stale,'time':time.time()}),flush=True)
        if process.returncode==0 and not stale:break
        if attempt>=5:raise RuntimeError('Repeated training failures; preserved last checkpoint')
        time.sleep(10)
if __name__=='__main__':main()
