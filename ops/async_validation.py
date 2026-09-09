"""Wait for the isolated architecture experiment, then validate async inference."""
import json, os, signal, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/async-validation'
OUT.mkdir(exist_ok=True)
def emit(event, **details):
    row=dict(time=time.time(),event=event,**details)
    with (OUT/'heartbeat.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
def child(label,args,seconds):
    with (OUT/(label+'.log')).open('w') as f:
        p=subprocess.Popen(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        end=time.monotonic()+seconds
        try:
            while p.poll() is None:
                if time.monotonic()>end:raise TimeoutError(label)
                emit('child_running',label=label);time.sleep(10)
            if p.returncode:raise RuntimeError(f'{label} exited {p.returncode}')
        finally:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=12)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=8)
    emit('child_complete',label=label)
def main():
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    end=time.monotonic()+9000
    while True:
        paths=sorted((ROOT/'reports/architecture-comparison').glob('*/summary.json'))
        summary=json.loads(paths[-1].read_text())
        if summary['status']!='running':break
        if time.monotonic()>end:raise TimeoutError('architecture comparison still running')
        emit('waiting_for_isolated_gpu',training=len(summary['training']),duels=len(summary['duels']));time.sleep(30)
    child('apply-check',['git','apply','--check','ops/enable_async_training.patch'],15)
    child('apply',['git','apply','ops/enable_async_training.patch'],15)
    child('benchmark',[sys.executable,'benchmarks/benchmark_async_selfplay.py'],180)
    result=json.loads((OUT/'benchmark.log').read_text());(OUT/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['exact_policy_identity'], 'GPU policy differential failed; inspect before training'
    run='runs/async-integration-smoke'
    child('training-smoke',[sys.executable,'-m','training.train','--run',run,'--kind','mlp','--width','64','--blocks','2','--seconds','30','--games','128','--simulations','16','--updates','1','--batch-size','32','--games-per-iteration','32','--max-ply','300','--async-workers','4'],100)
    metrics=[json.loads(l) for l in (ROOT/run/'metrics.jsonl').read_text().splitlines()]
    assert metrics[-1]['event']=='stopped' and metrics[-1]['updates']>0 and (ROOT/run/'latest.pt').exists()
    emit('complete',benchmark_speedup=result['speedup'],smoke_games=metrics[-1]['games'],smoke_updates=metrics[-1]['updates'])
if __name__=='__main__':
    try:main()
    except Exception as e:emit('failed',error=repr(e));raise
