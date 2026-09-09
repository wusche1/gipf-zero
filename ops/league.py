"""Periodically challenge the served champion with immutable training snapshots.

No external model calls. Every match has a wall-clock timeout and durable logs.
A replacement needs a >50% Wilson lower bound in a paired 80-game match and
at least 32/40 wins against the frozen greedy baseline. Cutoffs never count as wins.
"""
import argparse, datetime, hashlib, io, json, os, subprocess, sys, time
import torch
from pathlib import Path
from ops.promote import promote
ROOT=Path(__file__).resolve().parents[1]

def run(command, log, seconds):
    import signal
    with log.open('w') as stream:
        process=subprocess.Popen([sys.executable,'-u',*command],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        start=time.monotonic()
        try:
            while process.poll() is None:
                if time.monotonic()-start>seconds:raise TimeoutError(str(command))
                time.sleep(5)
            if process.returncode:raise RuntimeError(f'Command failed ({process.returncode}); see {log}')
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)


def main():
    import signal
    def stop(*_):raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop)
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--interval',type=int,default=1800);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    folder=ROOT/a.run; archive=ROOT/'checkpoints/league';archive.mkdir(parents=True,exist_ok=True)
    heartbeat=folder/'league-heartbeat.json';last_hash=None;next_check=time.time()+a.interval
    if heartbeat.exists():
        try:next_check=max(time.time(),float(json.loads(heartbeat.read_text()).get('next_check',next_check)))
        except (ValueError,TypeError):pass
    def log(event,**kw):
        data={'time':time.time(),'event':event,**kw};heartbeat.write_text(json.dumps(data)+'\n');print(json.dumps(data),flush=True)
    while time.time()<a.deadline:
        if time.time()<next_check:log('waiting',next_check=next_check);time.sleep(min(30,max(0,next_check-time.time())));continue
        try:
            source=folder/'latest.pt'
            if not source.exists():next_check=time.time()+30;continue
            raw=source.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            if digest==last_hash:next_check=time.time()+60;continue
            stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S')
            candidate=archive/f'candidate-{stamp}.pt';candidate.write_bytes(raw)
            champion_raw=(ROOT/'checkpoints/champion.pt').read_bytes()
            champion=archive/f'champion-{stamp}.pt';champion.write_bytes(champion_raw)
            def architecture(blob):
                cfg=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)['config']
                return (cfg['kind'],cfg['width'],cfg['blocks'],cfg.get('head','flat'))
            cross_architecture=architecture(raw)!=architecture(champion_raw)
            duel=ROOT/f'reports/league-{stamp}-duel.json';greedy=ROOT/f'reports/league-{stamp}-greedy.json'
            log('duel',candidate=str(candidate),comparison='equal_cpu_50ms' if cross_architecture else 'equal_128_simulations')
            limits=['--device','cpu','--threads','1','--batch','1','--budget-ms','50','--simulations','10000','--seconds','1200'] if cross_architecture else ['--seconds','600']
            run(['-m','training.duel','--candidate',str(candidate),'--champion',str(champion),'--games','80','--seed',str(300000+int(stamp)),*limits,'--output',str(duel)],folder/f'league-{stamp}-duel.log',1250 if cross_architecture else 650)
            result=json.loads(duel.read_text());n=result['wins']+result['losses']+result['cutoffs']
            # Treat cutoffs as non-wins for the replacement confidence bound.
            from baselines.evaluate import wilson_interval
            gate=not result['unfinished'] and n==80 and wilson_interval(result['wins'],n)[0]>.5
            if gate:
                log('baseline_gate',duel_wins=result['wins'],duel_losses=result['losses'])
                run(['-m','training.evaluate','--checkpoint',str(candidate),'--opponent','greedy','--games','40','--seed',str(600000+int(stamp)),'--seconds','400','--output',str(greedy)],folder/f'league-{stamp}-greedy.log',450)
                baseline=json.loads(greedy.read_text());gate=not baseline['unfinished'] and baseline['wins']>=32
                if gate and (ROOT/'checkpoints/champion.pt').read_bytes()!=champion_raw:
                    log('stale_opponent',reason='Champion changed during evaluation');gate=False
                if gate:
                    metadata=promote(candidate,f"GIPF Zero — {baseline['model_games']:,} games",[duel,greedy]);log('promoted',games=metadata['games_trained'])
            if not gate:log('retained_champion',wins=result['wins'],losses=result['losses'],cutoffs=result['cutoffs'])
            last_hash=digest;next_check=time.time()+a.interval
        except Exception as error:log('error',error=str(error));next_check=time.time()+120
    log('deadline')
if __name__=='__main__':main()
