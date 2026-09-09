"""Public, bounded game-only inference API. No account or instance credentials."""
import asyncio,json,time,threading,os
from pathlib import Path
from collections import defaultdict,deque
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI,HTTPException,Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field,ConfigDict
import torch
import gipf_engine as ge
from baselines import MCTSPlayer
from training.model import load_model
from training.search import choose_action

ROOT=Path(__file__).resolve().parents[1]
CHAMPION=ROOT/'checkpoints/champion.pt'
torch.set_num_threads(1)
app=FastAPI(title='GIPF Zero',docs_url=None,redoc_url=None,openapi_url=None)
app.add_middleware(CORSMiddleware,allow_origins=['https://wusche1.github.io'],allow_methods=['GET','POST'],allow_headers=['Content-Type'])
executor=ThreadPoolExecutor(max_workers=2)
lock=threading.Lock();model=None;model_stamp=0;model_info={}
inflight=0;requests=defaultdict(deque);global_requests=deque()

class MoveRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    state:dict
    budget_ms:int=Field(default=1000,ge=50,le=2500)
    difficulty:str=Field(default='strong',max_length=20)

@app.middleware('http')
async def bounds(request:Request,call_next):
    from starlette.responses import JSONResponse
    if request.method=='POST':
        try:length=int(request.headers.get('content-length','0'))
        except ValueError:return JSONResponse({'detail':'Invalid length'},400)
        if length>8192:return JSONResponse({'detail':'Request too large'},413)
        body=await request.body()
        if len(body)>8192:return JSONResponse({'detail':'Request too large'},413)
    return await call_next(request)

def validate_state(data):
    if not isinstance(data,dict):raise ValueError('state must be an object')
    s=ge.State.from_dict(data)
    if s.winner:raise ValueError('Game is already over')
    if s.ply>100000:raise ValueError('Invalid ply')
    for color in (1,-1):
        if not 1<=sum(x==2*color for x in s.board)<=3:raise ValueError('Invalid GIPF piece count')
    if not s.legal_actions():raise ValueError('No legal action')
    return s

def champion_metadata(info):
    """Accept only the checkpoint shape produced by the local trainer."""
    if not isinstance(info,Mapping) or not isinstance(info.get('config'),Mapping):
        raise ValueError('Invalid champion checkpoint schema')
    name=info.get('name','RL champion')
    iteration=info.get('iteration',0)
    games=info.get('games',0)
    if not isinstance(name,str) or len(name)>80 or any(not isinstance(value,int) or isinstance(value,bool) or value<0 for value in (iteration,games)):
        raise ValueError('Invalid champion checkpoint metadata')
    return {'name':name,'iteration':iteration,'games':games}

def infer(data,budget_ms):
    global model,model_stamp,model_info
    state=validate_state(data)
    # Load only a separately promoted champion; never a half-written training file.
    with lock:
        if CHAMPION.exists():
            stamp=CHAMPION.stat().st_mtime_ns
            if stamp!=model_stamp:
                try:
                    candidate,info=load_model(CHAMPION,'cpu')
                    metadata=champion_metadata(info)
                except (OSError,RuntimeError,TypeError,ValueError,KeyError):
                    # Keep a previously loaded champion (or the baseline) alive until a
                    # newly promoted, valid checkpoint changes the stamp.
                    model_stamp=stamp
                else:
                    model=candidate;model_stamp=stamp;model_info=metadata
        elif model is not None:
            model=None;model_stamp=0;model_info={}
        active=model;info=dict(model_info)
    start=time.monotonic()
    if active is None:
        action=MCTSPlayer(simulations=128,seed=int(time.time_ns())).choose_action(state,time_limit=budget_ms/1000)
        details={'model':'MCTS baseline','kind':'baseline'}
    else:
        action,stats=choose_action(active,state,'cpu',simulations=800,budget_ms=budget_ms)
        details={'model':info.get('name','RL champion'),'kind':'rl',**stats}
    if action not in state.legal_actions():raise RuntimeError('Inference returned illegal action')
    return {'action':action,'elapsed_ms':round((time.monotonic()-start)*1000),**details}

@app.get('/api/status')
def status():
    report={}
    p=ROOT/'reports/champion.json'
    if p.exists():
        try:report=json.loads(p.read_text())
        except ValueError:pass
    return {'ok':True,'game':'standard-gipf','model':report or {'name':'MCTS baseline','kind':'baseline'},'training_deadline':'2026-09-09T22:00:00Z'}

@app.post('/api/move')
async def move(body:MoveRequest,request:Request):
    global inflight
    # Cloudflare supplies this header; direct localhost users are grouped together.
    ip=request.headers.get('cf-connecting-ip',request.client.host if request.client else 'unknown')
    if len(ip)>64:ip='invalid-client'
    now=time.monotonic()
    if len(requests)>5000:
        for key in list(requests):
            if not requests[key] or now-requests[key][-1]>60:requests.pop(key,None)
    bucket=requests.get(ip)
    if bucket is None:
        # A forged/header-varying client identifier must not create unbounded
        # rate-limit bookkeeping. Overflow callers share one conservative bucket.
        bucket=requests[ip] if len(requests)<5000 else requests['_overflow']
    while bucket and now-bucket[0]>60:bucket.popleft()
    while global_requests and now-global_requests[0]>60:global_requests.popleft()
    if len(bucket)>=40 or len(global_requests)>=150:raise HTTPException(429,'Move limit reached; try again shortly')
    # Count malformed attempts too; otherwise a cheap invalid body bypasses limits.
    bucket.append(now);global_requests.append(now)
    if inflight>=2:raise HTTPException(503,'The AI is thinking for other players; try again shortly')
    try:validate_state(body.state)
    except (ValueError,TypeError,RuntimeError,OverflowError) as e:raise HTTPException(422,str(e))
    inflight+=1
    try:
        return await asyncio.get_running_loop().run_in_executor(executor,infer,body.state,body.budget_ms)
    finally:inflight-=1
