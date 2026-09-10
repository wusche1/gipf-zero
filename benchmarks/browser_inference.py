"""Measure shipped browser inference/search, blocking every non-static origin."""
import argparse,functools,json,threading,time,statistics
from pathlib import Path
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='reports/browser-inference.json');p.add_argument('--search',action='store_true');p.add_argument('--software-webgpu',action='store_true');p.add_argument('--model-variant',choices=['dynamic','single','portable'],default='dynamic');a=p.parse_args()
 fixtures=json.loads((ROOT/'web/models/champion-fixtures.json').read_text())['fixtures']
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(ROOT/'web')))
 threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}'
 result={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'model_variant':a.model_variant,'software_webgpu_requested':a.software_webgpu,'backends':{},'blocked_requests':[]}
 try:
  with sync_playwright() as pw:
   args=['--enable-unsafe-webgpu','--use-angle=swiftshader'] if a.software_webgpu else []
   browser=pw.chromium.launch(headless=True,args=args);page=browser.new_page();page.set_default_timeout(90000)
   def route(r):
    if r.request.url.startswith(base):r.continue_()
    else:result['blocked_requests'].append(r.request.url);r.abort()
   page.route('**/*',route);page.goto(base+'/models/champion-metadata.json')
   result['browser']=browser.version
   for backend in ['wasm','webgpu']:
    page.evaluate("""() => {window.worker?.terminate();window.worker=new Worker('/ai-worker.js');let serial=0;window.rpc=(body)=>new Promise((resolve,reject)=>{const id=++serial;const timeout=setTimeout(()=>{worker.removeEventListener('message',receive);reject(Error('worker timeout'));},60000);function receive(e){if(e.data.id!==id)return;clearTimeout(timeout);worker.removeEventListener('message',receive);resolve(e.data);}worker.addEventListener('message',receive);worker.postMessage({...body,id});});}""")
    started=time.perf_counter();info=page.evaluate('(request)=>rpc(request)',{'type':'init','backend':backend,'modelVariant':a.model_variant})
    row={'initializationMs':(time.perf_counter()-started)*1000,**info};result['backends'][backend]=row
    if 'error' in info:print(json.dumps({'backend':backend,'error':info['error']}),flush=True);continue
    records=[]
    for batch in ([1] if a.model_variant in ('single','portable') else [1,8,32]):
     selected=[fixtures[i%len(fixtures)] for i in range(batch)]
     response=page.evaluate('(request)=>rpc(request)',{'type':'benchmark','features':[f['encoded_input'] for f in selected],'repeats':30})
     if 'error' in response:raise RuntimeError(response['error'])
     expected=[x for f in selected for x in f['native_policy_logits']]
     maxerr=max(abs(x-y) for x,y in zip(response.pop('logits'),expected))
     valueerr=max(abs(x-f['native_value']) for x,f in zip(response.pop('values'),selected))
     response['maxPolicyAbsoluteError']=maxerr;response['maxValueAbsoluteError']=valueerr;records.append(response)
     if maxerr>0.002 or valueerr>0.0002:raise AssertionError(f'{backend} model parity failed {maxerr}, {valueerr}')
    row['inference']=records
    if a.search:
     row['search']=[]
     for f in [fixtures[0],fixtures[3],fixtures[8],fixtures[-1]]:
      response=page.evaluate('(request)=>rpc(request)',{'type':'search','state':f['state'],'simulations':128,'budgetMs':10000})
      if 'error' in response:raise RuntimeError(response['error'])
      assert response['action'] in f['legal_actions']
      row['search'].append({'fixture':f['id'],'phase':f['phase'],**response})
    print(json.dumps({'backend':backend,'batch1Ms':records[0]['medianMs'],'searches':len(row.get('search',[]))}),flush=True)
   browser.close()
 finally:server.shutdown()
 target=ROOT/a.output;target.parent.mkdir(exist_ok=True,parents=True);target.write_text(json.dumps(result,indent=2)+'\n')
 print(str(target),flush=True)
if __name__=='__main__':main()
