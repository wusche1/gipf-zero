const button=document.getElementById('run'), status=document.getElementById('status'), output=document.getElementById('result');
let worker,serial=0;
function rpc(body){return new Promise((resolve,reject)=>{const id=++serial;const timer=setTimeout(()=>{worker?.terminate();reject(Error('Benchmark timed out.'));},60000);const handler=({data})=>{if(data.id!==id)return;clearTimeout(timer);worker.removeEventListener('message',handler);data.error?reject(Error(data.error)):resolve(data);};worker.addEventListener('message',handler);worker.postMessage({id,...body});});}
button.addEventListener('click',async()=>{
 button.disabled=true;output.hidden=false;const report={browser:navigator.userAgent,results:{}};
 try{
  status.textContent='Loading the reference position…';
  const fixtures=await (await fetch('./models/champion-fixtures.json')).json();const fixture=fixtures.fixtures[0];
  for(const backend of ['wasm','webgpu']){
   status.textContent=`Testing ${backend==='wasm'?'CPU':'GPU'} inference and search…`;
   worker=new Worker(new URL('./ai-worker.js',import.meta.url));
   try{
    const init=await rpc({type:'init',backend,modelVariant:backend==='webgpu'?'portable':'dynamic'});
    const inference=await rpc({type:'benchmark',features:fixture.encoded_input,repeats:30});
    const search=await rpc({type:'search',state:fixture.state,simulations:128,budgetMs:10000});
    report.results[backend]={model:init.model,adapter:init.adapterInfo,inferenceMedianMs:inference.medianMs,searchMs:search.stats.elapsedMs,simulations:search.stats.simulations,action:search.action};
   }catch(error){report.results[backend]={unavailable:error.message};}
   finally{worker?.terminate();worker=null;output.textContent=JSON.stringify(report,null,2);}
  }
  status.textContent='Finished. The game uses the verified CPU path by default.';
 }catch(error){status.textContent=error.message;}
 finally{button.disabled=false;}
});
