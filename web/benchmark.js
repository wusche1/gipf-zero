const button=document.getElementById('run'), status=document.getElementById('status'), output=document.getElementById('result');
button.addEventListener('click',()=>{
 button.disabled=true;status.textContent='Loading the model and searching…';
 const worker=new Worker(new URL('./ai-worker.js',import.meta.url));
 const finish=(text)=>{clearTimeout(timer);worker.terminate();status.textContent=text;button.disabled=false;};
 const timer=setTimeout(()=>finish('Benchmark timed out.'),60000);
 worker.onerror=()=>finish('The benchmark could not start.');
 worker.onmessage=({data})=>{
  if(data.error)return finish(data.error);
  const {simulations,elapsedMs,evalsPerSec}=data.stats;
  output.hidden=false;output.textContent=JSON.stringify({browser:navigator.userAgent,model:data.model,simulations,searchMs:elapsedMs,positionsPerSecond:evalsPerSec,action:data.action},null,2);
  finish('Finished.');
 };
 worker.postMessage({id:1,type:'benchmark'});
});
