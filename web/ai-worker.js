/* Learned GIPF policy/value search, entirely on the visitor's device. */
'use strict';
let engine, session, model, backend, initialization, tree, active;
let queue = Promise.resolve();
let stopBelow = 0, ponderStopBelow = 0;
const asset = (path) => new URL(path, self.location.href).href;
const median = (xs) => [...xs].sort((a,b) => a-b)[Math.floor(xs.length/2)];
// Awaiting ONNX alone never returns to the event loop, so stop requests would
// sit unread; a message-channel hop lets them in without a timer clamp.
const channel = new MessageChannel();
const tick = () => new Promise((resolve) => { channel.port1.onmessage = resolve; channel.port2.postMessage(0); });

async function initialize(request = {}) {
  if (initialization) return initialization;
  initialization = (async () => {
    backend = request.backend === 'webgpu' ? 'webgpu' : 'wasm';
    let adapterInfo = null;
    if (backend === 'webgpu') {
      if (!self.navigator.gpu) throw new Error('WebGPU is not available in this browser.');
      const adapter = await self.navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter is available.');
      adapterInfo = adapter.info ? {vendor:adapter.info.vendor, architecture:adapter.info.architecture, device:adapter.info.device, description:adapter.info.description, isFallbackAdapter:adapter.info.isFallbackAdapter} : {};
    }
    importScripts(asset('gipf_engine.js'), asset(`vendor/onnxruntime/ort.${backend === 'webgpu' ? 'webgpu' : 'wasm'}.min.js`));
    ort.env.wasm.wasmPaths = asset('vendor/onnxruntime/');
    // Works on ordinary GitHub Pages: no SharedArrayBuffer or special headers.
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    // config.json names the model; models/<name>.json holds its file, hash,
    // search parameters and thinking budgets.
    const config = await (await fetch(asset('config.json'), {cache:'no-store'})).json();
    const response = await fetch(asset(`models/${config.model}.json`));
    if (!response.ok) throw new Error('Could not download the AI model metadata.');
    model = await response.json();
    const modelVariant = request.modelVariant || (backend === 'webgpu' ? 'portable' : 'dynamic');
    const modelSpec = modelVariant === 'dynamic' ? model : model.variants?.[modelVariant];
    if (!modelSpec || (backend === 'webgpu' && !model.fixtures)) throw new Error('Requested model variant is unavailable.');
    const modelResponse = await fetch(asset(`models/${modelSpec.path}?v=${modelSpec.sha256}`));
    if (!modelResponse.ok) throw new Error('Could not download the AI model. Please retry.');
    const bytes = await modelResponse.arrayBuffer();
    if (self.crypto?.subtle) {
      const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
      if (hash !== modelSpec.sha256) throw new Error('AI model checksum mismatch. Please reload the page.');
    }
    engine = await createGipfEngine({locateFile:(name)=>asset(name)});
    session = await ort.InferenceSession.create(bytes, {executionProviders:[backend], graphOptimizationLevel:'all'});
    const warmup = await forward(new Float32Array(441), 1);
    if (!Number.isFinite(warmup.value[0])) throw new Error('AI model initialization returned an invalid value.');
    // GPU implementations vary. Never play with a backend that fails the
    // model's known outputs, even if session creation succeeds.
    if (backend === 'webgpu') {
      const fixtureResponse = await fetch(asset(`models/${model.fixtures}`));
      if (!fixtureResponse.ok) throw new Error('Could not verify GPU inference.');
      const fixtures = (await fixtureResponse.json()).fixtures;
      for (const fixture of fixtures) {
        const actual = await forward(Float32Array.from(fixture.encoded_input.flat(Infinity)));
        const error = Math.max(...actual.logits.map((v,i)=>Math.abs(v-fixture.native_policy_logits[i])));
        if (!Number.isFinite(error) || !Number.isFinite(actual.value[0]) || error > 0.002 || Math.abs(actual.value[0]-fixture.native_value) > 0.0002)
          throw new Error(`GPU inference failed model verification (position ${fixture.id}). Use browser CPU mode.`);
      }
    }
    return {ready:true, model:model.name, backend, adapterInfo, modelVariant, sha256:modelSpec.sha256};
  })();
  return initialization;
}

async function forward(features, batch = 1) {
  const tensor = new ort.Tensor('float32', features, [batch,9,7,7]);
  const output = await session.run({board_features:tensor});
  try {
    const logits = output.policy_logits.data;
    const value = output.value.data;
    // Return owned arrays; ORT output tensors can be disposed immediately.
    return {logits:new Float32Array(logits), value:new Float32Array(value)};
  } finally {
    tensor.dispose();
    for (const t of Object.values(output)) t.dispose();
  }
}

async function evaluate() {
  const prediction = await forward(new Float32Array(tree.features()));
  tree.complete(prediction.logits, prediction.value[0]);
}

// One search or ponder on the persistent tree. A position reached from the
// current root keeps its subtree; anything else (undo, new game) starts fresh.
async function search(request, ponder) {
  const info = await initialize(request);
  const job = active = {ponder, stop: request.id < (ponder ? Math.max(stopBelow, ponderStopBelow) : stopBelow)};
  if (job.stop) throw new Error('Search stopped.');
  const state = JSON.stringify(request.state);
  if (request.fresh || !tree?.reroot(state)) {
    tree?.delete(); tree = null;
    tree = new engine.BrowserSearch(state);
    tree.set_fpu(model.fpu);
  }
  if (tree.result().terminal) throw new Error('This game is already over.');
  const reusedVisits = tree.root_visits();
  const started = performance.now();
  let simulations = 0, evaluations = 0, posted = started, yielded = started;
  const stats = () => {
    const elapsedMs = performance.now() - started;
    return {...tree.result(), simulations, evaluations, elapsedMs, evalsPerSec: evaluations / elapsedMs * 1000, reusedVisits, ponder};
  };
  if (!tree.choices() && tree.prepare()) { await evaluate(); evaluations++; }
  // An explicit simulation count is an exact request: no time management.
  const exact = Number(request.simulations) || 0;
  const limit = Math.min(20000, exact || 10000);
  let budget = ponder ? model.ponder.maxMs : Math.min(10000, Math.max(50, Number(request.budgetMs) || model.budgetsMs[request.level] || model.budgetsMs.normal));
  // Capture choices continue a turn whose push was already searched.
  if (!ponder && !exact && request.state.phase === 'capture') budget /= 4;
  const forced = !ponder && !exact && tree.choices() === 1;
  while (!forced && !job.stop && simulations < limit) {
    const now = performance.now(), elapsed = now - started;
    if (elapsed >= budget) break;
    if (ponder && tree.root_visits() >= model.ponder.maxVisits) break;
    // Stop once the runner-up cannot catch the leader in the time left.
    if (!ponder && !exact && simulations >= 32 && tree.margin() > simulations / elapsed * (budget - elapsed)) break;
    if (tree.step(model.cpuct)) { await evaluate(); evaluations++; }
    simulations++;
    if (now - yielded > 20) { yielded = now; await tick(); }
    if (request.progress && now - posted > 150) { posted = now; self.postMessage({id:request.id, progress:true, stats:stats()}); }
  }
  const result = stats();
  return {...info, action:result.action, stats:result};
}

async function benchmark(request) {
  const info = await initialize(request);
  const features = Float32Array.from(request.features.flat(Infinity));
  const batch = features.length / 441;
  if (!Number.isInteger(batch) || batch < 1 || batch > 128) throw new Error('Invalid inference batch.');
  for (let i=0;i<5;i++) await forward(features,batch);
  const times=[]; let output;
  const count=Math.min(100,Math.max(1,request.repeats||20));
  for(let i=0;i<count;i++){const start=performance.now();output=await forward(features,batch);times.push(performance.now()-start);}
  return {...info,batch,timesMs:times,medianMs:median(times),logits:Array.from(output.logits),values:Array.from(output.value)};
}

self.onmessage = ({data}) => {
  if (data.type === 'stop') {
    if (data.ponder) ponderStopBelow = data.id; else stopBelow = data.id;
    if (active && (active.ponder || !data.ponder)) active.stop = true;
    return;
  }
  // The human moved (or the page asked again): pondering always gives way.
  if (data.type === 'search' || data.type === 'ponder') {
    ponderStopBelow = data.id;
    if (active?.ponder) active.stop = true;
  }
  queue = queue.then(async () => {
    try {
      let response;
      if (data.type === 'init') response=await initialize(data);
      else if (data.type === 'search' || data.type === 'ponder') response=await search(data, data.type === 'ponder');
      else if (data.type === 'benchmark') response=await benchmark(data);
      else throw new Error('Unknown AI request.');
      self.postMessage({id:data.id,...response});
    } catch (error) { self.postMessage({id:data.id,error:error?.message||String(error)}); }
    active = null;
  });
};
