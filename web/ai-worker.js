/* Learned GIPF policy/value search, entirely on the visitor's device. */
'use strict';
let engine, session, metadata, backend, initialization;
let queue = Promise.resolve();
const asset = (path) => new URL(path, self.location.href).href;
const median = (xs) => [...xs].sort((a,b) => a-b)[Math.floor(xs.length/2)];

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
    const response = await fetch(asset('models/champion-metadata.json'));
    if (!response.ok) throw new Error('Could not download the AI model metadata.');
    metadata = await response.json();
    const modelVariant = request.modelVariant || (backend === 'webgpu' ? 'portable' : 'dynamic');
    const modelSpec = modelVariant === 'single' ? metadata.onnx_single
      : modelVariant === 'portable' ? metadata.onnx_portable : metadata.onnx;
    if (!modelSpec) throw new Error('Requested model variant is unavailable.');
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
    // original champion's known outputs, even if session creation succeeds.
    if (backend === 'webgpu') {
      const fixtureResponse = await fetch(asset('models/champion-fixtures.json'));
      if (!fixtureResponse.ok) throw new Error('Could not verify GPU inference.');
      const fixtures = (await fixtureResponse.json()).fixtures;
      for (const fixture of fixtures) {
        const actual = await forward(Float32Array.from(fixture.encoded_input.flat(Infinity)));
        const error = Math.max(...actual.logits.map((v,i)=>Math.abs(v-fixture.native_policy_logits[i])));
        if (!Number.isFinite(error) || !Number.isFinite(actual.value[0]) || error > 0.002 || Math.abs(actual.value[0]-fixture.native_value) > 0.0002)
          throw new Error(`GPU inference failed champion verification (position ${fixture.id}). Use browser CPU mode.`);
      }
    }
    return {ready:true, model:`GIPF Zero — ${Number(metadata.games).toLocaleString('en-US')} games`, backend, adapterInfo, modelVariant, checkpointSha256:metadata.checkpoint_sha256};
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

async function search(request) {
  const info = await initialize(request);
  const budget = Math.min(10000, Math.max(50, Number(request.budgetMs) || 1500));
  const limit = Math.min(20000, Math.max(1, Number(request.simulations) || 10000));
  const started = performance.now();
  const tree = new engine.BrowserSearch(JSON.stringify(request.state));
  let evaluations = 0, simulations = 0;
  try {
    if (tree.result().terminal) throw new Error('This game is already over.');
    let pending = tree.prepare();
    if (pending) {const prediction=await forward(new Float32Array(tree.features()));tree.complete(prediction.logits, prediction.value[0]);evaluations++;}
    while (simulations < limit && performance.now()-started < budget) {
      pending = tree.step(1.5);
      if (pending) {const prediction=await forward(new Float32Array(tree.features()));tree.complete(prediction.logits,prediction.value[0]);evaluations++;}
      simulations++;
    }
    const result = tree.result();
    return {...info, action:result.action, stats:{...result,simulations,evaluations,elapsedMs:performance.now()-started}};
  } finally { tree.delete(); }
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
  queue = queue.then(async () => {
    try {
      let response;
      if (data.type === 'init') response=await initialize(data);
      else if (data.type === 'search') response=await search(data);
      else if (data.type === 'benchmark') response=await benchmark(data);
      else throw new Error('Unknown AI request.');
      self.postMessage({id:data.id,...response});
    } catch (error) { self.postMessage({id:data.id,error:error?.message||String(error)}); }
  });
};
