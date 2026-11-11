/* Learned GIPF policy/value search, entirely on the visitor's device. */
'use strict';
let engine, model, initialization, tree, active;
let queue = Promise.resolve();
let stopBelow = 0, ponderStopBelow = 0;
const asset = (path) => new URL(path, self.location.href).href;
// Awaiting a search slice alone never returns to the event loop, so stop requests would
// sit unread; a message-channel hop lets them in without a timer clamp.
const channel = new MessageChannel();
const tick = () => new Promise((resolve) => { channel.port1.onmessage = resolve; channel.port2.postMessage(0); });

async function initialize() {
  if (initialization) return initialization;
  initialization = (async () => {
    importScripts(asset('gipf_engine.js'));
    // config.json names the model; models/<name>.json holds its weight file,
    // architecture, hash, search parameters and thinking budgets.
    const name = (await (await fetch(asset('config.json'), {cache:'no-store'})).json()).model;
    const response = await fetch(asset(`models/${name}.json`));
    if (!response.ok) throw new Error('Could not download the AI model metadata.');
    model = await response.json();
    const modelResponse = await fetch(asset(`models/${model.path}?v=${model.sha256}`));
    if (!modelResponse.ok) throw new Error('Could not download the AI model. Please retry.');
    const bytes = await modelResponse.arrayBuffer();
    if (self.crypto?.subtle) {
      const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
      if (hash !== model.sha256) throw new Error('AI model checksum mismatch. Please reload the page.');
    }
    engine = await createGipfEngine({locateFile:(name)=>asset(name)});
    if (!engine.load_kernel(new Float32Array(bytes), model.arch)) throw new Error('AI model does not match its metadata.');
    if (!Number.isFinite(engine.kernel_forward(new Float32Array(441))[2730])) throw new Error('AI model initialization returned an invalid value.');
    return {ready:true, model:model.name, sha256:model.sha256};
  })();
  return initialization;
}

// One search or ponder on the persistent tree. A position reached from the
// current root keeps its subtree; anything else (undo, new game) starts fresh.
async function search(request, ponder) {
  const info = await initialize();
  const job = active = {ponder, stop: request.id < (ponder ? Math.max(stopBelow, ponderStopBelow) : stopBelow)};
  if (job.stop) throw new Error('Search stopped.');
  const state = JSON.stringify(request.state);
  if (request.fresh || !tree?.reroot(state)) {
    tree?.delete(); tree = null;
    tree = new engine.BrowserSearch(state);
    tree.set_fpu(model.fpu); tree.set_cpuct(model.cpuct);
  }
  if (tree.result().terminal) throw new Error('This game is already over.');
  const reusedVisits = tree.root_visits();
  const started = performance.now();
  let simulations = 0, posted = started;
  const stats = () => {
    const elapsedMs = performance.now() - started, result = tree.result();
    return {...result, simulations, elapsedMs, evalsPerSec: result.evaluations / elapsedMs * 1000, reusedVisits, ponder};
  };
  tree.think(0, 0);
  // An explicit simulation count is an exact request: no time management.
  const exact = Number(request.simulations) || 0;
  const limit = Math.min(20000, exact || 10000);
  let budget = ponder ? model.ponder.maxMs : exact ? Infinity : Math.min(10000, Math.max(50, Number(request.budgetMs) || model.budgetsMs[request.level] || model.budgetsMs.normal));
  // Capture choices continue a turn whose push was already searched.
  if (!ponder && !exact && request.state.phase === 'capture') budget /= 4;
  const forced = !ponder && !exact && tree.choices() === 1;
  while (!forced && !job.stop && simulations < limit) {
    const now = performance.now(), elapsed = now - started;
    if (elapsed >= budget) break;
    if (ponder && tree.root_visits() >= model.ponder.maxVisits) break;
    // Stop once the runner-up cannot catch the leader in the time left.
    if (!ponder && !exact && simulations >= 32 && tree.margin() > simulations / elapsed * (budget - elapsed)) break;
    const room = ponder ? model.ponder.maxVisits - tree.root_visits() : limit;
    simulations += tree.think(Math.min(limit - simulations, room), Math.min(15, budget - elapsed));
    await tick();
    if (request.progress && now - posted > 150) { posted = now; self.postMessage({id:request.id, progress:true, stats:stats()}); }
  }
  const result = stats();
  return {...info, action:result.action, stats:result};
}

// The benchmark page: a fixed search from the opening position.
async function benchmark(request) {
  await initialize();
  const start = new engine.State(), state = start.serialize();
  start.delete();
  return search({...request, state, simulations:128, fresh:true}, false);
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
      if (data.type === 'init') response=await initialize();
      else if (data.type === 'search' || data.type === 'ponder') response=await search(data, data.type === 'ponder');
      else if (data.type === 'benchmark') response=await benchmark(data);
      else throw new Error('Unknown AI request.');
      self.postMessage({id:data.id,...response});
    } catch (error) { self.postMessage({id:data.id,error:error?.message||String(error)}); }
    active = null;
  });
};
