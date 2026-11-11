// node benchmarks/wasm_arena.mjs --a bin=X.bin,arch=X.json,int16=1,cpuct=1,fpu=0.1,sims=800 --b bin=...,ms=300 --games 100 --seed 1
import {createRequire} from 'module';
import {readFileSync} from 'fs';

const createGipfEngine = createRequire(import.meta.url)('../web/gipf_engine.js');
const args = process.argv.slice(2);
const flag = (name, d) => { const i = args.indexOf(name); return i < 0 ? d : args[i + 1]; };
const games = +flag('--games', 100), seed = +flag('--seed', 1), maxPly = +flag('--max-ply', 300);

let s = seed >>> 0;
const rand = (n) => {
  s = (s + 0x6D2B79F5) >>> 0;
  let t = Math.imul(s ^ (s >>> 15), 1 | s);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return Math.floor(((t ^ (t >>> 14)) >>> 0) / 4294967296 * n);
};
const legal = (state) => { const v = state.legal_actions(), out = []; for (let i = 0; i < v.size(); i++) out.push(v.get(i)); v.delete(); return out; };

async function player(spec) {
  const p = Object.fromEntries(spec.split(',').map((kv) => kv.split('=')));
  const engine = await createGipfEngine(), w = readFileSync(p.bin);
  const arch = {...JSON.parse(readFileSync(p.arch)), int16: !!+(p.int16 || 0)};
  if (!engine.load_kernel(new Float32Array(w.buffer, w.byteOffset, w.length / 4), arch)) throw new Error(`weights do not fit arch: ${spec}`);
  return {spec, engine, cpuct: +p.cpuct, fpu: +p.fpu, sims: +(p.sims || 0), ms: +(p.ms || 0), n: 0, simulations: 0, millis: 0};
}

// One decision the way web/ai-worker.js search() takes it, on a fresh tree.
function decide(p, state) {
  const tree = new p.engine.BrowserSearch(JSON.stringify(state));
  tree.set_fpu(p.fpu); tree.set_cpuct(p.cpuct);
  const started = performance.now();
  let simulations = 0;
  if (p.sims) simulations = tree.think(p.sims, Infinity);
  else {
    const budget = state.phase === 'capture' ? p.ms / 4 : p.ms;
    tree.think(0, 0);
    for (;;) {
      const elapsed = performance.now() - started;
      if (elapsed >= budget || simulations >= 10000) break;
      if (simulations >= 32 && tree.margin() > simulations / elapsed * (budget - elapsed)) break;
      simulations += tree.think(10000 - simulations, Math.min(15, budget - elapsed));
    }
  }
  p.millis += performance.now() - started; p.simulations += simulations; p.n++;
  const action = tree.result().action;
  tree.delete();
  return action;
}

const A = await player(flag('--a')), B = await player(flag('--b'));
const tally = {wins: 0, losses: 0, draws: 0};
let opening;
for (let g = 0; g < games; g++) {
  const state = new A.engine.State(), aWhite = g % 2 === 0;
  if (aWhite) {
    opening = [];
    while (state.ply < 4 || state.phase === 'capture') { const a = legal(state); opening.push(a[rand(a.length)]); state.apply(opening.at(-1)); }
  } else for (const a of opening) state.apply(a);
  while (!state.winner && state.ply < maxPly) {
    const actions = legal(state);
    state.apply(actions.length === 1 ? actions[0] : decide((state.current_player === 1) === aWhite ? A : B, state.serialize()));
  }
  const result = state.winner * (aWhite ? 1 : -1);
  tally[result > 0 ? 'wins' : result < 0 ? 'losses' : 'draws']++;
  console.log(JSON.stringify({game: g, seed, aWhite, opening, plies: state.ply, winner: state.winner, result}));
  state.delete();
}
const n = tally.wins + tally.losses + tally.draws, score = (tally.wins + tally.draws / 2) / n;
const sd = Math.sqrt((tally.wins * (1 - score) ** 2 + tally.losses * score ** 2 + tally.draws * (0.5 - score) ** 2) / n / n);
const stats = (p) => ({spec: p.spec, decisions: p.n, simsPerDecision: p.simulations / p.n, msPerDecision: p.millis / p.n, msPerSim: p.millis / p.simulations});
console.log(JSON.stringify({summary: true, seed, games: n, ...tally, score, ci95: [score - 1.96 * sd, score + 1.96 * sd], a: stats(A), b: stats(B)}));
