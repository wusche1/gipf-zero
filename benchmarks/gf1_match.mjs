// node benchmarks/gf1_match.mjs --depth 4 --ms 300 --games 20 --seed 1
//   [--sims 1000] [--model web/models/latest.bin] [--config web/models/latest.json] [--cpuct X] [--fpu X]
//   [--opening-plies 2] [--randomchoose 1] [--max-ply 300] [--bridge path/to/gf1bridge]
// Build the GF1 side first: benchmarks/gf1/build.sh
import {createRequire} from 'module';
import {readFileSync} from 'fs';
import {spawn} from 'child_process';
import {createInterface} from 'readline';
import {fileURLToPath} from 'url';

const repo = fileURLToPath(new URL('..', import.meta.url));
const createGipfEngine = createRequire(import.meta.url)('../web/gipf_engine.js');
const args = process.argv.slice(2);
const flag = (name, d) => { const i = args.indexOf(name); return i < 0 ? d : args[i + 1]; };
const games = +flag('--games', 20), seed = +flag('--seed', 1), maxPly = +flag('--max-ply', 300), depth = +flag('--depth', 3);
const sims = +flag('--sims', 0), ms = +flag('--ms', 300), openingPlies = +flag('--opening-plies', 2), randomChoose = flag('--randomchoose', '1');
const bridge = flag('--bridge', '/workspace/tmp/gipf-work/external/gf1-harness/gf1bridge');
const config = JSON.parse(readFileSync(flag('--config', repo + 'web/models/latest.json')));
const cpuct = +flag('--cpuct', config.cpuct), fpu = +flag('--fpu', config.fpu);

const engine = await createGipfEngine(), weights = readFileSync(flag('--model', repo + 'web/models/latest.bin'));
if (!engine.load_kernel(new Float32Array(weights.buffer, weights.byteOffset, weights.length / 4), config.arch)) throw new Error('weights do not fit arch');

let s = seed >>> 0;
const rand = (n) => {
  s = (s + 0x6D2B79F5) >>> 0;
  let t = Math.imul(s ^ (s >>> 15), 1 | s);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return Math.floor(((t ^ (t >>> 14)) >>> 0) / 4294967296 * n);
};
const legal = (state) => { const v = state.legal_actions(), out = []; for (let i = 0; i < v.size(); i++) out.push(v.get(i)); v.delete(); return out; };

// GF1 names points <column a..i><row from the bottom of that column, 1..>, dots included; ours is axial [q, r].
const {coordinates, lines, rays} = engine.geometry();
const name = ([q, r]) => 'abcdefghi'[q + 4] + (r - Math.max(-4, -q - 4) + 1);
const cellName = coordinates.map(name), cellIndex = Object.fromEntries(cellName.map((n, i) => [n, i]));
const gf1Order = [...cellName].sort();  // the bridge dumps b2..b5, c2..c6, ..., h2..h5
const rayName = rays.map((ray) => {
  const [q, r] = coordinates[ray[0]], [q1, r1] = coordinates[ray[1]];
  return `${name([2 * q - q1, 2 * r - r1])}-${cellName[ray[0]]}`;
});
const lineOf = (a, b) => lines.findIndex((line) => line.includes(cellIndex[a]) && line.includes(cellIndex[b]));
const piece = {'.': 0, o: 1, O: 2, x: -1, X: -2}, colour = {o: 1, x: -1, '.': 0};

// One decision the way web/ai-worker.js search() takes it, on a fresh tree.
const ours = {n: 0, simulations: 0, millis: 0};
function decide(state) {
  const tree = new engine.BrowserSearch(JSON.stringify(state));
  tree.set_fpu(fpu); tree.set_cpuct(cpuct);
  const started = performance.now();
  let simulations = 0;
  if (sims) simulations = tree.think(sims, Infinity);
  else {
    const budget = state.phase === 'capture' ? ms / 4 : ms;
    tree.think(0, 0);
    for (;;) {
      const elapsed = performance.now() - started;
      if (elapsed >= budget || simulations >= 10000) break;
      if (simulations >= 32 && tree.margin() > simulations / elapsed * (budget - elapsed)) break;
      simulations += tree.think(10000 - simulations, Math.min(15, budget - elapsed));
    }
  }
  ours.millis += performance.now() - started; ours.simulations += simulations; ours.n++;
  const action = tree.result().action;
  tree.delete();
  return action;
}

function parse(reply) {
  if (!reply || reply.startsWith('ERR')) throw new Error(`bridge: ${reply}`);
  const [did, st, pending] = reply.split(';'), f = st.split(' ');
  return {
    raw: reply, did: did.trim().split(' ').filter(Boolean), status: f[0], decider: colour[f[1]],
    board: Object.fromEntries([...f[2]].map((c, i) => [cellIndex[gf1Order[i]], piece[c]])),
    reserves: [+f[3], +f[4]], captured: [+f[5], +f[6]], winner: colour[f[7]], ms: +f[8], pending: pending.trim().split(' ').filter(Boolean),
  };
}

const captureLine = (a) => Math.floor((a - 42) / 128), captureMask = (a) => (a - 42) % 128;
const sameBoard = (o, g) => o.board.every((v, i) => v === g.board[i]) && o.reserves.join() === g.reserves.join() && o.captured.join() === g.captured.join();

// Brings our state to the point GF1 is at (GF1 removes a lone row without doubles by itself) and compares everything.
function check(state, g) {
  while (!state.winner && state.phase === 'capture' && legal(state).length === 1) state.apply(legal(state)[0]);
  const o = state.serialize();
  if (o.winner && 'RG'.includes(g.status)) return 'game end: we stop as soon as a colour has no GIPF piece left, GF1 first finishes the pending captures';
  if (o.winner) return g.status === 'F' && g.winner === o.winner && (o.phase === 'capture' || sameBoard(o, g)) ? null : 'game end';
  if (g.status === 'F') return 'game end';
  if (o.current_player !== g.decider) return 'side to decide';
  if (o.phase === 'push') return g.status === 'N' && sameBoard(o, g) ? null : 'position';
  const actions = legal(state), rowLines = [...new Set(actions.map(captureLine))];
  if (rowLines.length > 1) {
    const theirs = g.pending.map((t) => lineOf(t.slice(2, 4), t.slice(5, 7))).sort((a, b) => a - b);
    return g.status === 'R' && sameBoard(o, g) && theirs.join() === rowLines.join() ? null : 'rows to choose from';
  }
  const kept = state.clone();
  kept.apply(42 + rowLines[0] * 128);
  const k = kept.serialize(), doubles = lines[rowLines[0]].filter((c, i) => Math.max(...actions.map(captureMask)) >> i & 1).map((c) => cellName[c]).sort();
  kept.delete();
  return g.status === 'G' && sameBoard(k, g) && g.pending.map((t) => t.slice(2)).sort().join() === doubles.join() ? null : 'optional GIPF pieces';
}

const gipfTokens = (line, doubles, mask) => lines[line].flatMap((c, i) => doubles >> i & 1 ? [`g:${cellName[c]}=${mask >> i & 1 ? 'y' : 'n'}`] : []);
function toTokens(state, action, g) {
  if (action < 42) return [rayName[action]];
  const line = captureLine(action), doubles = legal(state).filter((a) => captureLine(a) === line).reduce((m, a) => m | captureMask(a), 0);
  const row = g.pending.find((t) => t[0] === 'r' && lineOf(t.slice(2, 4), t.slice(5, 7)) === line);
  return [...(row ? [row] : []), ...gipfTokens(line, doubles, captureMask(action))];
}
function toAction(state, tokens) {
  if (tokens[0][1] !== ':') return rayName.indexOf(tokens[0]);
  const line = tokens[0][0] === 'r' ? lineOf(tokens[0].slice(2, 4), tokens[0].slice(5, 7)) : captureLine(legal(state)[0]);
  const mask = tokens.filter((t) => t.endsWith('=y')).reduce((m, t) => m | 1 << lines[line].indexOf(cellIndex[t.slice(2, 4)]), 0);
  return 42 + line * 128 + mask;
}

async function play(game, weWhite, opening, fresh) {
  const child = spawn(bridge, [weWhite ? 'x' : 'o', `${depth}`, `${seed * 1000 + game}`, randomChoose], {stdio: ['pipe', 'pipe', 'inherit']});
  const replies = createInterface({input: child.stdout})[Symbol.asyncIterator]();
  const ask = async (command) => { child.stdin.write(command + '\n'); return parse((await replies.next()).value); };
  const state = new engine.State(), us = weWhite ? 1 : -1, history = [], gf1 = {millis: 0, moves: 0};
  const record = {game, seed, depth, weWhite, opening};
  try {
    let g = await ask('dump'), step = 0;
    for (;;) {
      const problem = check(state, g);
      if (problem) { record.divergence = {problem, after: history.at(-1), ours: state.serialize(), gf1: g.raw, history}; break; }
      if (state.winner || state.ply >= maxPly) break;
      const inOpening = fresh ? state.ply < openingPlies : step < opening.length;
      let tokens;
      if (inOpening || state.current_player === us) {
        const actions = legal(state);
        const action = !inOpening ? decide(state.serialize()) : fresh ? actions[rand(actions.length)] : opening[step];
        if (inOpening && fresh) opening.push(action);
        step++;
        tokens = toTokens(state, action, g);
        state.apply(action);
        g = await ask('play ' + tokens.join(' '));
      } else {
        g = await ask('go');
        tokens = g.did; gf1.millis += g.ms; gf1.moves += state.phase === 'push';
        const action = toAction(state, tokens);
        if (!legal(state).includes(action)) { record.divergence = {problem: 'GF1 decision is illegal for us', after: tokens.join(' '), ours: state.serialize(), gf1: g.raw, history}; break; }
        state.apply(action);
      }
      history.push(tokens.join(' '));
    }
  } catch (error) {
    record.divergence = {problem: String(error.message), after: history.at(-1), ours: state.serialize(), history};
  }
  child.kill();
  Object.assign(record, {plies: state.ply, winner: state.winner, result: record.divergence ? null : state.winner * us, gf1MsPerMove: gf1.millis / gf1.moves});
  state.delete();
  return {record, gf1};
}

const tally = {wins: 0, losses: 0, draws: 0, aborted: 0}, gf1Total = {millis: 0, moves: 0};
let opening;
for (let game = 0; game < games; game++) {
  const weWhite = game % 2 === 0, before = {...ours};
  if (weWhite) opening = [];
  const {record, gf1} = await play(game, weWhite, opening, weWhite);
  const n = ours.n - before.n;
  Object.assign(record, {ourMsPerDecision: (ours.millis - before.millis) / n, ourSimsPerDecision: (ours.simulations - before.simulations) / n});
  if (record.divergence) console.error(`RULE DIVERGENCE in game ${game}: ${record.divergence.problem} after "${record.divergence.after}"`);
  tally[record.divergence ? 'aborted' : record.result > 0 ? 'wins' : record.result < 0 ? 'losses' : 'draws']++;
  gf1Total.millis += gf1.millis; gf1Total.moves += gf1.moves;
  console.log(JSON.stringify(record));
}
const n = tally.wins + tally.losses + tally.draws, score = (tally.wins + tally.draws / 2) / n;
const sd = Math.sqrt((tally.wins * (1 - score) ** 2 + tally.losses * score ** 2 + tally.draws * (0.5 - score) ** 2) / n / n);
console.log(JSON.stringify({summary: true, seed, depth, budget: sims ? `${sims} sims` : `${ms} ms`, games: n, ...tally, score, ci95: [score - 1.96 * sd, score + 1.96 * sd],
  ourSimsPerDecision: ours.simulations / ours.n, ourMsPerDecision: ours.millis / ours.n, gf1MsPerMove: gf1Total.millis / gf1Total.moves}));
