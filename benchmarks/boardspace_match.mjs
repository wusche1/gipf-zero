// node benchmarks/boardspace_match.mjs --level bestbot --ms 300 --games 20 --seed 1
//   [--sims 1000] [--model web/models/latest.bin] [--config web/models/latest.json] [--cpuct X] [--fpu X]
//   [--opening-plies 2] [--max-ply 300] [--java path/to/java] [--classes dir]
// Build the BoardSpace side first: bash benchmarks/boardspace/build.sh (see benchmarks/boardspace/NOTES.md)
import {createRequire} from 'module';
import {readFileSync} from 'fs';
import {spawn} from 'child_process';
import {createInterface} from 'readline';
import {fileURLToPath} from 'url';

const repo = fileURLToPath(new URL('..', import.meta.url));
const createGipfEngine = createRequire(import.meta.url)('../web/gipf_engine.js');
const args = process.argv.slice(2);
const flag = (name, d) => { const i = args.indexOf(name); return i < 0 ? d : args[i + 1]; };
const games = +flag('--games', 20), seed = +flag('--seed', 1), maxPly = +flag('--max-ply', 300), level = flag('--level', 'bestbot');
const sims = +flag('--sims', 0), ms = +flag('--ms', 300), openingPlies = +flag('--opening-plies', 2);
const java = flag('--java', '/workspace/tmp/gipf-work/external/boardspace/jdk/bin/java'), classes = flag('--classes', '/dev/shm/boardspace/classes');
const strategy = {dumbot: 0, smartbot: 1, bestbot: 2, depth3: 3, weakbot: 4}[level];  // RobotProtocol.*_LEVEL
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

// BoardSpace names spots <column A..I><row from the bottom of that column, 1..>, edge spots included; ours is axial [q, r].
const {coordinates, lines, rays} = engine.geometry();
const name = ([q, r]) => 'abcdefghi'[q + 4] + (r - Math.max(-4, -q - 4) + 1);
const cellName = coordinates.map(name), cellIndex = Object.fromEntries(cellName.map((n, i) => [n, i]));
const bsOrder = [...cellName].sort();  // the bridge dumps b2..b5, c2..c6, ..., h2..h5
const rayName = rays.map((ray) => {
  const [q, r] = coordinates[ray[0]], [q1, r1] = coordinates[ray[1]];
  return `${name([2 * q - q1, 2 * r - r1])} ${cellName[ray[0]]}`;
});
const piece = {'.': 0, o: 1, O: 2, x: -1, X: -2}, colour = {w: 1, b: -1, '-': 0};

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
  return reply.split('|').map((raw) => {
    const f = raw.split(' ');
    return {
      raw, state: f[0], over: f[0] === 'GAMEOVER_STATE', decider: colour[f[1]],
      board: Object.fromEntries([...f[2]].map((c, i) => [cellIndex[bsOrder[i]], piece[c]])),
      reserves: [+f[3], +f[4]], captured: [+f[5], +f[6]], winner: colour[f[7]], ms: +f[8], did: f[9],
    };
  });
}

const captureLine = (a) => Math.floor((a - 42) / 128), captureMask = (a) => (a - 42) % 128;
const sameBoard = (o, g) => o.board.every((v, i) => v === g.board[i]) && o.reserves.join() === g.reserves.join() && o.captured.join() === g.captured.join();
const command = (action) => {
  if (action < 42) return `push ${rayName[action]}`;
  const line = lines[captureLine(action)];
  return ['capture', cellName[line[0]], cellName[line.at(-1)], ...line.filter((c, i) => captureMask(action) >> i & 1).map((c) => cellName[c])].join(' ');
};

// Both engines wait for the same player to make the same kind of decision in the same position.
function check(o, g) {
  if (!sameBoard(o, g)) return 'position';
  if (g.over) return 'game end';
  if (o.current_player !== g.decider) return 'side to decide';
  return (o.phase === 'push') === (g.state === 'PLACE_STATE') ? null : 'phase';
}

// The actions of ours with which `player` gets from `state` to a state accepted by `hit`.
function match(state, hit, player, depth = 4) {
  if (hit(state)) return [];
  if (!depth || state.winner || state.current_player !== player) return null;
  for (const action of legal(state)) {
    const next = state.clone();
    next.apply(action);
    const rest = match(next, hit, player, depth - 1);
    next.delete();
    if (rest) return [action, ...rest];
  }
  return null;
}

async function play(game, weWhite, opening, fresh) {
  const child = spawn(java, ['-Djava.awt.headless=true', '-cp', classes, 'gipf.BsBridge', `${strategy}`, `${seed * 1000 + game}`], {stdio: ['pipe', 'pipe', 'ignore']});
  const replies = createInterface({input: child.stdout})[Symbol.asyncIterator]();
  const ask = async (text) => { child.stdin.write(text + '\n'); return parse((await replies.next()).value); };
  const state = new engine.State(), us = weWhite ? 1 : -1, history = [], bs = {millis: 0, moves: 0};
  const record = {game, seed, level, weWhite, opening};
  const diverge = (problem, g) => { record.divergence = {problem, after: history.at(-1), ours: state.serialize(), boardspace: g.raw, history}; };
  try {
    let g = (await ask('dump'))[0], step = 0;
    for (;;) {
      if (state.winner) {
        // BoardSpace only looks for the end of the game when the next turn starts.
        if (!sameBoard(state.serialize(), g) && !g.over) { diverge('position', g); break; }
        if (!g.over) { g = (await ask('finish'))[0]; if (g.did.includes('Slide')) record.boardspaceEndedLater = g.did; }
        if (!g.over || g.winner !== state.winner) diverge('game end', g);
        break;
      }
      const problem = check(state.serialize(), g);
      if (problem) { diverge(problem, g); break; }
      if (state.ply >= maxPly || history.length >= 4 * maxPly) break;
      const inOpening = fresh ? state.ply < openingPlies : step < opening.length;
      if (inOpening || state.current_player === us) {
        const actions = legal(state), actor = state.current_player;
        const action = !inOpening ? decide(state.serialize()) : fresh ? actions[rand(actions.length)] : opening[step];
        if (inOpening && fresh) opening.push(action);
        step++;
        history.push(command(action));
        state.apply(action);
        g = (await ask(command(action)))[0];
        const o = state.serialize();
        if (!o.winner && o.turn_player === actor && !(o.phase === 'capture' && o.current_player === actor)) g = (await ask('done'))[0];
      } else {
        const dumps = await ask('go'), them = state.current_player;
        g = dumps.at(-1);
        bs.millis += g.ms; bs.moves++;
        for (const d of dumps) {
          const path = match(state, (t) => sameBoard(t.serialize(), d), them) ?? (g.over ? match(state, (t) => t.winner !== 0, them) : null);
          if (!path) { diverge('BoardSpace turn has no counterpart in our rules', d); break; }
          for (const action of path) { history.push(`${command(action)} [${g.did}]`); state.apply(action); }
          if (state.winner) break;
        }
        if (record.divergence) break;
      }
    }
  } catch (error) {
    record.divergence = {problem: String(error.message), after: history.at(-1), ours: state.serialize(), history};
  }
  child.kill();
  Object.assign(record, {plies: state.ply, winner: state.winner, result: record.divergence ? null : state.winner * us, boardspaceSecPerMove: bs.millis / bs.moves / 1000, checkedPositions: history.length});
  state.delete();
  return {record, bs};
}

const tally = {wins: 0, losses: 0, draws: 0, aborted: 0}, bsTotal = {millis: 0, moves: 0};
let opening;
for (let game = 0; game < games; game++) {
  const weWhite = game % 2 === 0, before = {...ours};
  if (weWhite) opening = [];
  const {record, bs} = await play(game, weWhite, opening, weWhite);
  const n = ours.n - before.n;
  Object.assign(record, {ourMsPerDecision: (ours.millis - before.millis) / n, ourSimsPerDecision: (ours.simulations - before.simulations) / n});
  if (record.divergence) console.error(`RULE DIVERGENCE in game ${game}: ${record.divergence.problem} after "${record.divergence.after}"`);
  tally[record.divergence ? 'aborted' : record.result > 0 ? 'wins' : record.result < 0 ? 'losses' : 'draws']++;
  bsTotal.millis += bs.millis; bsTotal.moves += bs.moves;
  console.log(JSON.stringify(record));
}
const n = tally.wins + tally.losses + tally.draws, score = (tally.wins + tally.draws / 2) / n;
const sd = Math.sqrt((tally.wins * (1 - score) ** 2 + tally.losses * score ** 2 + tally.draws * (0.5 - score) ** 2) / n / n);
console.log(JSON.stringify({summary: true, seed, level, budget: sims ? `${sims} sims` : `${ms} ms`, games: n, ...tally, score, ci95: [score - 1.96 * sd, score + 1.96 * sd],
  ourSimsPerDecision: ours.simulations / ours.n, ourMsPerDecision: ours.millis / ours.n, boardspaceSecPerMove: bsTotal.millis / bsTotal.moves / 1000}));
