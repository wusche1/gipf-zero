/*
 * Boundary between the board view and the rules engine.
 *
 * The page looks for a module exposed as `window.GIPF_ENGINE` (or loaded from
 * config.json's engineUrl). Its methods may use either JS camelCase or the
 * snake_case names commonly emitted by a WASM bridge. Keeping that translation
 * here lets the view stay unaware of the engine's build format.
 */

const CENTER = 400;
const SCALE = 63;
const SQRT3 = Math.sqrt(3);
const SQRT3_2 = Math.sqrt(3) / 2;

export const geometry = buildGeometry();
let engine = null;
let engineSource = 'offline shell';

export async function loadEngine(config = {}) {
  engine = window.GIPF_ENGINE || window.gipfEngine || window.createGipfEngine || null;
  if (!engine && config.engineUrl) {
    try {
      const loaded = await import(config.engineUrl);
      const candidate = loaded.default || loaded.GIPF_ENGINE || loaded;
      // Emscripten MODULARIZE builds export an async factory rather than the
      // embind namespace itself.
      engine = typeof candidate === 'function' ? await candidate() : candidate;
      engineSource = 'WASM engine';
    } catch (error) {
      console.info('[gipf] engine unavailable; using offline shell', error);
    }
  } else if (engine) {
    if (typeof engine === 'function') engine = await engine();
    engineSource = 'WASM engine';
  }
  syncEngineGeometry();
  return { available: Boolean(engine), label: engineSource };
}

export function hasEngine() { return Boolean(engine); }
export function sourceLabel() { return engineSource; }

export function newGame(options = {}) {
  let result = callEngine(['newGame', 'new_game'], options);
  if (result === undefined && engine) {
    const State = engine.State || engine.state;
    if (typeof State === 'function') {
      try { result = new State(); } catch (error) { console.info('[gipf] State constructor unavailable', error); }
    }
  }
  return result === undefined ? makeOfflineState(options) : decode(result);
}

export function legalActions(state) {
  const result = callRule(state, ['legalActions', 'legal_actions', 'actions']);
  if (result !== undefined) return toArray(decode(result));
  // The opening can still be rendered without the artifact, but moves stay
  // disabled rather than pretending that a partial rules implementation is a game.
  return [];
}

export function applyAction(state, action) {
  const result = callRule(state, ['applyAction', 'apply_action', 'apply'], action);
  // The C++ contract mutates State in place and returns void. Preserve the
  // object in that case so the view can immediately read its new fields.
  if (result === undefined && engine && (typeof state?.apply === 'function' || typeof state?.apply_action === 'function')) return state;
  return result === undefined ? state : decode(result);
}

export function actionPointKey(action) {
  if (action == null) return null;
  if (typeof action === 'number') return geometry.rays[action]?.entryKey || null;
  return action.pointKey || action.point || action.target || action.destination || null;
}

function callEngine(methods, ...args) {
  if (!engine) return undefined;
  for (const method of methods) {
    if (typeof engine[method] === 'function') {
      try { return engine[method](...args); } catch (error) { console.warn(`[gipf] ${method} failed`, error); return undefined; }
    }
  }
  return undefined;
}

function callRule(state, methods, ...args) {
  for (const method of methods) {
    if (typeof state?.[method] === 'function') {
      try { return state[method](...args); } catch (error) { console.warn(`[gipf] state.${method} failed`, error); return undefined; }
    }
  }
  return callEngine(methods, state, ...args);
}

function decode(value) {
  if (typeof value !== 'string') return value;
  try { return JSON.parse(value); } catch { return value; }
}

function toArray(value) {
  if (value == null) return [];
  if (Array.isArray(value)) return value;
  if (typeof value[Symbol.iterator] === 'function') return Array.from(value);
  if (typeof value.size === 'function' && typeof value.get === 'function') return Array.from({ length: value.size() }, (_, index) => value.get(index));
  return value;
}

function syncEngineGeometry() {
  if (!engine || typeof engine.geometry !== 'function') return;
  try {
    const raw = engine.geometry();
    geometry.engine = raw;
    const rays = toArray(raw?.rays);
    const outer = new Map();
    rays.forEach((ray, index) => {
      const indices = toArray(ray);
      const a = geometry.nodes[Number(indices[0])];
      const b = geometry.nodes[Number(indices[1])];
      if (!a || !geometry.rays[index]) return;
      const source = b ? { x: a.x * 2 - b.x, y: a.y * 2 - b.y } : { x: a.x, y: a.y };
      geometry.rays[index].entryKey = a.key;
      geometry.rays[index].inner = { x: a.x, y: a.y };
      geometry.rays[index].outer = source;
      outer.set(`${Math.round(source.x)},${Math.round(source.y)}`, source);
    });
    geometry.outerDots = [...outer.values()];
  } catch (error) { console.info('[gipf] engine geometry unavailable', error); }
}

function makeOfflineState(options = {}) {
  const openingCorners = {
    '-3,0': { owner: 'white', kind: 'double' },
    '0,3': { owner: 'white', kind: 'double' },
    '3,-3': { owner: 'white', kind: 'double' },
    '0,-3': { owner: 'black', kind: 'double' },
    '3,0': { owner: 'black', kind: 'double' },
    '-3,3': { owner: 'black', kind: 'double' }
  };
  return {
    board: openingCorners,
    reserves: { white: 12, black: 12 },
    currentPlayer: options.firstPlayer || 'white',
    phase: 'push',
    moves: 0,
    pendingCapture: null,
    history: []
  };
}

function buildGeometry() {
  const nodes = [];
  const byKey = {};
  // This is also the engine's documented board-index order: r first, then q.
  for (let r = -3; r <= 3; r += 1) {
    for (let q = Math.max(-3, -r - 3); q <= Math.min(3, -r + 3); q += 1) {
      const s = -q - r;
      if (Math.max(Math.abs(q), Math.abs(r), Math.abs(s)) > 3) continue;
      const node = { q, r, key: `${q},${r}`, x: CENTER + SCALE * (q + r * .5), y: CENTER + SCALE * SQRT3_2 * r, rayIds: [] };
      nodes.push(node); byKey[node.key] = node;
    }
  }

  const sides = [];
  const rays = [];
  const outerDots = [];
  const sideVectors = [
    { x: 0, y: -1 }, { x: SQRT3 / 2, y: -.5 }, { x: SQRT3 / 2, y: .5 },
    { x: 0, y: 1 }, { x: -SQRT3 / 2, y: .5 }, { x: -SQRT3 / 2, y: -.5 }
  ];
  const tangentVectors = sideVectors.map(v => ({ x: -v.y, y: v.x }));
  const perimeterRadius = 282;
  const outerRadius = 347;
  for (let side = 0; side < 6; side += 1) {
    const radial = sideVectors[side];
    const tangent = tangentVectors[side];
    sides.push({ radial, tangent });
    for (let i = 0; i < 7; i += 1) {
      const offset = (i - 3) * 43;
      const outer = pointOnSide(radial, tangent, outerRadius, offset);
      const inner = pointOnSide(radial, tangent, perimeterRadius, offset);
      const rayId = side * 7 + i;
      const nearest = nodes.reduce((best, node) => distance(node, inner) < distance(best, inner) ? node : best, nodes[0]);
      rays.push({ rayId, side, index: i, outer, inner, entryKey: nearest.key });
      nearest.rayIds.push(rayId);
    }
    for (let i = 0; i < 4; i += 1) outerDots.push(pointOnSide(radial, tangent, outerRadius + 10, (i - 1.5) * 82));
  }

  const lines = [];
  const directions = [[1, 0], [0, 1], [1, -1]];
  for (const node of nodes) {
    for (const [dq, dr] of directions) {
      const next = byKey[`${node.q + dq},${node.r + dr}`];
      if (next) lines.push({ x1: node.x, y1: node.y, x2: next.x, y2: next.y });
    }
  }
  return { nodes, byKey, lines, rays, outerDots, sides };
}

function pointOnSide(radial, tangent, radius, offset) {
  return { x: CENTER + radial.x * radius + tangent.x * offset, y: CENTER + radial.y * radius + tangent.y * offset };
}
function distance(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
