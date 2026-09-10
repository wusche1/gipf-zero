import * as game from './game.js';
import { createLocalAiClient } from './local-ai.js';
import { registerOffline } from './offline.js';

const offlineRegistration = registerOffline();

const $ = (id) => document.getElementById(id);
const svgNS = 'http://www.w3.org/2000/svg';
const els = {
  board: $('board'), base: $('board-base'), rays: $('ray-layer'), grid: $('grid-layer'), captureLayer: $('capture-layer'), pieces: $('piece-layer'), removals: $('removal-layer'), outer: $('outer-layer'), interaction: $('interaction-layer'),
  status: $('engine-status'), phaseTitle: $('phase-title'), phaseBadge: $('phase-badge'), hint: $('board-hint'), round: $('round-number'), move: $('move-count'), explanation: $('explanation'),
  reserveWhite: $('reserve-white'), reserveBlack: $('reserve-black'), barWhite: $('reserve-bar-white'), barBlack: $('reserve-bar-black'), turnWhite: $('turn-white'), turnBlack: $('turn-black'), toast: $('toast'),
  capture: $('capture-card'), captureOptions: $('capture-options'), captureToggles: $('capture-toggles'), confirmCapture: $('confirm-capture'), mode: $('mode-select'), aiColor: $('ai-color'), difficulty: $('difficulty'), aiStatus: $('ai-status'),
  undo: $('undo')
};

let config = { aiMode: 'browser', engineUrl: '' };
let state;
let past = [];
let legal = [];
let legalByPoint = new Map();
let selectedCapture = null;
let aiThinking = false;
let aiBlocked = false;
let gameRevision = 0;
let aiModelName = '';
let toastTimer;
let activeMotion = null;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const requestedAiBackend = new URLSearchParams(window.location.search).get('aiBackend');
const localAi = createLocalAiClient({
  backend: requestedAiBackend === 'webgpu' ? 'webgpu' : 'wasm',
  onStatus(status, detail) {
    if (status === 'loading') { els.aiStatus.textContent = 'Loading local model…'; $('retry-ai').hidden = true; }
    if (status === 'ready') {
      aiModelName = detail?.model || aiModelName;
      els.aiStatus.textContent = aiModelName ? `${aiModelName} ready` : 'Local model ready';
      $('retry-ai').hidden = true;
    }
    if (status === 'error') { els.aiStatus.textContent = 'Local model unavailable'; $('retry-ai').hidden = false; }
    if (status === 'cancelled') els.aiStatus.textContent = aiModelName ? `${aiModelName} ready` : 'Local model ready when needed';
  }
});

start();

async function start() {
  try {
    const response = await fetch('./config.json', { cache: 'no-store' });
    if (response.ok) config = { ...config, ...(await response.json()) };
  } catch { /* GitHub Pages can still run from the static shell. */ }
  const engineInfo = await game.loadEngine(config);
  els.status.textContent = engineInfo.available ? 'Ready to play' : 'Rules engine unavailable';
  if (engineInfo.available) document.querySelector('.status-dot').style.background = '#618d72';
  state = game.newGame();
  drawBoardGeometry();
  bindControls();
  els.aiStatus.textContent = 'Local model ready when needed';
  render();
}

function drawBoardGeometry() {
  const g = game.geometry;
  els.base.appendChild(svg('polygon', { points: hexPoints(313), class: 'board-surface' }));
  els.base.appendChild(svg('polygon', { points: hexPoints(293), class: 'board-inner' }));
  for (const line of g.lines) els.grid.appendChild(svg('line', { ...line, class: 'grid-line' }));
  for (const ray of g.rays) {
    const arrow = svg('line', { x1: ray.outer.x, y1: ray.outer.y, x2: ray.inner.x, y2: ray.inner.y, class: 'ray-line', 'data-ray-id': ray.rayId, tabindex: 0 });
    const hit = rayHitTarget(ray);
    arrow.addEventListener('click', () => chooseRay(ray.rayId));
    arrow.addEventListener('mouseenter', () => showPreviewRay(ray.rayId));
    arrow.addEventListener('mouseleave', clearPreview);
    hit.addEventListener('click', () => chooseRay(ray.rayId));
    hit.addEventListener('mouseenter', () => showPreviewRay(ray.rayId));
    hit.addEventListener('mouseleave', clearPreview);
    els.rays.append(arrow, hit);
  }
  for (const point of g.outerDots) els.outer.appendChild(svg('circle', { cx: point.x, cy: point.y, r: 4, class: 'outer-dot' }));
  for (const node of g.nodes) {
    const dot = svg('circle', { cx: node.x, cy: node.y, r: 5.6, class: 'grid-node', 'data-key': node.key, tabindex: 0, role: 'button', 'aria-label': `Point ${node.key}` });
    dot.addEventListener('click', () => choosePoint(node.key));
    dot.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') choosePoint(node.key); });
    dot.addEventListener('mouseenter', () => showPreview(node.key));
    dot.addEventListener('mouseleave', clearPreview);
    els.grid.appendChild(dot);
  }
}

function render(transition = null) {
  cancelMotion();
  const actions = game.legalActions(state) || [];
  legal = actions;
  legalByPoint = new Map();
  for (const action of actions) {
    const point = game.actionPointKey(action) || game.geometry.rays.find(ray => ray.rayId === action.rayId)?.entryKey;
    if (point) legalByPoint.set(point, [...(legalByPoint.get(point) || []), action]);
  }
  const numericPushes = actions.some(action => typeof action === 'number' && action < 42);
  for (const dot of els.grid.querySelectorAll('.grid-node')) {
    const open = !numericPushes && legalByPoint.has(dot.dataset.key) && !readBoard(state)[dot.dataset.key];
    dot.classList.toggle('open', Boolean(open));
    dot.setAttribute('aria-disabled', String(!open));
  }
  renderMeta();
  renderCapture();
  renderCaptureHighlight();
  renderPieces();
  if (transition) animateBoardChange(transition);
  els.undo.disabled = !canUndo();
  if (winnerOf(state) !== 0) {
    aiThinking = false;
    aiBlocked = true;
    els.capture.hidden = true;
  }
  maybeAiMove();
}

function renderPieces() {
  els.pieces.replaceChildren();
  const board = readBoard(state);
  const captureInfo = selectedCaptureInfo();
  for (const [key, value] of Object.entries(board)) {
    const node = game.geometry.byKey[key];
    if (!node || !value) continue;
    const owner = normaliseOwner(value.owner || value.player || value.color);
    const kind = value.kind || (value.double || value.isGipf ? 'double' : 'single');
    const capturePart = captureInfo?.parts.find(part => part.key === key);
    const captureClass = kind === 'double' && capturePart ? (capturePart.masked ? ' capture-remove' : ' capture-keep') : '';
    const stone = svg('g', { class: 'stone', 'data-key': key });
    pieceElements({ x: node.x, y: node.y, owner, kind, captureClass, capturePart }).forEach(element => stone.appendChild(element));
    els.pieces.appendChild(stone);
  }
}

function cancelMotion() {
  const previous = activeMotion;
  activeMotion = null;
  previous?.animations.forEach(animation => animation.cancel());
  els.removals.replaceChildren();
  els.board.setAttribute('aria-busy', 'false');
}

function animateBoardChange({ before, action, previousStones, actor, player }) {
  if (reducedMotion.matches) return;
  const after = readBoard(state);
  const signature = value => value ? `${normaliseOwner(value.owner || value.player || value.color)}:${value.kind || (value.double || value.isGipf ? 'double' : 'single')}` : '';
  const keys = new Set([...Object.keys(before), ...Object.keys(after)]);
  if (![...keys].some(key => signature(before[key]) !== signature(after[key]))) return;

  const animations = [];
  const stones = new Map([...els.pieces.children].map(stone => [stone.dataset.key, stone]));
  const rayId = typeof action === 'number' && action < 42 ? action : action?.rayId;
  const ray = game.geometry.rays.find(candidate => candidate.rayId === rayId);
  if (ray) {
    const indices = arrayLike(arrayLike(game.geometry.engine?.rays)?.[rayId]);
    const nodes = indices?.map(index => game.geometry.nodes[Number(index)]).filter(Boolean) || [];
    let from = ray.outer;
    // Every stone in the occupied prefix travels one step, including adjacent
    // stones of the same color. Comparing colors alone cannot track that motion.
    for (const node of nodes) {
      const stone = stones.get(node.key);
      if (stone) animations.push(stone.animate([
        { transform: `translate(${from.x - node.x}px, ${from.y - node.y}px)` },
        { transform: 'translate(0px, 0px)' }
      ], { duration: 440, easing: 'cubic-bezier(.22, .61, .36, 1)' }));
      if (!before[node.key]) break;
      from = node;
    }
  }

  const removed = Object.entries(before).filter(([key, value]) => value && !after[key]);
  const removalTiming = { delay: actor === 'ai' ? 650 : 180, duration: 420, easing: 'cubic-bezier(.4, 0, 1, 1)', fill: 'forwards' };
  if (removed.length && typeof action === 'number' && action >= 42) {
    const lineId = Math.floor((action - 42) / 128);
    const indices = arrayLike(arrayLike(game.geometry.engine?.lines)?.[lineId]);
    const nodes = indices?.map(index => game.geometry.nodes[Number(index)]).filter(Boolean) || [];
    if (nodes.length > 1) {
      const from = nodes[0];
      const to = nodes[nodes.length - 1];
      const line = svg('line', { x1: from.x, y1: from.y, x2: to.x, y2: to.y, class: 'removal-row-line' });
      els.removals.appendChild(line);
      animations.push(line.animate([{ opacity: 1 }, { opacity: 0 }], removalTiming));
    }
  }
  for (const [key] of removed) {
    const previous = previousStones.get(key);
    if (!previous) continue;
    const ghost = previous.cloneNode(true);
    ghost.classList.add('stone-removing');
    ghost.querySelectorAll('.semantic-piece, .double-label').forEach(element => element.remove());
    els.removals.appendChild(ghost);
    animations.push(ghost.animate([
      { opacity: 1, transform: 'scale(1)' },
      { opacity: 0, transform: 'scale(.35)' }
    ], removalTiming));
  }
  if (!animations.length) return;
  const motion = { animations };
  activeMotion = motion;
  clearPreview();
  els.board.setAttribute('aria-busy', 'true');
  const playerName = player === 'white' ? 'Ivory' : 'Obsidian';
  if (removed.length) {
    const returned = removed.filter(([, value]) => normaliseOwner(value.owner || value.player || value.color) === player).length;
    const captured = removed.length - returned;
    const counts = [captured && `${captured} captured`, returned && `${returned} returned`].filter(Boolean);
    els.hint.textContent = `${playerName} removes ${removed.length} ${removed.length === 1 ? 'stone' : 'stones'} · ${counts.join(' · ')}`;
  } else {
    els.hint.textContent = `${playerName} pushes a stone onto the board`;
  }
  renderCapture();
  Promise.allSettled(animations.map(animation => animation.finished)).then(() => {
    if (activeMotion !== motion) return;
    cancelMotion();
    renderMeta();
    renderCapture();
    maybeAiMove();
  });
}

function playAction(action, actor) {
  const previous = clone(state);
  const before = readBoard(previous);
  const previousStones = new Map([...els.pieces.children].map(stone => [stone.dataset.key, stone]));
  past.push({ state: previous, actor });
  state = game.applyAction(state, action) || state;
  selectedCapture = null;
  render({ before, action, previousStones, actor, player: currentPlayer(previous) });
}

function pieceElements({ x, y, owner, kind, captureClass = '', capturePart = null }) {
  const color = owner === 'white' ? 'ivory' : 'obsidian';
  const stateClass = `${color}${captureClass}`;
  const elements = [];
  // Coincident semantic markers retain the unit count used by board inspection.
  const semanticCount = kind === 'double' ? 2 : 1;
  for (let index = 0; index < semanticCount; index += 1) {
    elements.push(svg('circle', { cx: x, cy: y, r: 1, class: `piece ${kind === 'double' ? 'double-piece' : 'single-piece'} ${stateClass} semantic-piece` }));
  }
  if (kind === 'double') elements.push(svg('circle', { cx: x, cy: y, r: 1, class: `double-ring ${stateClass} semantic-piece` }));
  elements.push(svg('circle', { cx: x, cy: y, r: 18, class: `piece-visual ${stateClass}` }));
  if (kind === 'double') {
    elements.push(svg('circle', { cx: x, cy: y, r: 9, class: `piece-center ${owner === 'white' ? 'obsidian' : 'ivory'}` }));
    if (capturePart) elements.push(svg('text', { x: x + 22, y: y + 3, class: `double-label${captureClass}` }, `D${capturePart.bit + 1}`));
  }
  return elements;
}

function renderMeta() {
  const player = currentPlayer();
  const reserves = readReserves(state);
  els.reserveWhite.textContent = reserves.white;
  els.reserveBlack.textContent = reserves.black;
  els.barWhite.style.width = `${Math.max(0, Math.min(100, reserves.white / 12 * 100))}%`;
  els.barBlack.style.width = `${Math.max(0, Math.min(100, reserves.black / 12 * 100))}%`;
  els.turnWhite.classList.toggle('active', player === 'white');
  els.turnBlack.classList.toggle('active', player === 'black');
  document.querySelector('[data-player="white"] .turn-pip').setAttribute('aria-hidden', String(player !== 'white'));
  const moveNumber = state.moves ?? state.move ?? state.ply ?? 0;
  els.round.textContent = String(Math.floor(moveNumber / 2) + 1).padStart(2, '0');
  els.move.textContent = `MOVE ${String(moveNumber).padStart(2, '0')}`;
  const winner = winnerOf(state);
  if (winner !== 0) {
    const winnerName = winner === 'white' ? 'Ivory' : 'Obsidian';
    els.phaseTitle.textContent = `${winnerName} wins`;
    els.phaseBadge.textContent = 'Game over';
    els.explanation.textContent = 'The last GIPF piece has been removed. Start a new game when you are ready for another table.';
    els.hint.innerHTML = '<span class="hint-dot"></span>Game over · choose New game to play again';
    return;
  }
  if (!game.hasEngine()) {
    els.phaseTitle.textContent = 'Rules engine unavailable';
    els.phaseBadge.textContent = 'WASM required';
    els.explanation.textContent = 'The board opening is shown for reference. Load the GIPF engine bundle to make moves.';
    els.hint.innerHTML = '<span class="hint-dot"></span>Rules engine required to play';
    return;
  }
  els.phaseTitle.textContent = captureChoices().length ? 'Resolve a row' : 'Place a piece';
  els.phaseBadge.textContent = captureChoices().length ? 'Capture required' : `${player === 'white' ? 'Ivory' : 'Obsidian'} to move`;
  els.explanation.textContent = captureChoices().length ? 'A row is complete. Choose which capture to resolve first, then the turn continues.' : 'Click an arrow at the board’s edge to push a piece inward. Occupied pieces shift toward the first empty point.';
  els.hint.innerHTML = captureChoices().length ? '<span class="hint-dot"></span>Choose a completed row in the capture panel' : '<span class="hint-dot"></span>Click an edge arrow to push a piece inward';
}

function renderCapture() {
  const choices = captureChoices();
  els.capture.hidden = choices.length === 0;
  els.captureOptions.replaceChildren();
  if (selectedCapture) {
    const previous = selectedCapture;
    const canonical = choices.find(choice => choice.id === previous.id) || null;
    selectedCapture = canonical && (!canonical.actions || canonical.actions.includes(previous.action)) ? { ...canonical, action: previous.action } : canonical;
  }
  const aiCapture = els.mode.value === 'ai' && currentPlayer() === els.aiColor.value;
  choices.forEach((choice, index) => {
    const button = document.createElement('button');
    button.type = 'button'; button.disabled = aiCapture || Boolean(activeMotion); button.className = `capture-option${selectedCapture?.id === choice.id ? ' selected' : ''}`;
    button.textContent = choice.label || `Row ${index + 1}`;
    button.addEventListener('click', () => { selectedCapture = choice; refreshCaptureView(); });
    els.captureOptions.appendChild(button);
  });
  els.captureToggles.replaceChildren();
  const captureActions = selectedCapture?.actions || [];
  const masks = captureActions.map(action => action - 42 - selectedCapture.line * 128);
  const bitset = masks.reduce((all, mask) => all | mask, 0);
  if (captureActions.length > 1 && bitset) {
    els.captureToggles.hidden = false;
    const label = document.createElement('span'); label.className = 'toggle-caption'; label.textContent = 'Optional doubles · green stays · red removed'; els.captureToggles.appendChild(label);
    for (let bit = 0; bit < 7; bit += 1) {
      if (!(bitset & (1 << bit))) continue;
      const currentMask = selectedCapture.action - 42 - selectedCapture.line * 128;
      const targetMask = currentMask ^ (1 << bit);
      const targetAction = captureActions.find(action => action - 42 - selectedCapture.line * 128 === targetMask);
      const toggle = document.createElement('button');
      const removing = Boolean(currentMask & (1 << bit));
      toggle.type = 'button'; toggle.className = `capture-toggle${removing ? ' selected' : ''}`; toggle.textContent = removing ? `Remove D${bit + 1}` : `Keep D${bit + 1}`; toggle.disabled = aiCapture || Boolean(activeMotion) || !targetAction;
      toggle.addEventListener('click', () => { if (targetAction) { selectedCapture = { ...selectedCapture, action: targetAction }; refreshCaptureView(); } });
      els.captureToggles.appendChild(toggle);
    }
  } else {
    els.captureToggles.hidden = true;
  }
  els.confirmCapture.disabled = !selectedCapture || aiCapture || Boolean(activeMotion);
}

function refreshCaptureView() {
  if (activeMotion) return;
  renderCapture();
  renderCaptureHighlight();
  renderPieces();
}

function renderCaptureHighlight() {
  els.captureLayer.replaceChildren();
  const info = selectedCaptureInfo();
  if (!info) return;
  for (let index = 1; index < info.nodes.length; index += 1) {
    const from = info.nodes[index - 1];
    const to = info.nodes[index];
    els.captureLayer.appendChild(svg('line', { x1: from.x, y1: from.y, x2: to.x, y2: to.y, class: 'capture-row-line' }));
  }
}

function selectedCaptureInfo() {
  if (!selectedCapture || selectedCapture.line == null) return null;
  const rawLines = game.geometry.engine?.lines;
  const rawLine = arrayLike(rawLines)?.[selectedCapture.line];
  const indices = arrayLike(rawLine);
  if (!indices?.length) return null;
  const mask = selectedCapture.action - 42 - selectedCapture.line * 128;
  const nodes = indices.map(index => game.geometry.nodes[Number(index)]).filter(Boolean);
  return { nodes, parts: indices.map((index, bit) => ({ key: game.geometry.nodes[Number(index)]?.key, bit, masked: Boolean(mask & (1 << bit)) })) };
}

function choosePoint(key) {
  if (activeMotion || aiThinking || winnerOf(state) !== 0 || captureChoices().length || (els.mode.value === 'ai' && currentPlayer() === els.aiColor.value)) return;
  const actions = legalByPoint.get(key) || [];
  if (actions.length !== 1) { showToast(actions.length > 1 ? 'Choose an incoming arrow' : 'That point is not open'); return; }
  const action = actions[0];
  const nextAction = typeof action === 'object' ? { ...action } : action;
  playAction(nextAction, 'human');
}

function chooseRay(rayId) {
  if (activeMotion || aiThinking || winnerOf(state) !== 0 || captureChoices().length || (els.mode.value === 'ai' && currentPlayer() === els.aiColor.value)) return;
  const action = legal.find(candidate => candidate === rayId);
  if (action == null) { showToast('That incoming arrow is unavailable'); return; }
  playAction(action, 'human');
}

function chooseCapture() {
  if (activeMotion || !selectedCapture || winnerOf(state) !== 0 || aiThinking || (els.mode.value === 'ai' && currentPlayer() === els.aiColor.value)) return;
  playAction(selectedCapture.action || { type: 'capture', row: selectedCapture.id || selectedCapture }, 'human');
}

function cancelAiSearch(reason) {
  localAi.cancel(reason);
  aiThinking = false;
  $('retry-ai').hidden = true;
}

function bindControls() {
  $('retry-ai').addEventListener('click', () => { gameRevision += 1; cancelAiSearch('retry'); aiBlocked = false; render(); });
  $('new-game').addEventListener('click', () => { gameRevision += 1; cancelAiSearch('new game'); aiBlocked = false; past = []; selectedCapture = null; state = game.newGame(); render(); showToast('A new table is ready'); });
  els.undo.addEventListener('click', () => {
    if (!canUndo()) return;
    gameRevision += 1; cancelAiSearch('undo'); aiBlocked = false;
    if (els.mode.value === 'ai') {
      let entry;
      while (past.length) {
        entry = past.pop();
        if (entry.actor === 'human' && currentPlayer(entry.state) !== els.aiColor.value) break;
      }
      state = entry?.state || state;
    } else {
      state = past.pop().state;
    }
    selectedCapture = null; render();
  });
  els.confirmCapture.addEventListener('click', chooseCapture);
  els.mode.addEventListener('change', () => { gameRevision += 1; cancelAiSearch('mode changed'); aiBlocked = false; document.querySelectorAll('.ai-only').forEach(el => { el.style.display = els.mode.value === 'ai' ? 'flex' : ''; }); render(); });
  els.aiColor.addEventListener('change', () => { gameRevision += 1; cancelAiSearch('machine colour changed'); aiBlocked = false; render(); });
  els.difficulty.addEventListener('change', maybeAiMove);
  const dialog = $('rules-dialog');
  [$('rules-open'), $('rules-open-secondary')].forEach(button => button.addEventListener('click', () => dialog.showModal()));
  [$('rules-close'), $('rules-done')].forEach(button => button.addEventListener('click', () => dialog.close()));
}

async function maybeAiMove() {
  if (activeMotion || els.mode.value !== 'ai' || aiThinking || aiBlocked || winnerOf(state) !== 0 || currentPlayer() !== els.aiColor.value) return;
  const revision = gameRevision;
  aiThinking = true; els.aiStatus.textContent = 'Machine thinking…'; els.hint.innerHTML = '<span class="hint-dot"></span>Machine is considering the position';
  const budget = { casual: 500, focused: 1500, deep: 5000 }[els.difficulty.value] || 1500;
  try {
    await offlineRegistration;
    if (revision !== gameRevision) return;
    const result = await localAi.search(serialiseState(state), { budgetMs: budget });
    if (revision !== gameRevision) return;
    const action = result?.action;
    const isLegal = action != null && legal.some(candidate => typeof candidate === 'number' && candidate === action);
    if (!isLegal) throw new Error('Worker returned an illegal action');
    aiModelName = result.model || aiModelName;
    els.aiStatus.textContent = aiModelName ? `${aiModelName} ready` : 'Local model ready';
    aiThinking = false;
    playAction(action, 'ai');
  } catch (error) {
    if (revision !== gameRevision || error?.name === 'AbortError') return;
    aiThinking = false;
    aiBlocked = true;
    showToast('AI could not finish · retry or switch to local mode');
    els.aiStatus.textContent = 'Local model unavailable';
    $('retry-ai').hidden = false;
    render();
  }
}

function showPreview(key) {
  clearPreview();
  if (activeMotion) return;
  if (!legalByPoint.has(key)) return;
  const node = game.geometry.byKey[key];
  const action = legalByPoint.get(key)[0];
  const ray = typeof action === 'number' ? game.geometry.rays[action] : action?.rayId != null && game.geometry.rays.find(item => item.rayId === action.rayId);
  const line = svg('line', { x1: node.x, y1: node.y, x2: ray?.outer.x || node.x, y2: ray?.outer.y || node.y, class: 'push-preview' });
  els.interaction.appendChild(line);
}
function showPreviewRay(rayId) {
  clearPreview();
  if (activeMotion) return;
  if (!legal.includes(rayId)) return;
  const ray = game.geometry.rays[rayId];
  if (ray) els.interaction.appendChild(svg('line', { x1: ray.outer.x, y1: ray.outer.y, x2: ray.inner.x, y2: ray.inner.y, class: 'push-preview' }));
}
function clearPreview() { els.interaction.replaceChildren(); }

function captureChoices() {
  const raw = state?.pendingCapture || state?.pending_capture || state?.captures || [];
  if (Array.isArray(raw) && raw.length) return raw.map((choice, index) => typeof choice === 'object' ? choice : { id: choice, label: `Row ${index + 1}`, action: choice });
  if ((state?.phase === 'capture' || state?.phase === 'CAPTURE') && Array.isArray(legal)) {
    const groups = new Map();
    legal.filter(action => typeof action === 'number' && action >= 42).forEach(action => {
      const line = Math.floor((action - 42) / 128);
      if (!groups.has(line)) groups.set(line, []);
      groups.get(line).push(action);
    });
    return [...groups].map(([line, actions]) => ({ id: `line-${line}`, line, label: `Line ${line + 1}`, actions, action: actions[0] }));
  }
  return [];
}
function currentPlayer(value = state) {
  const player = value?.currentPlayer ?? value?.current_player ?? value?.turn ?? 'white';
  return player === -1 || player === 'black' || player === 'b' ? 'black' : 'white';
}
function canUndo() {
  if (!past.length) return false;
  if (els.mode.value !== 'ai') return true;
  return past.some(entry => entry.actor === 'human' && currentPlayer(entry.state) !== els.aiColor.value);
}
function winnerOf(value) {
  const winner = value?.winner ?? 0;
  if (winner === 1 || winner === 'white' || winner === 'w') return 'white';
  if (winner === -1 || winner === 'black' || winner === 'b') return 'black';
  return 0;
}
function readReserves(value) {
  const reserves = arrayLike(value?.reserves);
  if (reserves) return { white: Number(reserves[0] ?? 12), black: Number(reserves[1] ?? 12) };
  return { white: Number(value?.reserves?.white ?? value?.reserve?.white ?? 12), black: Number(value?.reserves?.black ?? value?.reserve?.black ?? 12) };
}
function readBoard(value) {
  const board = value?.board || value?.pieces || {};
  const boardArray = arrayLike(board);
  if (boardArray) return Object.fromEntries(boardArray.map((piece, index) => {
    const node = game.geometry.nodes[index];
    if (typeof piece === 'number') return [node?.key, piece === 0 ? null : { owner: piece > 0 ? 'white' : 'black', kind: Math.abs(piece) === 2 ? 'double' : 'single' }];
    return [piece.point || piece.key || `${piece.q},${piece.r}`, piece];
  }).filter(([key]) => key));
  return board;
}
function arrayLike(value) {
  if (Array.isArray(value)) return value;
  if (value && typeof value[Symbol.iterator] === 'function') return Array.from(value);
  if (value && typeof value.size === 'function' && typeof value.get === 'function') return Array.from({ length: value.size() }, (_, index) => value.get(index));
  return null;
}
function normaliseOwner(owner) { return owner === 'black' || owner === 'b' || owner === -1 ? 'black' : 'white'; }
function clone(value) {
  if (typeof value?.clone === 'function') return value.clone();
  try { return typeof structuredClone === 'function' ? structuredClone(value) : JSON.parse(JSON.stringify(value)); }
  catch { return JSON.parse(JSON.stringify(value)); }
}
function serialiseState(value) {
  if (typeof value?.serialize === 'function') {
    try { return value.serialize(); } catch { /* use the plain object below */ }
  }
  return value;
}
function rayHitTarget(ray) {
  const dx = ray.inner.x - ray.outer.x;
  const dy = ray.inner.y - ray.outer.y;
  const length = Math.hypot(dx, dy) || 1;
  const x = -dy / length * 8;
  const y = dx / length * 8;
  const points = [[ray.outer.x + x, ray.outer.y + y], [ray.inner.x + x, ray.inner.y + y], [ray.inner.x - x, ray.inner.y - y], [ray.outer.x - x, ray.outer.y - y]]
    .map(point => point.join(',')).join(' ');
  return svg('polygon', { points, class: 'ray-hit', 'data-ray-id': ray.rayId });
}
function svg(tag, attrs, text = '') { const element = document.createElementNS(svgNS, tag); Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value)); if (text) element.appendChild(document.createTextNode(text)); return element; }
function hexPoints(radius) { return Array.from({ length: 6 }, (_, i) => { const angle = (-90 + i * 60) * Math.PI / 180; return `${400 + Math.cos(angle) * radius},${400 + Math.sin(angle) * radius}`; }).join(' '); }
function showToast(message) { clearTimeout(toastTimer); els.toast.textContent = message; els.toast.classList.add('show'); toastTimer = setTimeout(() => els.toast.classList.remove('show'), 2400); }
