const NS = 'http://www.w3.org/2000/svg';
const svg = (tag, attrs, parent) => {
  const node = document.createElementNS(NS, tag);
  for (const key in attrs) node.setAttribute(key, attrs[key]);
  parent?.appendChild(node);
  return node;
};
const H = Math.sqrt(3) / 2;
const POINTS = [];
for (let r = -3; r <= 3; r++) for (let q = -3; q <= 3; q++) if (Math.abs(q + r) <= 3) POINTS.push({ q, r, x: q + r / 2, y: r * H });
const hexagon = (radius, c) => Array.from({ length: 6 }, (_, i) => `${c + Math.cos(i * Math.PI / 3) * radius},${c + Math.sin(i * Math.PI / 3) * radius}`).join(' ');

const miniBoard = (s) => {
  svg('polygon', { points: hexagon(47, 50), class: 'art-surface' }, s);
  for (const p of POINTS) {
    const corner = Math.max(Math.abs(p.q), Math.abs(p.r), Math.abs(p.q + p.r)) === 3 && p.q * p.r * (p.q + p.r) === 0;
    const ivory = (p.q === 3 && p.r === 0) || (p.q === -3 && p.r === 3) || (p.q === 0 && p.r === -3);
    svg('circle', { cx: 50 + 12 * p.x, cy: 50 + 12 * p.y, r: corner ? 4.4 : 2.3, class: corner ? (ivory ? 'art-ivory' : 'art-ink') : 'art-dot' }, s);
  }
};
const ART = {
  board: miniBoard,
  tokens(s) {
    for (let i = 0; i < 6; i++) {
      svg('circle', { cx: 13, cy: 13 + i * 15, r: 3.6, class: i === 1 ? 'art-ink' : i === 4 ? 'art-ivory' : 'art-dot' }, s);
      svg('path', { d: `M21 ${13 + i * 15}h8`, class: 'art-line' }, s);
      for (let j = 0; j < 6; j++) svg('rect', { x: 33 + j * 10.5, y: 8.5 + i * 15, width: 8.5, height: 9, rx: 2, class: 'art-hot', opacity: .12 + ((i * 7 + j * 5 + i * j) % 9) / 10 }, s);
    }
  },
  layers(s) {
    for (let l = 0; l < 4; l++) {
      const x = 5 + l * 36.5;
      svg('rect', { x, y: 5, width: 31, height: 90, rx: 7, class: 'art-slab' }, s);
      for (let a = 0; a < 5; a++) for (let b = 0; b < 5; b++) svg('line', { x1: x + 6, y1: 16 + a * 17, x2: x + 25, y2: 16 + b * 17, class: 'art-mesh' }, s);
      for (let b = 0; b < 5; b++) svg('line', { x1: x + 6, y1: 16 + ((l + 1) % 5) * 17, x2: x + 25, y2: 16 + b * 17, class: 'art-hot-line', 'stroke-width': .9 }, s);
      for (let a = 0; a < 5; a++) for (const dx of [6, 25]) svg('circle', { cx: x + dx, cy: 16 + a * 17, r: 2.7, class: dx === 6 && a === (l + 1) % 5 ? 'art-hot' : 'art-dot' }, s);
    }
  },
  readout(s) {
    for (let i = 0; i < 6; i++) {
      const g = svg('g', { transform: `translate(50 50) rotate(${i * 60})` }, s);
      if (i < 3) svg('path', { d: 'M-45 0h90', class: 'art-line', 'stroke-dasharray': '1 5' }, g);
      svg('path', { d: 'M15 0h17', class: 'art-hot-line' }, g);
      svg('path', { d: 'M38 0l-8 -4.5v9z', class: 'art-hot' }, g);
    }
    svg('circle', { cx: 50, cy: 50, r: 9, class: 'art-ink' }, s);
    svg('circle', { cx: 50, cy: 50, r: 4, class: 'art-ivory' }, s);
  },
  outputs(s) {
    [20, 33, 13, 54, 26, 10, 18].forEach((h, i) => svg('rect', { x: 10 + i * 11.8, y: 66 - h, width: 8.5, height: h, rx: 2, class: i === 3 ? 'art-hot' : 'art-dot' }, s));
    svg('rect', { x: 10, y: 80, width: 80, height: 8, rx: 4, class: 'art-ink' }, s);
    svg('rect', { x: 10, y: 80, width: 52, height: 8, rx: 4, class: 'art-ivory' }, s);
  },
  search(s) {
    const edges = [[50, 12, 20, 40], [50, 12, 50, 40, 1], [50, 12, 80, 40], [20, 40, 10, 66], [20, 40, 30, 66], [50, 40, 46, 66], [50, 40, 64, 66, 1], [80, 40, 86, 66], [64, 66, 54, 90], [64, 66, 74, 90, 1]];
    for (const [x1, y1, x2, y2, hot] of edges) svg('line', { x1, y1, x2, y2, class: hot ? 'art-hot-line' : 'art-line' }, s);
    svg('circle', { cx: 50, cy: 12, r: 5, class: 'art-ink' }, s);
    for (const [, , cx, cy, hot] of edges) svg('circle', { cx, cy, r: hot ? 4.6 : 3.4, class: hot ? 'art-hot' : 'art-dot' }, s);
  },
  move(s) {
    miniBoard(s);
    svg('path', { d: 'M65.5 97.5l-5 -8.7', class: 'art-hot-line' }, s);
    svg('circle', { cx: 56, cy: 81.2, r: 4.6, class: 'art-hot' }, s);
  }
};
for (const s of document.querySelectorAll('[data-art]')) ART[s.dataset.art](s);

const stages = [...document.querySelectorAll('.stage')];
const detail = document.getElementById('stage-detail');
const showStage = (stage) => {
  for (const s of stages) s.classList.toggle('active', s === stage);
  detail.replaceChildren(stage.querySelector('.stage-copy').cloneNode(true));
};
for (const stage of stages) for (const type of ['click', 'mouseenter', 'focus']) stage.addEventListener(type, () => showStage(stage));
showStage(stages[2]);

const { bias } = await (await fetch(new URL('./about-data.json', import.meta.url))).json();
const view = { from: POINTS.findIndex(p => p.q === 0 && p.r === 0), layer: 3, head: 1 };
const bonus = (layer, head, to) => bias[layer][head][(to.q - POINTS[view.from].q + 6) * 13 + to.r - POINTS[view.from].r + 6];
const heat = (b) => Math.min(1, Math.max(0, (b + 6) / 9)) ** 1.5;
const COLD = [241, 235, 227], HOT = [180, 87, 63];
const colour = (t) => `rgb(${COLD.map((c, i) => Math.round(c + (HOT[i] - c) * t)).join(',')})`;

const board = document.getElementById('att-board');
const readout = document.getElementById('att-readout');
svg('polygon', { points: hexagon(372, 400), class: 'att-surface' }, board);
for (const a of POINTS) for (const b of POINTS) {
  if ((b.q - a.q === 1 && b.r === a.r) || (b.q === a.q && b.r - a.r === 1) || (b.q - a.q === 1 && b.r - a.r === -1)) svg('line', { x1: 400 + 100 * a.x, y1: 400 + 100 * a.y, x2: 400 + 100 * b.x, y2: 400 + 100 * b.y, class: 'att-grid' }, board);
}
const nodes = POINTS.map((p, i) => {
  const node = svg('circle', { cx: 400 + 100 * p.x, cy: 400 + 100 * p.y, class: 'att-node', 'data-key': `${p.q},${p.r}`, tabindex: 0, role: 'button', 'aria-label': `Point ${p.q},${p.r}` }, board);
  const select = () => { view.from = i; paint(); };
  node.addEventListener('click', select);
  node.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); } });
  node.addEventListener('mouseenter', () => describe(i));
  node.addEventListener('mouseleave', () => describe());
  return node;
});

const heads = document.getElementById('heads');
const thumbs = [];
for (let layer = 0; layer < 4; layer++) {
  heads.insertAdjacentHTML('beforeend', `<span class="row">Layer ${layer + 1}</span>`);
  for (let head = 0; head < 4; head++) {
    const button = document.createElement('button');
    button.className = 'head';
    button.type = 'button';
    button.setAttribute('aria-label', `Layer ${layer + 1}, head ${head + 1}`);
    const s = svg('svg', { viewBox: '-3.6 -3.2 7.2 6.4', 'aria-hidden': 'true' }, button);
    const dots = POINTS.map(p => svg('circle', { cx: p.x, cy: p.y, r: .42 }, s));
    button.addEventListener('click', () => { view.layer = layer; view.head = head; paint(); });
    heads.appendChild(button);
    thumbs.push({ layer, head, button, dots });
  }
}

function describe(i) {
  const name = `Layer ${view.layer + 1} · head ${view.head + 1}`;
  if (i === undefined || i === view.from) readout.textContent = `${name} — hover a point for its bonus, click to move the viewpoint`;
  else readout.textContent = `${name} — bonus ${bonus(view.layer, view.head, POINTS[i]).toFixed(1).replace('-', '−').replace(/^(?=\d)/, '+')} for looking here`;
}

function paint() {
  nodes.forEach((node, i) => {
    const t = heat(bonus(view.layer, view.head, POINTS[i]));
    node.classList.toggle('selected', i === view.from);
    node.style.fill = colour(t);
    node.setAttribute('r', i === view.from ? 25 : 15 + 17 * t);
  });
  for (const { layer, head, button, dots } of thumbs) {
    button.setAttribute('aria-pressed', layer === view.layer && head === view.head);
    dots.forEach((dot, i) => { dot.style.fill = i === view.from ? '#26302e' : colour(heat(bonus(layer, head, POINTS[i]))); });
  }
  describe();
}
paint();
