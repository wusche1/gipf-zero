// usage: node former_kernel.mjs [--fix fixtures] [--rounds 15] [--inner 100] spec...
// spec: model.onnx | kernel.mjs:weights.bin:width:blocks:heads:ff[:relu[:int16]]; the first spec is the ratio reference
import {createRequire} from 'module';
import {readFileSync} from 'fs';
import {dirname, resolve} from 'path';
import {fileURLToPath} from 'url';

const vendor = resolve(dirname(fileURLToPath(import.meta.url)), '../web/vendor/onnxruntime');
const ort = createRequire(import.meta.url)(vendor + '/ort.wasm.min.js');
ort.env.wasm.wasmPaths = vendor + '/';
ort.env.wasm.numThreads = 1;
const args = process.argv.slice(2);
const flag = (name, d) => { const i = args.indexOf(name); return i < 0 ? d : args.splice(i, 2)[1]; };
const rounds = +flag('--rounds', 15), inner = +flag('--inner', 100), fix = flag('--fix');

async function onnx(file) {
  const s = await ort.InferenceSession.create(readFileSync(file), {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
  return async (x) => {
    const o = await s.run({board_features: new ort.Tensor('float32', x, [1, 9, 7, 7])});
    return [o.policy_logits.data, o.value.data[0]];
  };
}

async function kernel(spec) {
  const [js, bin, width, blocks, heads, ff, relu = 0, int16 = 0] = spec.split(':');
  const m = await (await import(resolve(js))).default();
  const w = new Float32Array(readFileSync(bin).buffer);
  const pw = m._malloc(w.byteLength), px = m._malloc(441 * 4), po = m._malloc(2731 * 4);
  m.HEAPF32.set(w, pw >> 2);
  if (!m._former_init(pw, w.length, +width, +blocks, +heads, +ff, +relu, +int16)) throw new Error('bad weights ' + spec);
  return (x) => {
    m.HEAPF32.set(x, px >> 2);
    m._former_forward(px, po, po + 2730 * 4);
    return [m.HEAPF32.subarray(po >> 2, (po >> 2) + 2730), m.HEAPF32[(po >> 2) + 2730]];
  };
}

const runs = [];
for (const a of args) runs.push(a.endsWith('.onnx') ? await onnx(a) : await kernel(a));
if (fix) {
  const f = new Float32Array(readFileSync(fix).buffer), n = f.length / (441 + 2731);
  for (let k = 0; k < args.length; k++) {
    let el = 0, ev = 0;
    for (let i = 0; i < n; i++) {
      const [l, v] = await runs[k](f.subarray(i * 441, (i + 1) * 441));
      for (let j = 0; j < 2730; j++) el = Math.max(el, Math.abs(l[j] - f[n * 441 + i * 2730 + j]));
      ev = Math.max(ev, Math.abs(v - f[n * 3171 + i]));
    }
    console.log(args[k].split('/').pop(), 'max_logit_error', el.toExponential(2), 'max_value_error', ev.toExponential(2));
  }
}
const x = new Float32Array(441);
for (let i = 0; i < 441; i++) x[i] = Math.floor(i / 49) === 4 ? 1 : Math.floor(i / 49) > 4 ? 0.5 : +(Math.random() < 0.15);
const t = args.map(() => []);
for (let r = 0; r <= rounds; r++)
  for (let k = 0; k < args.length; k++) {
    const t0 = performance.now();
    for (let i = 0; i < inner; i++) await runs[k](x);
    if (r) t[k].push((performance.now() - t0) / inner);
  }
const med = (a) => [...a].sort((p, q) => p - q)[a.length >> 1];
for (let k = 0; k < args.length; k++)
  console.log(JSON.stringify({spec: args[k].split('/').pop(), median_ms: +med(t[k]).toFixed(4), min_ms: +Math.min(...t[k]).toFixed(4), ratio: +med(t[k].map((v, i) => v / t[0][i])).toFixed(3)}));
