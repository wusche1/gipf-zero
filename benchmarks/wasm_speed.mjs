import {createRequire} from 'module';
import {readFileSync} from 'fs';
import {dirname, resolve} from 'path';
import {fileURLToPath} from 'url';

const vendor = resolve(dirname(fileURLToPath(import.meta.url)), '../web/vendor/onnxruntime');
const ort = createRequire(import.meta.url)(vendor + '/ort.wasm.min.js');
const args = process.argv.slice(2);
const flag = (name, d) => { const i = args.indexOf(name); return i < 0 ? d : args.splice(i, 2)[1]; };
const threads = +flag('--threads', 1);
const batches = flag('--batches', '1,4,8,16,32').split(',').map(Number);
const budget = +flag('--runs', 200);
ort.env.wasm.wasmPaths = vendor + '/';
ort.env.wasm.numThreads = threads;
ort.env.wasm.simd = true;
ort.env.wasm.proxy = false;
const median = (xs) => xs.sort((a, b) => a - b)[xs.length >> 1];

for (const file of args) {
  const session = await ort.InferenceSession.create(readFileSync(file), {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
  for (const batch of batches) {
    const x = new Float32Array(batch * 441);
    for (let b = 0; b < batch; b++) for (let i = 0; i < 441; i++)
      x[b * 441 + i] = Math.floor(i / 49) === 4 ? 1 : Math.floor(i / 49) > 4 ? 0.5 : +(Math.random() < 0.15);
    const feeds = {board_features: new ort.Tensor('float32', x, [batch, 9, 7, 7])};
    const runs = Math.max(30, Math.ceil(budget / batch));
    const times = [];
    const warmup = Math.max(10, runs >> 2);
    for (let i = 0; i < warmup + runs; i++) {
      const t = performance.now();
      const out = await session.run(feeds);
      if (i >= warmup) times.push(performance.now() - t);
      for (const v of Object.values(out)) v.dispose();
    }
    const median_ms = median(times);
    console.log(JSON.stringify({file, threads, batch, runs, median_ms: +median_ms.toFixed(4), min_ms: +times[0].toFixed(4), ms_per_position: +(median_ms / batch).toFixed(4)}));
  }
  await session.release();
}
