import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

FORWARD = r'''
const [engine, weights, arch] = process.argv.slice(1), fs = require('fs');
require(engine)().then((m) => {
  const w = fs.readFileSync(weights), x = fs.readFileSync(0);
  if (!m.load_kernel(new Float32Array(w.buffer, w.byteOffset, w.length / 4), JSON.parse(arch))) throw new Error('weights do not fit arch');
  const f = new Float32Array(x.buffer, x.byteOffset, x.length / 4), out = new Float32Array(f.length / 441 * 2731);
  for (let i = 0; i < f.length / 441; i++) out.set(m.kernel_forward(f.subarray(i * 441, (i + 1) * 441)), i * 2731);
  process.stdout.write(Buffer.from(out.buffer));
});
'''


def positions(count, seed=0):
    import gipf_engine as ge
    rng, states, state = random.Random(seed), [], ge.State()
    while len(states) < count:
        if state.winner:
            state = ge.State()
        for _ in range(rng.randrange(1, 12)):
            if not state.winner:
                state.apply(rng.choice(state.legal_actions()))
        if not state.winner:
            states.append(state.clone())
    return states


# Largest |WASM kernel - PyTorch| over all logits and over the values.
def kernel_error(model, weights, arch):
    import numpy as np
    import torch
    import gipf_engine as ge
    x = ge.encode_batch(positions(16))
    out = subprocess.run(['node', '-e', FORWARD, str(Path(__file__).resolve().parents[1] / 'web/gipf_engine.js'), str(weights), json.dumps(arch)],
                         input=x.tobytes(), capture_output=True, check=True).stdout
    got = np.frombuffer(out, np.float32).reshape(len(x), 2731)
    with torch.inference_mode():
        logits, value = model(torch.from_numpy(x))
    return float(np.abs(got[:, :2730] - logits.numpy()).max()), float(np.abs(got[:, 2730] - value.numpy()).max())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('--title', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--int16', action='store_true')
    p.add_argument('--cpuct', type=float, default=1.5)
    p.add_argument('--fpu', type=float, default=-1.0)
    p.add_argument('--budgets', type=int, nargs=3, default=[300, 1000, 3000])
    p.add_argument('--web', type=Path, default=Path('web'))
    a = p.parse_args()
    models = a.web / 'models'
    from ops.dump_former import dump
    from training import net
    model = net.load(a.checkpoint)[0]
    block = model.blocks[0]
    arch = {'width': model.embed.out_features, 'blocks': len(model.blocks), 'heads': block.h, 'ff': block.f1.out_features,
            'relu': False, 'int16': a.int16}
    meta = {'name': a.title, 'format': 'kernel', 'path': f'{a.name}.bin', 'arch': arch}
    dump(model).tofile(models / meta['path'])
    logit_error, value_error = kernel_error(model, models / meta['path'], arch)
    print(f'WASM kernel vs PyTorch on 16 positions: max logit error {logit_error:.2e}, max value error {value_error:.2e}')
    if not max(logit_error, value_error) < (0.02 if a.int16 else 1e-3):
        raise SystemExit('kernel does not reproduce the checkpoint; metadata not written')
    meta |= {'sha256': hashlib.sha256((models / meta['path']).read_bytes()).hexdigest(),
             'cpuct': a.cpuct, 'fpu': a.fpu, 'budgetsMs': dict(zip(('fast', 'normal', 'strong'), a.budgets)),
             'ponder': {'maxMs': 20000, 'maxVisits': 20000}}
    (models / f'{a.name}.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')
    print(f'wrote {models / meta["path"]} and {models / a.name}.json')
