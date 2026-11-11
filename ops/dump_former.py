import argparse
import numpy as np
from training import net


def dump(model, norm='ln'):
    sd = {k: v.double().numpy() for k, v in model.state_dict().items()}
    d, heads = model.embed.out_features, model.blocks[0].h
    rd = 1 if norm == 'none' else d ** 0.5
    rel = model.rel.numpy()

    def center(w):
        return w - w.mean(-1, keepdims=True) if norm == 'ln' else w

    def read(p, n):
        w, b = sd[p + '.weight'], sd[p + '.bias']
        return [w.T * sd[n + '.weight'][:, None] * rd, b + w @ sd[n + '.bias']]

    def write(p):
        return [center(sd[p + '.weight'].T), center(sd[p + '.bias'])]

    lines = np.full((21, 7), -1)
    for l, cells in enumerate(net.LINES):
        lines[l, :len(cells)] = cells
    push = [r[0] * 32 + net.DIRS.index(tuple(net.COORDS[r[1]] - net.COORDS[r[0]])) for r in net.RAYS]
    out = [net.FLAT.numpy(), push, lines, center(sd['embed.weight'].T), center(sd['embed.bias'] + sd['pos'])]
    for i in range(len(model.blocks)):
        p = f'blocks.{i}.'
        w, b = read(p + 'qkv', p + 'n1')
        w[:, :d] *= (d // heads) ** -0.5
        b[:d] *= (d // heads) ** -0.5
        out += [w, b, sd[p + 'bias'][:, rel]] + write(p + 'o') + read(p + 'f1', p + 'n2') + write(p + 'f2')
    pw, pb = read('head.policy', 'norm')
    vw, vb = read('head.vproj', 'norm')
    out += [np.concatenate([pw, vw], 1), pb, vb, sd['head.value.0.weight'].T, sd['head.value.0.bias'],
            sd['head.value.2.weight'], sd['head.value.2.bias']]
    return np.concatenate([np.asarray(a, np.float64).ravel() for a in out]).astype(np.float32)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    dump(net.load(a.checkpoint)[0]).tofile(a.output)
