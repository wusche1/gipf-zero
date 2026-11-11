import argparse, itertools, json
import numpy as np
from .net import load
from .arena import play


def fit(records, anchor=None):
    names = sorted({r[k] for r in records for k in ('a', 'b')})
    ix = {n: i for i, n in enumerate(names)}
    W = np.zeros((len(names), len(names)))
    for r in records:
        i, j = ix[r['a']], ix[r['b']]
        W[i, j] += r['wins'] + r['cutoffs'] / 2; W[j, i] += r['losses'] + r['cutoffs'] / 2
    N = W + W.T; r = np.zeros(len(names))
    s = np.log(10) / 400
    for _ in range(2000):
        P = 1 / (1 + np.exp(s * (r[None] - r[:, None])))
        g = (W - N * P).sum(1) - 1e-4 * r
        h = (N * P * (1 - P)).sum(1) * s + 1e-4
        r += 0.5 * g / h
    r -= r[ix[anchor]] if anchor else r.mean()
    return dict(zip(names, r)), N.sum(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--players', nargs='*', default=[]); p.add_argument('--games', type=int, default=100)
    p.add_argument('--out', required=True); p.add_argument('--pairs', default='all')
    p.add_argument('--device', default='cuda'); p.add_argument('--seed', type=int, default=1); p.add_argument('--anchor'); p.add_argument('--shard', default='0/1')
    a = p.parse_args()
    players = {}
    for s in a.players:
        name, rest = s.split('=', 1); path, sims = rest.rsplit(':', 1); players[name] = (path, int(sims))
    names = list(players)
    pairs = list(itertools.combinations(names, 2)) if a.pairs == 'all' else [(names[0], n) for n in names[1:]] if a.pairs == 'first' else list(zip(names, names[1:]))
    k, n = map(int, a.shard.split('/'))
    models = {path: load(path, a.device)[0] for path in {players[z][0] for pr in pairs[k::n] for z in pr}}
    for x, y in pairs[k::n]:
        w, l, c = play([models[players[x][0]], models[players[y][0]]], [players[x][1], players[y][1]], a.games, a.seed, a.device)
        rec = dict(a=x, b=y, wins=w, losses=l, cutoffs=c, seed=a.seed)
        print(json.dumps(rec), flush=True)
        with open(a.out, 'a') as f:
            f.write(json.dumps(rec) + '\n')
    records = [json.loads(l) for l in open(a.out)]
    elo, n = fit(records, a.anchor)
    for (name, e), k in sorted(zip(elo.items(), n), key=lambda t: -t[0][1]):
        print(f'{name:40s} {e:8.0f}  games {k:.0f}')


if __name__ == '__main__':
    main()
