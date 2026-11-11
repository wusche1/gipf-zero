import argparse, json, random
import numpy as np
import gipf_engine as ge
from .net import load
from .selfplay import search
from .evaluate import settle_opening, is_insertion_cutoff


def play(models, sims, games, seed, device, max_ply=300, cpuct=1.5, opening=4):
    active = []
    for g in range(games):
        s = ge.State(); settle_opening(s, opening, random.Random(seed + g // 2))
        active.append([s, 1 if g % 2 == 0 else -1, 0])
    result = [0, 0, 0]
    while active:
        for who in (0, 1):
            group = [g for g in active if not g[0].winner and (g[0].current_player == g[1]) == (who == 0)]
            if not group:
                continue
            nodes = [ge.NativeNode(g[0]) for g in group]
            search(models[who], nodes, sims[who], device, cpuct)
            for g, n in zip(group, nodes):
                g[0].apply(n.actions[int(np.argmax(n.visits))]); g[2] += 1
        rest = []
        for g in active:
            s = g[0]
            if s.winner:
                result[0 if s.winner == g[1] else 1] += 1
            elif is_insertion_cutoff(s, max_ply) or g[2] >= 2000:
                result[2] += 1
            else:
                rest.append(g)
        active = rest
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a', required=True); p.add_argument('--b', required=True)
    p.add_argument('--sims-a', type=int, default=100); p.add_argument('--sims-b', type=int, default=100)
    p.add_argument('--games', type=int, default=200); p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', default='cuda'); p.add_argument('--out')
    a = p.parse_args()
    models = [load(a.a, a.device)[0], load(a.b, a.device)[0]]
    w, l, c = play(models, [a.sims_a, a.sims_b], a.games, a.seed, a.device)
    rec = dict(a=a.a, b=a.b, sims_a=a.sims_a, sims_b=a.sims_b, wins=w, losses=l, cutoffs=c, seed=a.seed)
    print(json.dumps(rec), flush=True)
    if a.out:
        with open(a.out, 'a') as f:
            f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()
