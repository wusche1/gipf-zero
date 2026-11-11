import argparse, os, time
from pathlib import Path
import numpy as np
import torch
import gipf_engine as ge
from .net import load, compact, MAXA


@torch.inference_mode()
def evaluate(model, forest, n, device):
    if not n:
        return
    x = torch.from_numpy(forest.encode_pending()).to(device)
    idx = torch.from_numpy(forest.pending_action_indices()).to(device, torch.long)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device != 'cpu'):
        logits, values = model(x)
    forest.finish_selected(torch.gather(logits.float(), 1, idx).cpu().numpy(), values.float().cpu().numpy())


def search(model, nodes, sims, device, cpuct=1.5):
    forest = ge.NativeForest(nodes)
    evaluate(model, forest, forest.expand_unexpanded_root_pending(), device)
    for _ in range(sims):
        evaluate(model, forest, forest.select_pending(cpuct), device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True); p.add_argument('--out', required=True)
    p.add_argument('--games', type=int, default=512); p.add_argument('--fast', type=int, default=32)
    p.add_argument('--full', type=int, default=200); p.add_argument('--pfull', type=float, default=.25)
    p.add_argument('--max-ply', type=int, default=240); p.add_argument('--seed', type=int, default=1)
    p.add_argument('--shard', type=int, default=20000); p.add_argument('--device', default='cuda')
    p.add_argument('--seconds', type=float, default=1e9)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed); dev = a.device
    torch.backends.cudnn.benchmark = True
    mtime = os.path.getmtime(a.weights); model, _ = load(a.weights, dev)
    G = a.games
    roots = [ge.NativeNode(ge.State()) for _ in range(G)]; hist = [[] for _ in range(G)]
    buf = []; shards = games = positions = cutoffs = white = 0; start = last = time.time(); lengths = []
    while time.time() - start < a.seconds:
        m = os.path.getmtime(a.weights)
        if m != mtime:
            mtime = m; model, _ = load(a.weights, dev)
            roots = [ge.NativeNode(r.state) for r in roots]
        full = rng.random(G) < a.pfull
        forest = ge.NativeForest(roots)
        evaluate(model, forest, forest.expand_unexpanded_root_pending(), dev)
        for i in np.flatnonzero(full):
            n = len(roots[i].actions); roots[i].mix_noise(rng.dirichlet(np.full(n, 10 / n)))
        for _ in range(a.fast):
            evaluate(model, forest, forest.select_pending(1.5), dev)
        if full.any() and a.full > a.fast:
            sub = ge.NativeForest([roots[i] for i in np.flatnonzero(full)])
            for _ in range(a.full - a.fast):
                evaluate(model, sub, sub.select_pending(1.5), dev)
        board, meta = compact(ge.encode_batch([r.state for r in roots]))
        for i, root in enumerate(roots):
            visits = np.asarray(root.visits, np.float64); total = visits.sum(); pi = visits / total
            q = float(np.sum(root.values) / total); acts = np.asarray(root.actions, np.int16)
            w = pi if root.state.ply < 16 else pi ** 4
            sel = int(rng.choice(len(pi), p=w / w.sum()))
            if len(acts) > MAXA:
                keep = np.argsort(-pi)[:MAXA]; hacts, hpi = acts[keep], pi[keep] / pi[keep].sum()
            else:
                hacts, hpi = acts, pi
            hist[i].append((board[i], meta[i], hacts, hpi, bool(full[i]), root.actor, q, root.state.ply))
            child = root.child(sel); roots[i] = child; s = child.state
            cutoff = (s.ply >= a.max_ply and s.phase != 'capture') or len(hist[i]) >= 2000
            if s.winner or cutoff:
                games += 1; cutoffs += not s.winner; white += s.winner == 1; lengths.append(len(hist[i]))
                buf.extend((b, mt, ac, pp, f, s.winner * actor, qq, ply) for b, mt, ac, pp, f, actor, qq, ply in hist[i])
                roots[i] = ge.NativeNode(ge.State()); hist[i] = []
        positions += G
        if len(buf) >= a.shard:
            N = len(buf); act = np.full((N, MAXA), -1, np.int16); pi = np.zeros((N, MAXA), np.float16)
            for j, r in enumerate(buf):
                act[j, :len(r[2])] = r[2]; pi[j, :len(r[2])] = r[3]
            tmp = out / f'tmp-{a.seed}.npz'
            np.savez_compressed(tmp, board=np.stack([r[0] for r in buf]), meta=np.stack([r[1] for r in buf]), act=act, pi=pi,
                     full=np.array([r[4] for r in buf]), z=np.array([r[5] for r in buf], np.int8),
                     q=np.array([r[6] for r in buf], np.float16), ply=np.array([r[7] for r in buf], np.int16))
            os.replace(tmp, out / f'shard-{a.seed}-{shards:06d}.npz'); shards += 1; buf = []
        if time.time() - last > 30:
            last = time.time()
            print(f'{last - start:.0f}s games {games} pos {positions} pos/s {positions / (last - start):.0f} '
                  f'len {np.mean(lengths[-500:]) if lengths else 0:.0f} cut {cutoffs / max(1, games):.3f} white {white / max(1, games):.3f}', flush=True)


if __name__ == '__main__':
    main()
