import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import torch
from .net import build, planes, CELL_PERM, ACTION_PERM, MAXA


class Ring:
    def __init__(self, size, dev):
        self.size, self.dev, self.n, self.total, self.seen = size, dev, 0, 0, set()
        z = lambda *s, dtype: torch.zeros(*s, dtype=dtype, device=dev)
        self.d = dict(board=z(size, 37, dtype=torch.int8), meta=z(size, 4, dtype=torch.uint8), act=z(size, MAXA, dtype=torch.int16),
                      pi=z(size, MAXA, dtype=torch.float16), full=z(size, dtype=torch.bool), z=z(size, dtype=torch.int8))

    def add(self, path):
        f = np.load(path); N = len(f['z'])
        idx = (torch.arange(N) + self.total) % self.size
        for k, t in self.d.items():
            t[idx.to(self.dev)] = torch.from_numpy(f[k]).to(self.dev)
        self.total += N; self.n = min(self.size, self.total)

    def poll(self, data):
        new = sorted((p for p in Path(data).glob('shard-*.npz') if p.name not in self.seen), key=os.path.getmtime)
        for p in new:
            self.add(p); self.seen.add(p.name)
        return len(new)

    def batch(self, B, lo=0, hi=None, augment=True):
        i = torch.randint(lo, hi or self.n, (B,), device=self.dev)
        s = torch.randint(12 if augment else 1, (B,), device=self.dev)
        board = self.d['board'][i].gather(1, CELL_PERM.to(self.dev)[s])
        act = self.d['act'][i].long(); pad = act < 0
        act = ACTION_PERM.to(self.dev)[s[:, None], act.clamp(min=0)]
        return planes(board, self.d['meta'][i]), act, pad, self.d['pi'][i].float(), self.d['full'][i].float(), self.d['z'][i].float()


def losses(model, b, amp):
    x, act, pad, pi, full, z = b
    with torch.autocast('cuda', dtype=amp or torch.bfloat16, enabled=amp is not None):
        logits, v = model(x)
    lp = torch.log_softmax(logits.float().gather(1, act).masked_fill(pad, -1e9), 1)
    ploss = (-(pi * lp).sum(1) * full).sum() / full.sum().clamp(min=1)
    return ploss, (v.float() - z).square().mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True); p.add_argument('--out', required=True)
    p.add_argument('--kind', default='hex'); p.add_argument('--width', type=int, default=64); p.add_argument('--blocks', type=int, default=6)
    p.add_argument('--batch', type=int, default=1024); p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--reuse', type=float, default=4.0); p.add_argument('--window', type=int, default=1000000)
    p.add_argument('--min', type=int, default=20000); p.add_argument('--steps', type=int, default=0)
    p.add_argument('--val', type=int, default=0); p.add_argument('--init'); p.add_argument('--device', default='cuda')
    p.add_argument('--ckpt-every', type=int, default=250000); p.add_argument('--seconds', type=float, default=1e9)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--amp', default='bf16'); p.add_argument('--sync-budget', type=float); p.add_argument('--compile', action='store_true')
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dev = a.device; amp = None if dev == 'cpu' else dict(bf16=torch.bfloat16, fp16=torch.float16)[a.amp]
    torch.manual_seed(a.seed); torch.backends.cudnn.benchmark = True
    config = dict(kind=a.kind, width=a.width, blocks=a.blocks)
    steps = 0; next_ckpt = a.ckpt_every
    if a.init:
        data = torch.load(a.init, map_location=dev, weights_only=False); config = data['config']
    model = build(config).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    if a.init:
        model.load_state_dict(data['model'])
        if not a.steps:
            steps = data['steps']; next_ckpt = (data['positions'] // a.ckpt_every + 1) * a.ckpt_every
            if 'opt' in data:
                opt.load_state_dict(data['opt'])
    params = sum(x.numel() for x in model.parameters())
    ring = Ring(a.window, dev); offset = None
    fwd = torch.compile(model) if a.compile else model

    def save(path, full=False):
        if not all(torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):
            good = torch.load(out / 'weights.pt', map_location=dev, weights_only=False)
            model.load_state_dict(good['model']); opt.load_state_dict(good['opt']); log(event='rollback'); return
        d = dict(model=model.state_dict(), config=config, steps=steps, positions=ring.total, params=params)
        if full:
            d['opt'] = opt.state_dict()
        torch.save(d, str(path) + '.tmp'); os.replace(str(path) + '.tmp', path)

    def log(**kw):
        rec = dict(time=time.time(), steps=steps, positions=ring.total, **kw)
        with (out / 'log.jsonl').open('a') as f:
            f.write(json.dumps(rec) + '\n')
        print(json.dumps(rec), flush=True)

    if not (out / 'weights.pt').exists() or a.init:
        model.eval(); save(out / 'weights.pt')
    log(event='start', params=params, config=config)
    start = time.time(); last_pub = start; acc = []; bad = []; gns = []

    def step(b, lr):
        nonlocal steps
        for g in opt.param_groups:
            g['lr'] = lr * min(1, (steps + 1) / 500)
        pl, vl = losses(fwd, b, amp)
        opt.zero_grad(set_to_none=True); (pl + vl).backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5); steps += 1
        gns.append(gn.item())
        if not torch.isfinite(gn):
            if not bad:
                torch.save(dict(batch=b, model=model.state_dict(), loss=(pl.item(), vl.item())), out / 'badbatch.pt')
            bad.append(steps); return
        opt.step(); acc.append((pl.item(), vl.item()))

    if a.steps:
        ring.poll(a.data); train_n = ring.n - a.val
        for s in range(a.steps):
            model.train(); step(ring.batch(a.batch, 0, train_n), a.lr * (0.5 + 0.5 * math.cos(math.pi * s / a.steps)))
            if (s + 1) % max(1, a.steps // 20) == 0 or s + 1 == a.steps:
                model.eval(); vals = []
                with torch.no_grad():
                    for j in range(0, a.val, 4096):
                        i = torch.arange(train_n + j, min(ring.n, train_n + j + 4096), device=dev)
                        x = planes(ring.d['board'][i], ring.d['meta'][i]); act = ring.d['act'][i].long()
                        pl, vl = losses(model, (x, act.clamp(min=0), act < 0, ring.d['pi'][i].float(), ring.d['full'][i].float(), ring.d['z'][i].float()), amp)
                        vals.append((pl.item() * ring.d['full'][i].sum().item(), vl.item() * len(i), ring.d['full'][i].sum().item(), len(i)))
                v = np.sum(vals, 0) if vals else np.ones(4)
                m = np.mean(acc, 0); acc.clear()
                log(event='train', ploss=m[0], vloss=m[1], val_ploss=v[0] / max(1, v[2]), val_vloss=v[1] / max(1, v[3]), elapsed=time.time() - start)
        save(out / 'weights.pt', True)
        return

    while time.time() - start < a.seconds:
        ring.poll(a.data)
        if offset is None:
            offset = 0 if a.sync_budget is None else int((ring.total - a.sync_budget * ring.n) * a.reuse / a.batch) - steps
        budget = int(ring.total * a.reuse / a.batch) - offset - steps
        if ring.total < a.min or budget <= 0:
            time.sleep(2); continue
        model.train()
        for _ in range(min(budget, 200)):
            step(ring.batch(a.batch), a.lr)
        model.eval()
        if time.time() - last_pub > 45:
            last_pub = time.time(); save(out / 'weights.pt', True)
            m = np.mean(acc, 0) if acc else [float('nan')] * 2; acc.clear()
            log(event='train', ploss=m[0], vloss=m[1], elapsed=time.time() - start, bad=len(bad), gn_max=max(gns), gn_mean=float(np.mean(gns))); gns.clear()
        if ring.total >= next_ckpt:
            save(out / f'ckpt-{next_ckpt // 1000:07d}k.pt'); next_ckpt += a.ckpt_every
    save(out / 'weights.pt', True)


if __name__ == '__main__':
    main()
