import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .model import GEO, COORDS, ROWS, COLS, ACTIONS, SYMMETRIES, PolicyValue, ModelConfig

CELLS = 37
FLAT = torch.tensor(ROWS * 7 + COLS)
DIRS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
RAYS = [r['cells'] if isinstance(r, dict) else r for r in GEO['rays']]
LINES = GEO['lines']
MAXA = 64


def _head_tables():
    push = torch.zeros(42, 6 * CELLS)
    for a, ray in enumerate(RAYS):
        d = DIRS.index(tuple(COORDS[ray[1]] - COORDS[ray[0]]))
        push[a, d * CELLS + ray[0]] = 1
    line = torch.zeros(21, 3 * CELLS)
    coeff = torch.zeros(21 * 128, CELLS)
    for l, cells in enumerate(LINES):
        for c in cells:
            line[l, (l // 7) * CELLS + c] = 1 / len(cells)
        for bits in range(128):
            for j, c in enumerate(cells):
                if bits >> j & 1:
                    coeff[l * 128 + bits, c] = 1
    return push, line, coeff


class Head(nn.Module):
    def __init__(self, c, vc=16):
        super().__init__()
        self.policy = nn.Linear(c, 10)
        self.vproj = nn.Linear(c, vc)
        self.value = nn.Sequential(nn.Linear(vc * CELLS, 128), nn.ReLU(), nn.Linear(128, 1))
        push, line, coeff = _head_tables()
        self.register_buffer('push', push.T.contiguous(), persistent=False)
        self.register_buffer('line', line.T.contiguous(), persistent=False)
        self.register_buffer('coeff', coeff.T.contiguous(), persistent=False)

    def forward(self, h):
        p = self.policy(h).transpose(1, 2)
        push = p[:, :6].flatten(1) @ self.push
        line = p[:, 6:9].flatten(1) @ self.line
        cap = line.repeat_interleave(128, 1) + p[:, 9] @ self.coeff
        v = self.value(torch.relu(self.vproj(h)).flatten(1)).squeeze(-1)
        return torch.cat((push, cap), 1), torch.tanh(v)


class HexConv(nn.Conv2d):
    def __init__(self, i, o):
        super().__init__(i, o, 3, padding=1, bias=False)
        m = torch.ones(1, 1, 3, 3)
        m[0, 0, 0, 0] = m[0, 0, 2, 2] = 0
        self.register_buffer('m', m, persistent=False)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.m, padding=1)


class HexBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1, self.b1, self.c2, self.b2 = HexConv(c, c), nn.BatchNorm2d(c), HexConv(c, c), nn.BatchNorm2d(c)

    def forward(self, x, mask):
        y = torch.relu(self.b1(self.c1(x))) * mask
        return torch.relu(x + self.b2(self.c2(y))) * mask


class HexNet(nn.Module):
    def __init__(self, width, blocks):
        super().__init__()
        self.stem, self.bn = HexConv(9, width), nn.BatchNorm2d(width)
        self.blocks = nn.ModuleList(HexBlock(width) for _ in range(blocks))
        self.head = Head(width)
        self.register_buffer('flat', FLAT, persistent=False)

    def forward(self, x):
        mask = x[:, 4:5]
        h = torch.relu(self.bn(self.stem(x))) * mask
        for b in self.blocks:
            h = b(h, mask)
        return self.head(h.flatten(2)[:, :, self.flat].transpose(1, 2))


class TBlock(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.f1, self.f2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.bias = nn.Parameter(torch.zeros(heads, 169))

    def forward(self, x, rel):
        B, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-1, -2)) * (d // self.h) ** -0.5 + self.bias[:, rel]
        x = x + self.o((a.softmax(-1) @ v).transpose(1, 2).reshape(B, T, d))
        return x + self.f2(F.gelu(self.f1(self.n2(x))))


class HexFormer(nn.Module):
    def __init__(self, width, blocks, heads=4, ffmul=2):
        super().__init__()
        self.embed = nn.Linear(9, width)
        self.pos = nn.Parameter(torch.zeros(CELLS, width))
        self.blocks = nn.ModuleList(TBlock(width, heads, ffmul * width) for _ in range(blocks))
        self.norm = nn.LayerNorm(width)
        self.head = Head(width)
        c = torch.tensor(COORDS)
        d = c[:, None] - c[None]
        self.register_buffer('rel', (d[..., 0] + 6) * 13 + d[..., 1] + 6, persistent=False)
        self.register_buffer('flat', FLAT, persistent=False)

    def forward(self, x):
        h = self.embed(x.flatten(2)[:, :, self.flat].transpose(1, 2)) + self.pos
        for b in self.blocks:
            h = b(h, self.rel)
        return self.head(self.norm(h))


def build(config):
    kind = config['kind']
    if kind == 'hex':
        return HexNet(config['width'], config['blocks'])
    if kind == 'former':
        return HexFormer(config['width'], config['blocks'], config.get('heads', 4), config.get('ffmul', 2))
    return PolicyValue(ModelConfig(**config))


def load(path, device='cpu'):
    data = torch.load(path, map_location=device, weights_only=False)
    m = build(data['config']).to(device)
    m.load_state_dict(data['model'])
    return m.eval(), data


def planes(board, meta):
    B = board.shape[0]
    x = torch.zeros(B, 9, 49, device=board.device)
    flat = FLAT.to(board.device)
    for i, p in enumerate((1, 2, -1, -2)):
        x[:, i, flat] = (board == p).float()
    x[:, 4, flat] = 1
    m = meta.float()
    x[:, 5, flat] = m[:, 0:1] / 18
    x[:, 6, flat] = m[:, 1:2] / 18
    x[:, 7, flat] = m[:, 2:3]
    x[:, 8, flat] = m[:, 3:4]
    return x.view(B, 9, 7, 7)


def compact(x):
    c = x.reshape(len(x), 9, 49)[:, :, FLAT.numpy()]
    board = (c[:, 0] + 2 * c[:, 1] - c[:, 2] - 2 * c[:, 3]).astype(np.int8)
    meta = np.stack([np.rint(c[:, 5, 0] * 18), np.rint(c[:, 6, 0] * 18), c[:, 7, 0], c[:, 8, 0]], 1).astype(np.uint8)
    return board, meta


CELL_PERM = torch.tensor(np.stack([np.argsort(p) for p, _ in SYMMETRIES]))
ACTION_PERM = torch.tensor(np.stack([ap for _, ap in SYMMETRIES]))
