import argparse
import json
import os
import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval
from training import net

TAPS = [(ky, kx) for ky in range(3) for kx in range(3) if (ky, kx) not in ((0, 0), (2, 2))]


def fuse(conv, bn):
    plain = nn.Conv2d(conv.in_channels, conv.out_channels, 3, padding=1, bias=False).eval()
    plain.weight.data = conv.weight.data * conv.m
    return fuse_conv_bn_eval(plain, bn)


def bake(model):
    model.stem, model.bn = fuse(model.stem, model.bn), nn.Identity()
    for b in model.blocks:
        b.c1, b.b1, b.c2, b.b2 = fuse(b.c1, b.b1), nn.Identity(), fuse(b.c2, b.b2), nn.Identity()
    return model


def linear(conv, bn):
    f = fuse(conv, bn)
    l = nn.Linear(7 * f.in_channels, f.out_channels)
    l.weight.data = torch.cat([f.weight[:, :, ky, kx] for ky, kx in TAPS], 1)
    l.bias.data = f.bias
    return l


class TokenHex(nn.Module):
    def __init__(self, m):
        super().__init__()
        cell = {rc: i for i, rc in enumerate(zip(net.ROWS.tolist(), net.COLS.tolist()))}
        self.register_buffer('nb', torch.tensor([cell.get((r + ky - 1, c + kx - 1), 37) for r, c in cell for ky, kx in TAPS]))
        self.register_buffer('flat', m.flat)
        self.stem = linear(m.stem, m.bn)
        self.blocks = nn.ModuleList(nn.ModuleList([linear(b.c1, b.b1), linear(b.c2, b.b2)]) for b in m.blocks)
        self.head = m.head

    def conv(self, l, h):
        return l(F.pad(h, (0, 0, 0, 1))[:, self.nb].reshape(-1, 37, l.in_features))

    def forward(self, x):
        h = torch.relu(self.conv(self.stem, x.flatten(2)[:, :, self.flat].transpose(1, 2)))
        for c1, c2 in self.blocks:
            h = torch.relu(h + self.conv(c2, torch.relu(self.conv(c1, h))))
        return self.head(h)


def export(model, output, tokens=False):
    model = model.eval()
    params = sum(p.numel() for p in model.parameters())
    x = torch.zeros(5, 9, 49)
    x[:, :, net.FLAT] = torch.rand(5, 9, 37)
    x[:, 4, net.FLAT] = 1
    x = x.view(5, 9, 7, 7)
    with torch.no_grad():
        logits, value = model(x)
        if isinstance(model, net.HexNet):
            model = TokenHex(model) if tokens else bake(model)
        torch.onnx.export(model, x, output, input_names=['board_features'], output_names=['policy_logits', 'value'],
                          dynamic_axes={k: {0: 'batch'} for k in ('board_features', 'policy_logits', 'value')},
                          opset_version=17, do_constant_folding=True, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    got = ort.InferenceSession(output, options, providers=['CPUExecutionProvider']).run(None, {'board_features': x.numpy()})
    error = float(max(np.abs(got[0] - logits.numpy()).max(), np.abs(got[1] - value.numpy()).max()))
    assert error < 1e-4, error
    return {'file': output, 'params': params, 'onnx_bytes': os.path.getsize(output), 'max_error': error}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint')
    p.add_argument('--kind')
    p.add_argument('--width', type=int)
    p.add_argument('--blocks', type=int)
    p.add_argument('--output', required=True)
    p.add_argument('--tokens', action='store_true')
    a = p.parse_args()
    model = net.load(a.checkpoint)[0] if a.checkpoint else net.build({'kind': a.kind, 'width': a.width, 'blocks': a.blocks})
    print(json.dumps(export(model, a.output, a.tokens)))
