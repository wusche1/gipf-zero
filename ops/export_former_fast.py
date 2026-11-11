import argparse
import json
import os
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto as TP, helper as H, numpy_helper as NH
from training import net

N = net.CELLS


def export(model, output, act='gelu', norm='ln', check=True):
    model = model.eval()
    sd = {k: v.double().numpy() for k, v in model.state_dict().items()}
    d, heads = model.embed.out_features, model.blocks[0].h
    dh, rd = d // heads, d ** 0.5
    nodes, inits, count = [], [], [0]

    def const(a, dtype=np.float32):
        count[0] += 1
        inits.append(NH.from_array(np.ascontiguousarray(a, dtype), f'c{count[0]}'))
        return inits[-1].name

    def op(kind, ins, outs=1, **kw):
        count[0] += 1
        names = [f't{count[0]}_{i}' for i in range(outs)]
        nodes.append(H.make_node(kind, ins, names, **kw))
        return names[0] if outs == 1 else names

    def ints(*a):
        return const(a, np.int64)

    def center(w):
        return w - w.mean(-1, keepdims=True) if norm == 'ln' else w

    def gemm(x, w, b, **kw):
        return op('Gemm', [x, const(w.T), const(b)], **kw)

    def normed(x):
        if norm == 'none':
            return x
        s = op('Sqrt', [op('Add', [op('ReduceSumSquare', [x, ints(1)]), const(1e-5 * d)])])
        return op('Div', [x, s])

    def read(p, n):
        w, b = sd[p + '.weight'], sd[p + '.bias']
        return w * sd[n + '.weight'] * (1 if norm == 'none' else rd), b + w @ sd[n + '.bias']

    def write(p):
        return center(sd[p + '.weight'].T).T, center(sd[p + '.bias'])

    x = op('Gather', [op('Reshape', ['board_features', ints(9, 49)]), ints(*net.FLAT.tolist())], axis=1)
    h = gemm(x, center(sd['embed.weight'].T).T, center(sd['embed.bias'] + sd['pos']), transA=1)
    rel = model.rel.numpy()
    for i in range(len(model.blocks)):
        p = f'blocks.{i}.'
        w, b = read(p + 'qkv', p + 'n1')
        w[:d] *= dh ** -0.5
        b[:d] *= dh ** -0.5
        qkv = op('Transpose', [op('Reshape', [gemm(normed(h), w, b), ints(N, 3, heads, dh)])], perm=[1, 2, 0, 3])
        q, k, v = op('Split', [qkv], 3, axis=0, num_outputs=3)
        a = op('MatMul', [q, op('Transpose', [k], perm=[0, 1, 3, 2])])
        a = op('Softmax', [op('Add', [a, const(sd[p + 'bias'][:, rel][None])])], axis=-1)
        o = op('Reshape', [op('Transpose', [op('MatMul', [a, v])], perm=[2, 0, 1, 3]), ints(N, d)])
        h = op('Add', [h, gemm(o, *write(p + 'o'))])
        f = gemm(normed(h), *read(p + 'f1', p + 'n2'))
        f = op('Relu', [f]) if act == 'relu' else op('Gelu', [f], approximate='tanh' if act == 'tanh' else 'none')
        h = op('Add', [h, gemm(f, *write(p + 'f2'))])

    pw, pb = read('head.policy', 'norm')
    vw, vb = read('head.vproj', 'norm')
    vc = len(vb)
    pv = gemm(normed(h), np.concatenate([pw, vw]), np.concatenate([pb, vb]))
    v = op('Reshape', [op('Relu', [op('Slice', [pv, ints(10), ints(10 + vc), ints(1)])]), ints(1, N * vc)])
    v = op('Relu', [gemm(v, sd['head.value.0.weight'], sd['head.value.0.bias'])])
    v = op('Tanh', [gemm(v, sd['head.value.2.weight'], sd['head.value.2.bias'])])
    nodes.append(H.make_node('Reshape', [v, ints(1)], ['value']))

    push = [r[0] * (10 + vc) + net.DIRS.index(tuple(net.COORDS[r[1]] - net.COORDS[r[0]])) for r in net.RAYS]
    idx, scale = np.zeros((21, 14), np.int64), np.zeros((21, 14))
    for l, cells in enumerate(net.LINES):
        for j, c in enumerate(cells):
            idx[l, j], scale[l, j] = c * (10 + vc) + 9, 1
            idx[l, 7 + j], scale[l, 7 + j] = c * (10 + vc) + 6 + l // 7, 1 / len(cells)
    bits = np.array([[b >> j & 1 for b in range(128)] if j < 7 else [1] * 128 for j in range(14)])
    flat = op('Reshape', [pv, ints(-1)])
    g = op('Mul', [op('Gather', [flat, const(idx, np.int64)]), const(scale)])
    cap = op('Reshape', [op('MatMul', [g, const(bits)]), ints(21 * 128)])
    logits = op('Concat', [op('Gather', [flat, ints(*push)]), cap], axis=0)
    nodes.append(H.make_node('Reshape', [logits, ints(1, -1)], ['policy_logits']))

    graph = H.make_graph(nodes, 'former', [H.make_tensor_value_info('board_features', TP.FLOAT, [1, 9, 7, 7])],
                         [H.make_tensor_value_info('policy_logits', TP.FLOAT, [1, net.ACTIONS]),
                          H.make_tensor_value_info('value', TP.FLOAT, [1])], inits)
    onnx.save(H.make_model(graph, opset_imports=[H.make_opsetid('', 20)], ir_version=9), output)
    result = {'file': output, 'onnx_bytes': os.path.getsize(output)}
    if check:
        x = torch.zeros(8, 9, 49)
        x[:, :, net.FLAT] = torch.rand(8, 9, N)
        x[:, 4, net.FLAT] = 1
        x = x.view(8, 9, 7, 7)
        with torch.no_grad():
            logits, value = model(x)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        s = ort.InferenceSession(output, options, providers=['CPUExecutionProvider'])
        got = [s.run(None, {'board_features': x[i:i + 1].numpy()}) for i in range(8)]
        error = float(max(np.abs(np.concatenate([g[j] for g in got]) - t.numpy()).max() for j, t in enumerate((logits, value))))
        assert error < 1e-4, error
        result['max_error'] = error
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint')
    p.add_argument('--width', type=int)
    p.add_argument('--blocks', type=int)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--ffmul', type=int, default=2)
    p.add_argument('--act', default='gelu')
    p.add_argument('--norm', default='ln')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    model = net.load(a.checkpoint)[0] if a.checkpoint else net.HexFormer(a.width, a.blocks, a.heads, a.ffmul)
    print(json.dumps(export(model, a.output, a.act, a.norm, check=(a.act, a.norm) == ('gelu', 'ln'))))
