"""Reproduce query-transformer packed capture-query timing.

Example:
  PYTHONPATH=/workspace/gipf /venv/main/bin/python benchmarks/benchmark_query_packing.py --device cuda
"""
import argparse
import json
import statistics
import time

import torch
import gipf_engine as ge

from training.model import ModelConfig
from training.query_transformer import QueryTransformer


def capture_state(player):
    board = [0] * 37
    line = ge.geometry()['lines'][3 if player == 1 else 9]
    for cell in line[:4]: board[cell] = player
    board[line[4]], board[line[5]] = 2 * player, -2 * player
    used = [sum(abs(piece) for piece in board if piece > 0),
            sum(abs(piece) for piece in board if piece < 0)]
    reserves = [8, 10] if player == 1 else [10, 8]
    return ge.State.from_dict({
        'board': board, 'reserves': reserves,
        'captured': [18 - reserves[0] - used[0], 18 - reserves[1] - used[1]],
        'current_player': player, 'turn_player': player, 'phase': 'capture',
        'winner': 0, 'ply': 1 if player == 1 else 2,
    })


def time_call(fn, states, cuda):
    if cuda: torch.cuda.synchronize()
    start = time.perf_counter()
    logits, values = fn(states)
    if cuda: torch.cuda.synchronize()
    return time.perf_counter() - start, logits, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--batch', type=int, action='append', default=[])
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but unavailable')
    if args.device == 'cpu': torch.set_num_threads(1)
    batches = args.batch or ([64, 256] if args.device == 'cuda' else [32])
    model = QueryTransformer(ModelConfig('transformer', 64, 2, 'query')).to(args.device).eval()
    result = {'device': args.device, 'model': {'kind': 'transformer', 'width': 64, 'blocks': 2,
              'checkpoint': None, 'initialization': 'random'}, 'repeats': args.repeats,
              'state_cycle': ['initial_push', 'white_capture', 'black_capture', 'initial_push'],
              'tolerance': {'rtol': 2e-5 if args.device == 'cuda' else 1e-5,
                            'atol': 2e-6 if args.device == 'cuda' else 1e-6}, 'batches': []}
    with torch.inference_mode():
        for batch in batches:
            cycle = [ge.State(), capture_state(1), capture_state(-1), ge.State()]
            states = (cycle * ((batch + 3) // 4))[:batch]
            for _ in range(3 if args.device == 'cuda' else 2):
                model(states); model.forward_reference(states)
            packed, reference = [], []
            for _ in range(args.repeats):
                seconds, packed_logits, packed_values = time_call(model, states, args.device == 'cuda'); packed.append(seconds)
                seconds, reference_logits, reference_values = time_call(model.forward_reference, states, args.device == 'cuda'); reference.append(seconds)
            torch.testing.assert_close(packed_logits, reference_logits,
                                       rtol=result['tolerance']['rtol'], atol=result['tolerance']['atol'])
            torch.testing.assert_close(packed_values, reference_values,
                                       rtol=result['tolerance']['rtol'], atol=result['tolerance']['atol'])
            packed_median, reference_median = statistics.median(packed), statistics.median(reference)
            result['batches'].append({'size': batch, 'packed_seconds': packed,
                                      'reference_seconds': reference,
                                      'packed_median_seconds': packed_median,
                                      'reference_median_seconds': reference_median,
                                      'speedup': reference_median / packed_median,
                                      'policy_value_close': True})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
