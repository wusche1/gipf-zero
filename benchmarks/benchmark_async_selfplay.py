"""Bounded shared-inference comparison; run only when the GPU is available.

PYTHONPATH=/workspace/gipf /venv/main/bin/python benchmarks/benchmark_async_selfplay.py
"""
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import gipf_engine as ge
import numpy as np
import torch

from training.async_selfplay import AsyncNativeBatchedMCTS, SharedInferenceQueue
from training.evaluate import settle_opening
from training.model import load_model
from training.native_search import NativeBatchedMCTS, NativeNode


def make_states(count):
    return [settle_opening(ge.State(), 8, random.Random(1000 + i)) for i in range(count)]


def run_native(model, states, simulations=64):
    NativeBatchedMCTS(model, 'cuda', seed=0).search([NativeNode(s.clone()) for s in states[:16]], 4)
    start = time.perf_counter()
    policies = NativeBatchedMCTS(model, 'cuda', seed=0).search([NativeNode(s.clone()) for s in states], simulations, noise=False)
    torch.cuda.synchronize()
    return time.perf_counter() - start, policies


def run_async(model, states, workers=4, simulations=64):
    roots_per_worker = len(states) // workers
    queue = SharedInferenceQueue(model, 'cuda', max_batch=workers * roots_per_worker, coalesce_ms=2)
    try:
        AsyncNativeBatchedMCTS(queue, seed=0).search([NativeNode(s.clone()) for s in states[:16]], 4)
        def work(worker):
            chunk = states[worker * roots_per_worker:(worker + 1) * roots_per_worker]
            return AsyncNativeBatchedMCTS(queue, seed=worker).search([NativeNode(s.clone()) for s in chunk], simulations, noise=False)
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool: chunks = list(pool.map(work, range(workers)))
        torch.cuda.synchronize()
        return time.perf_counter() - start, [p for chunk in chunks for p in chunk], list(queue.batch_sizes), list(queue.wait_seconds)
    finally:
        queue.close()


if __name__ == '__main__':
    torch.set_num_threads(2)
    model, data = load_model('checkpoints/pilot-mlp256-final.pt', 'cuda')
    states = make_states(128); native, async_result, exact = [], [], []
    for repeat in range(3):
        order = ('native', 'async') if repeat % 2 == 0 else ('async', 'native')
        results = {}
        for backend in order:
            results[backend] = run_native(model, states) if backend == 'native' else run_async(model, states)
        native.append(results['native'][0]); async_result.append(results['async'])
        exact.append(all(np.array_equal(a, b) for a, b in zip(results['native'][1], results['async'][1])))
    async_times = [item[0] for item in async_result]
    fills = [item for _, _, batches, _ in async_result for item in batches]
    waits = [item for _, _, _, request_waits in async_result for item in request_waits]
    print(json.dumps({'model_config': data['config'], 'workers': 4, 'roots_per_worker': 32,
                      'simulations': 64, 'native_seconds': native, 'async_seconds': async_times,
                      'native_median': statistics.median(native), 'async_median': statistics.median(async_times),
                      'speedup': statistics.median(native) / statistics.median(async_times),
                      'exact_policy_identity': all(exact),
                      'queue_batches': len(fills), 'mean_batch_fill': statistics.mean(fills),
                      'max_batch_fill': max(fills), 'mean_queue_wait_ms':1000 * statistics.mean(waits),
                      'p95_queue_wait_ms':1000 * sorted(waits)[int(.95 * (len(waits)-1))]}, indent=2))
