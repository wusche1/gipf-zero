from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
import gipf_engine as ge

from training.async_selfplay import AsyncNativeBatchedMCTS, SharedInferenceQueue
from training.model import ModelConfig, PolicyValue
from training.native_search import NativeBatchedMCTS, NativeNode


def capture_rich_states(count=8):
    rng = np.random.default_rng(0); state = ge.State()
    while not state.winner:
        if state.phase == 'capture': return [state.clone() for _ in range(count)]
        actions = state.legal_actions(); state.apply(actions[int(rng.integers(len(actions)))])
    raise AssertionError('seeded trajectory did not reach a capture')


def test_async_queue_matches_native_forest_and_coalesces_cpu_requests():
    torch.manual_seed(31)
    model = PolicyValue(ModelConfig('mlp', 16, 1)).eval()
    baseline = NativeBatchedMCTS(model, 'cpu', seed=5)
    states = capture_rich_states()
    expected_roots = [NativeNode(state.clone()) for state in states]
    expected = baseline.search(expected_roots, 16, noise=True)
    queue = SharedInferenceQueue(model, 'cpu', max_batch=32, coalesce_ms=20)
    try:
        def run(seed):
            search = AsyncNativeBatchedMCTS(queue, seed=seed)
            roots = [NativeNode(state.clone()) for state in states]
            return search.search(roots, 16, noise=True), roots
        with ThreadPoolExecutor(max_workers=2) as pool:
            (actual, actual_roots), _ = list(pool.map(run, (5, 9)))
        for left, right in zip(expected, actual): assert np.array_equal(left, right)
        for left, right in zip(expected_roots, actual_roots):
            assert np.array_equal(left.n, right.n) and np.array_equal(left.w, right.w)
        assert max(queue.batch_sizes) >= 16
        old_version = queue._version
        queue.update(lambda m: None)
        assert queue._version == old_version + 1
    finally:
        queue.close()


def test_async_queue_propagates_worker_exception_without_hanging():
    class Broken(torch.nn.Module):
        def forward(self, _): raise RuntimeError('intentional worker failure')
    queue = SharedInferenceQueue(Broken(), 'cpu')
    try:
        with pytest.raises(RuntimeError, match='intentional worker failure'):
            queue.submit(np.zeros((1, 9, 7, 7), np.float32), np.zeros((1, 1), np.int32), 0, timeout=2)
    finally:
        queue.close()


def test_async_queue_update_barrier_and_weight_version():
    torch.manual_seed(32)
    model = PolicyValue(ModelConfig('mlp', 16, 1)).eval()
    queue = SharedInferenceQueue(model, 'cpu')
    features = ge.encode_batch([ge.State()])
    indices = np.asarray([ge.State().legal_actions()], dtype=np.int32)
    try:
        with queue.epoch() as version:
            before, _ = queue.submit(features, indices, version)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with queue.epoch():
                waiting = pool.submit(queue.update, lambda m: next(m.parameters()).data.add_(.1), timeout=1)
                assert not waiting.done()
            waiting.result(timeout=2)
        with queue.epoch() as version:
            after, _ = queue.submit(features, indices, version)
        assert queue._version == 1
        assert not np.array_equal(before, after)
    finally:
        queue.close()


def test_async_queue_matches_native_forest_across_complete_game():
    torch.manual_seed(33)
    model = PolicyValue(ModelConfig('mlp', 16, 1)).eval()
    baseline = NativeBatchedMCTS(model, 'cpu', seed=12)
    queue = SharedInferenceQueue(model, 'cpu')
    asynchronous = AsyncNativeBatchedMCTS(queue, seed=12)
    left, right = NativeNode(ge.State()), NativeNode(ge.State())
    try:
        for _ in range(150):
            expected = baseline.search([left], 4, noise=True)[0]
            actual = asynchronous.search([right], 4, noise=True)[0]
            assert np.array_equal(expected, actual)
            selected = int(np.argmax(expected))
            left, right = left.children.get(selected), right.children.get(selected)
            assert left is not None and right is not None
            assert left.state.serialize() == right.state.serialize()
            if left.state.winner: break
        else: raise AssertionError('complete asynchronous game exceeded decision bound')
    finally:
        queue.close()
