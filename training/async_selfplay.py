"""Experimental shared-model inference queue for independent native forests.

Each search submits at most one pending-leaf batch and waits for its result.
The worker coalesces simultaneous searches into one PyTorch forward.  Model
updates must use ``update``: it waits for all inference epochs, so a search
never mixes policy/value outputs from different weight versions.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from queue import Queue, Empty
from threading import Condition, Event, Lock, Thread
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import gipf_engine as ge


@dataclass
class _Request:
    features: np.ndarray
    indices: np.ndarray
    version: int
    ready: Event = field(default_factory=Event)
    result: tuple[np.ndarray, np.ndarray] | None = None
    error: BaseException | None = None
    submitted_at: float = field(default_factory=time.perf_counter)


class SharedInferenceQueue:
    def __init__(self, model, device='cuda', max_batch=512, coalesce_ms=2):
        self.model, self.device = model, device
        self.max_batch, self.coalesce_s = max_batch, coalesce_ms / 1000
        self._queue = Queue(); self._condition = Condition(); self._version = 0
        self._active_epochs = 0; self._closed = False
        self.batch_sizes = deque(maxlen=256); self.wait_seconds = deque(maxlen=256)
        self.batch_count = self.batch_total = 0
        self._worker = Thread(target=self._run, name='gipf-inference', daemon=True)
        self._worker.start()

    @contextmanager
    def epoch(self):
        with self._condition:
            if self._closed: raise RuntimeError('inference queue is closed')
            self._active_epochs += 1; version = self._version
        try:
            yield version
        finally:
            with self._condition:
                self._active_epochs -= 1; self._condition.notify_all()

    def update(self, fn, timeout=30):
        """Run an in-place model update only after all search epochs settle."""
        with self._condition:
            end = time.monotonic() + timeout
            while self._active_epochs:
                remaining = end - time.monotonic()
                if remaining <= 0: raise TimeoutError('active inference epoch did not settle before update timeout')
                self._condition.wait(remaining)
            result = fn(self.model); self._version += 1
            return result

    def submit(self, features, indices, version, timeout=30):
        with self._condition:
            if self._closed: raise RuntimeError('inference queue is closed')
        request = _Request(features, indices, version)
        self._queue.put(request)
        if not request.ready.wait(timeout):
            raise TimeoutError('shared inference worker did not respond before timeout')
        if request.error: raise request.error
        return request.result

    def close(self):
        with self._condition: self._closed = True
        self._queue.put(None); self._worker.join(timeout=5)
        if self._worker.is_alive(): raise RuntimeError('inference worker did not stop within close timeout')

    def _run(self):
        deferred = None
        while True:
            first = deferred if deferred is not None else self._queue.get()
            deferred = None
            if first is None: self._queue.task_done(); return
            group, total, stop_after_group = [first], len(first.features), False
            until = time.monotonic() + self.coalesce_s
            while total < self.max_batch:
                try: item = self._queue.get(timeout=max(0, until - time.monotonic()))
                except Empty: break
                if item is None:
                    stop_after_group = True
                    break
                if total + len(item.features) > self.max_batch:
                    deferred = item; break
                group.append(item); total += len(item.features)
            try:
                with torch.inference_mode():
                    features = torch.from_numpy(np.concatenate([r.features for r in group])).to(self.device)
                    logits, values = self.model(features)
                    width = max(r.indices.shape[1] for r in group)
                    all_indices = np.zeros((total, width), dtype=np.int64)
                    offset = 0
                    for request in group:
                        size, request_width = len(request.features), request.indices.shape[1]
                        all_indices[offset:offset + size, :request_width] = request.indices
                        offset += size
                    selected = torch.gather(logits.float(), 1, torch.from_numpy(all_indices).to(logits.device))
                    selected_cpu, values_cpu = selected.cpu().numpy(), values.float().cpu().numpy()
                    offset = 0
                    for request in group:
                        size, request_width = len(request.features), request.indices.shape[1]
                        request.result = (selected_cpu[offset:offset + size, :request_width].copy(),
                                          values_cpu[offset:offset + size].copy())
                        offset += size
                self.batch_sizes.append(total)
                self.wait_seconds.extend(time.perf_counter() - request.submitted_at for request in group)
                self.batch_count += 1; self.batch_total += total
            except BaseException as exc:
                for request in group: request.error = exc
            finally:
                for request in group:
                    request.ready.set(); self._queue.task_done()
            if stop_after_group:
                self._queue.task_done()
                return


class AsyncNativeBatchedMCTS:
    """Native Forest MCTS that shares a ``SharedInferenceQueue`` with peers."""
    def __init__(self, inference_queue, cpuct=1.5, seed=0):
        self.queue, self.cpuct = inference_queue, cpuct
        self.model, self.device = inference_queue.model, inference_queue.device
        self.inference = None
        self.rng = np.random.default_rng(seed); self.evaluations = 0; self._outstanding = False

    def _evaluate(self, forest, count, version):
        if not count: return
        if self._outstanding: raise RuntimeError('only one outstanding request is allowed per forest')
        self._outstanding = True
        try:
            selected, values = self.queue.submit(forest.encode_pending(), forest.pending_action_indices(), version)
            forest.finish_selected(selected, values); self.evaluations += count
        finally:
            self._outstanding = False

    def search(self, roots, simulations=64, noise=False, deadline=None):
        if any(root.state.winner for root in roots): raise ValueError('cannot search from a terminal root')
        forest = ge.NativeForest([root._node for root in roots])
        with self.queue.epoch() as version:
            self._evaluate(forest, forest.expand_unexpanded_root_pending(), version)
            if noise:
                for root in roots:
                    if len(root.actions): root._node.mix_noise(self.rng.dirichlet(np.full(len(root.actions), 10 / len(root.actions))))
            for _ in range(simulations):
                if deadline is not None and time.monotonic() >= deadline: break
                self._evaluate(forest, forest.select_pending(self.cpuct), version)
        return [root.policy() for root in roots]


class AsyncNativeForestPool:
    """Drop-in search facade that partitions training roots across forests.

    It has the same ``search``, ``rng`` and ``evaluations`` surface as the
    synchronous searcher.  It is intentionally opt-in: Python threads still
    serialize C++ forest calls while they hold the GIL, but their waits can
    overlap GPU inference in the queue worker.
    """
    def __init__(self, model, device='cuda', cpuct=1.5, seed=0, workers=4, max_batch=512,
                 cuda_graph_batch=0):
        if cuda_graph_batch:
            raise ValueError('CUDA graphs are not supported by the shared inference queue')
        self.model, self.device, self.cpuct = model, device, cpuct
        self.rng = np.random.default_rng(seed); self.workers = workers; self.evaluations = 0
        self.inference = None
        self.queue = SharedInferenceQueue(model, device, max_batch=max_batch)

    def search(self, roots, simulations=64, noise=False, deadline=None):
        groups = [roots[i::self.workers] for i in range(self.workers)]
        groups = [(i, group) for i, group in enumerate(groups) if group]
        seeds = [int(self.rng.integers(2**63)) for _ in groups]
        def work(item):
            (index, group), seed = item
            search = AsyncNativeBatchedMCTS(self.queue, self.cpuct, seed)
            return index, search.search(group, simulations, noise, deadline), search.evaluations
        with ThreadPoolExecutor(max_workers=len(groups)) as pool:
            completed = list(pool.map(work, zip(groups, seeds)))
        self.evaluations += sum(item[2] for item in completed)
        policies = [None] * len(roots)
        for index, values, _ in completed:
            policies[index::self.workers] = values
        return policies

    def close(self): self.queue.close()
