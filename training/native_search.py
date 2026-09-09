"""Optional native-rule-tree MCTS adapter.

This is intentionally separate from ``BatchedMCTS`` while it is evaluated.
Its tree traversal and backup are C++, while PyTorch inference remains batched
in Python.  ``NativeNode`` mirrors the public fields training uses, including
``children.get(action)`` for root reuse after a chosen move.
"""
from contextlib import nullcontext
import math
import time

import numpy as np
import torch
import gipf_engine as ge

from .inference import CudaGraphInference
from .model import encode


class _Children:
    def __init__(self, node):
        self.node = node

    def get(self, index, default=None):
        child = self.node._node.existing_child(int(index))
        return default if child is None else NativeNode(_node=child)


class NativeNode:
    """Python-compatible view of a C++ native search node."""
    def __init__(self, state=None, _node=None):
        if not hasattr(ge, 'NativeNode'):
            raise RuntimeError('gipf_engine lacks NativeNode; rebuild the native extension')
        self._node = ge.NativeNode(state) if _node is None else _node
        self.children = _Children(self)

    @property
    def state(self): return self._node.state
    @property
    def actor(self): return self._node.actor
    @property
    def actions(self): return np.asarray(self._node.actions, dtype=np.int32)
    @property
    def p(self): return np.asarray(self._node.priors, dtype=np.float64)
    @property
    def n(self): return np.asarray(self._node.visits, dtype=np.int32)
    @property
    def w(self): return np.asarray(self._node.values, dtype=np.float64)
    def policy(self):
        visits = self.n
        return self.p.copy() if visits.sum() == 0 else visits / visits.sum()


class NativeBatchedMCTS:
    """Experimental batched MCTS using ``gipf_engine.NativeForest``."""
    def __init__(self, model, device='cuda', cpuct=1.5, seed=0, cuda_graph_batch=0):
        if not hasattr(ge, 'NativeForest'):
            raise RuntimeError('gipf_engine lacks NativeForest; rebuild the native extension')
        self.model = model
        self.device = device
        self.cpuct = cpuct
        self.rng = np.random.default_rng(seed)
        self.evaluations = 0
        self.inference = (CudaGraphInference(model, device, cuda_graph_batch)
                          if cuda_graph_batch and str(device).startswith('cuda') else None)

    @torch.inference_mode()
    def _evaluate(self, forest, leaves):
        if not leaves:
            return
        encoded = ge.encode_batch(leaves) if hasattr(ge, 'encode_batch') else np.stack([encode(s) for s in leaves])
        features = torch.from_numpy(encoded)
        if self.inference is None:
            logits, values = self.model(features.to(self.device))
        else:
            logits, values = self.inference.forward(features)
        forest.finish(logits.float().cpu().numpy(), values.float().cpu().numpy())
        self.evaluations += len(leaves)

    def search(self, roots, simulations=64, noise=False, deadline=None):
        if any(root.state.winner for root in roots):
            raise ValueError('cannot search from a terminal root')
        forest = ge.NativeForest([root._node for root in roots])
        scope = self.inference.search_scope() if self.inference is not None else nullcontext()
        with scope:
            self._evaluate(forest, forest.expand_unexpanded_roots())
            if noise:
                for root in roots:
                    actions = root.actions
                    if len(actions):
                        eta = self.rng.dirichlet(np.full(len(actions), 10 / len(actions)))
                        root._node.mix_noise(eta)
            for _ in range(simulations):
                if deadline is not None and time.monotonic() >= deadline:
                    break
                self._evaluate(forest, forest.select_leaves(self.cpuct))
        return [root.policy() for root in roots]
