import numpy as np
import torch
import gipf_engine as ge

from training.model import ModelConfig, PolicyValue
from training.search import BatchedMCTS, Node


def sampled_states(count=48, seed=31):
    rng = np.random.default_rng(seed)
    state = ge.State()
    states = []
    while len(states) < count and not state.winner:
        states.append(state.clone())
        actions = state.legal_actions()
        state.apply(actions[int(rng.integers(len(actions)))])
    return states


def test_native_expand_policy_matches_reference_actions_and_softmax():
    assert hasattr(ge, 'expand_policy')
    rng = np.random.default_rng(17)
    for state in sampled_states():
        logits = rng.normal(size=2730).astype(np.float32)
        actions, priors = ge.expand_policy(state, logits)
        expected_actions = np.asarray(state.legal_actions(), dtype=np.int32)
        values = logits[expected_actions].astype(np.float64)
        values -= values.max()
        expected_priors = np.exp(values)
        expected_priors /= expected_priors.sum()
        assert np.array_equal(actions, expected_actions)
        np.testing.assert_allclose(priors, expected_priors, rtol=1e-15, atol=1e-15)


def test_native_expand_preserves_fixed_search_counts_and_policy():
    torch.manual_seed(42)
    model = PolicyValue(ModelConfig('mlp', 32, 1)).eval()
    states = sampled_states(16)
    reference = BatchedMCTS(model, 'cpu', seed=12, native_expand=False)
    native = BatchedMCTS(model, 'cpu', seed=12, native_expand=True)
    reference_roots = [Node(state.clone()) for state in states]
    native_roots = [Node(state.clone()) for state in states]
    expected = reference.search(reference_roots, simulations=32, noise=True)
    actual = native.search(native_roots, simulations=32, noise=True)
    for left, right, expected_policy, actual_policy in zip(
            reference_roots, native_roots, expected, actual):
        assert np.array_equal(left.actions, right.actions)
        assert np.array_equal(left.n, right.n)
        assert np.array_equal(left.w, right.w)
        assert np.array_equal(expected_policy, actual_policy)
