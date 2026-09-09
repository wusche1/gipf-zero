import numpy as np
import torch
import gipf_engine as ge

from training.model import ModelConfig, PolicyValue
from training.native_search import NativeBatchedMCTS, NativeNode
from training.search import BatchedMCTS, Node


GEO = ge.geometry()


def trajectory_states(seed=0, count=20):
    rng = np.random.default_rng(seed)
    state = ge.State()
    result = []
    saw_capture = False
    while len(result) < count and not state.winner:
        result.append(state.clone())
        saw_capture |= state.phase == 'capture'
        actions = state.legal_actions()
        state.apply(actions[int(rng.integers(len(actions)))])
    assert saw_capture  # Includes real capture-phase state/action chronology.
    return result


def terminal_capture_state():
    board = [0] * 37
    line = GEO['lines'][3]
    for cell in line[:4]:
        board[cell] = 1
    board[line[4]] = 2
    board[line[5]] = -2
    used = [sum(abs(x) for x in board if x > 0), sum(abs(x) for x in board if x < 0)]
    return ge.State.from_dict({'board': board, 'reserves': [8, 10],
                               'captured': [18 - 8 - used[0], 18 - 10 - used[1]],
                               'current_player': 1, 'turn_player': 1,
                               'phase': 'capture', 'winner': 0, 'ply': 1})


def test_native_forest_matches_reference_through_capture_states_and_root_reuse():
    torch.manual_seed(44)
    model = PolicyValue(ModelConfig('mlp', 32, 1)).eval()
    states = trajectory_states()
    reference = BatchedMCTS(model, 'cpu', seed=13, native_expand=True)
    native = NativeBatchedMCTS(model, 'cpu', seed=13)
    reference_roots = [Node(state.clone()) for state in states]
    native_roots = [NativeNode(state.clone()) for state in states]
    expected = reference.search(reference_roots, 32, noise=True)
    actual = native.search(native_roots, 32, noise=True)
    for left, right, expected_policy, actual_policy in zip(reference_roots, native_roots, expected, actual):
        assert np.array_equal(left.actions, right.actions)
        assert np.array_equal(left.n, right.n)
        assert np.array_equal(left.w, right.w)
        assert np.array_equal(expected_policy, actual_policy)

    # The selected child remains a valid native root in a second search.
    selected = int(np.argmax(native_roots[0].n))
    child = native_roots[0].children.get(selected)
    assert child is not None
    assert child.state.serialize() == reference_roots[0].children[selected].state.serialize()
    followup = native.search([child], 8)
    assert followup[0].sum() == 1


def test_native_forest_terminal_leaf_backs_up_absolute_winner():
    root = NativeNode(terminal_capture_state())
    forest = ge.NativeForest([root._node])
    initial = forest.expand_unexpanded_roots()
    logits = np.zeros((1, 2730), dtype=np.float32)
    # Remove both final doubles: FAQ terminal winner is white (+1).
    action = 42 + 3 * 128 + (1 << 4) + (1 << 5)
    logits[0, action] = 10
    forest.finish(logits, np.zeros(1, dtype=np.float32))
    assert forest.select_leaves(1.5) == []
    index = int(np.where(root.actions == action)[0][0])
    assert root.n[index] == 1
    assert root.w[index] == 1.0


def test_native_forest_matches_reference_across_a_complete_game():
    torch.manual_seed(71)
    model = PolicyValue(ModelConfig('mlp', 32, 1)).eval()
    reference_search = BatchedMCTS(model, 'cpu', seed=8, native_expand=True)
    native_search = NativeBatchedMCTS(model, 'cpu', seed=8)
    reference = Node(ge.State())
    native = NativeNode(ge.State())
    for _ in range(300):
        expected = reference_search.search([reference], 4, noise=True)[0]
        actual = native_search.search([native], 4, noise=True)[0]
        assert np.array_equal(expected, actual)
        chosen = int(np.argmax(expected))
        reference = reference.children[chosen]
        native = native.children.get(chosen)
        assert native is not None
        assert native.state.serialize() == reference.state.serialize()
        native.state.validate()
        if native.state.winner:
            assert native.state.winner == reference.state.winner
            break
    else:
        raise AssertionError('complete native self-play game exceeded decision bound')
