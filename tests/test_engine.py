import pytest

import gipf_engine


G = gipf_engine.geometry()


def index(q, r):
    return G["coordinates"].index([q, r])


def make_state(board, reserves=(10, 10), current=1, turn=None, phase="capture", ply=None):
    """Make a material-conserving test state from visible board material."""
    board = list(board)
    used = [0, 0]
    for p in board:
        if p:
            used[0 if p > 0 else 1] += abs(p)
    captured = [18 - reserves[i] - used[i] for i in range(2)]
    assert min(captured) >= 0
    if turn is None:
        turn = current if phase == "capture" else -current
    if ply is None:
        ply = 1 if turn == 1 else 2
    return gipf_engine.State.from_dict(
        {
            "board": board,
            "reserves": list(reserves),
            "captured": captured,
            "current_player": current,
            "turn_player": turn,
            "phase": phase,
            "winner": 0,
            "ply": ply,
        }
    )


def empty_board():
    return [0] * 37


def material(state, colour):
    return (
        state.reserves[0 if colour == 1 else 1]
        + state.captured[0 if colour == 1 else 1]
        + sum(abs(p) for p in state.board if (p > 0) == (colour > 0))
    )


def capture_action(line_id, mask=0):
    return 42 + line_id * 128 + mask


def test_opening_geometry_and_round_trip():
    state = gipf_engine.State()
    assert state.current_player == 1
    assert state.phase == "push"
    assert state.reserves == [12, 12]
    assert sum(abs(p) == 2 for p in state.board) == 6
    assert len(state.legal_actions()) == 42
    assert len(G["coordinates"]) == 37
    assert [len(line) for line in G["lines"]] == [4, 5, 6, 7, 6, 5, 4] * 3
    assert len(G["rays"]) == 42
    assert gipf_engine.State.from_dict(state.serialize()).serialize() == state.serialize()
    clone = state.clone()
    clone.apply(0)
    assert state.serialize() != clone.serialize()


def test_pushes_shift_only_contiguous_prefix_and_refuse_full_ray():
    board = empty_board()
    ray = G["rays"][6]  # q=0, ordered r=-3..3
    board[ray[0]], board[ray[1]] = -1, 1
    board[index(-3, 0)], board[index(3, -3)] = 2, -2
    state = make_state(board, reserves=(8, 8), phase="push")
    state.apply(6)
    assert state.board[ray[0:3][0]] == 1
    assert [state.board[i] for i in ray[:4]] == [1, -1, 1, 0]

    full = empty_board()
    for n, cell in enumerate(G["rays"][0]):
        full[cell] = 1 if n % 2 == 0 else -1
    full[index(0, -3)], full[index(3, -3)] = 2, -2
    blocked = make_state(full, reserves=(10, 10), phase="push")
    assert 0 not in blocked.legal_actions() and 1 not in blocked.legal_actions()
    with pytest.raises(ValueError):
        blocked.apply(0)


def test_capture_lists_optional_double_masks_and_returns_own_double_as_two():
    board = empty_board()
    line_id = 3  # q=0, length seven
    line = G["lines"][line_id]
    for cell in line[:4]:
        board[cell] = 1
    board[line[4]] = 2
    board[index(3, -3)] = 2  # a surviving own GIPF
    board[index(3, 0)] = -2  # opponent keeps one too
    state = make_state(board, reserves=(8, 10))
    masks = {a - 42 - line_id * 128 for a in state.legal_actions() if (a - 42) // 128 == line_id}
    assert masks == {0, 1 << 4}
    state.apply(capture_action(line_id, 1 << 4))
    assert state.reserves[0] == 14  # four singles plus the two basics in a GIPF
    assert state.board[line[4]] == 0
    assert state.winner == 0


def test_intersecting_rows_can_leave_or_remove_shared_double():
    board = empty_board()
    q_line, r_line = 3, 10
    for cell in G["lines"][q_line][:4]:
        board[cell] = 1
    for cell in G["lines"][r_line][:4]:
        board[cell] = 1
    shared = index(0, 0)
    board[shared] = 2
    board[index(3, -3)] = 2
    board[index(3, 0)] = -2
    state = make_state(board, reserves=(6, 10))
    assert capture_action(q_line, 0) in state.legal_actions()
    assert capture_action(r_line, 0) in state.legal_actions()
    left = state.clone()
    left.apply(capture_action(q_line, 0))
    assert left.phase == "capture"
    assert capture_action(r_line, 0) in left.legal_actions()  # shared GIPF preserves the other row
    state.apply(capture_action(q_line, 1 << 3))
    assert state.phase == "push" and state.board[shared] == 0


def test_original_mover_has_capture_priority_over_opponent_row():
    board = empty_board()
    for cell in G["lines"][0]:
        board[cell] = 1
    for cell in G["lines"][6]:
        board[cell] = -1
    board[index(0, -3)] = 2
    board[index(0, 3)] = -2
    state = make_state(board, reserves=(12, 12), current=1, turn=1, ply=1)
    actions = state.legal_actions()
    assert actions and all((a - 42) // 128 == 0 for a in actions)
    assert state.current_player == 1
    state.apply(capture_action(0))
    assert state.phase == "capture" and state.current_player == -1
    assert capture_action(6) in state.legal_actions()


def test_last_opponent_double_wins_even_when_own_last_double_is_removed():
    board = empty_board()
    line_id = 3
    line = G["lines"][line_id]
    for cell in line[:4]:
        board[cell] = 1
    board[line[4]] = 2
    board[line[5]] = -2
    state = make_state(board, reserves=(8, 10))
    state.apply(capture_action(line_id, (1 << 4) | (1 << 5)))
    assert state.winner == 1


def test_capture_takes_contiguous_extension_but_stops_at_a_gap():
    board = empty_board()
    line_id = 3
    line = G["lines"][line_id]
    for cell in line[:4]:
        board[cell] = 1
    board[line[4]] = -1  # an opponent single directly extends the compulsory row
    board[line[6]] = 1   # separated by line[5], so it is not part of the capture
    board[index(-3, 0)] = 2
    board[index(3, -3)] = -2
    state = make_state(board, reserves=(8, 10))
    black_captured = state.captured[1]
    state.apply(capture_action(line_id))
    assert all(state.board[cell] == 0 for cell in line[:5])
    assert state.board[line[6]] == 1
    assert state.captured[1] == black_captured + 1


def test_reserve_loss_waits_until_captures_have_settled():
    board = empty_board()
    ray_id = 6
    ray = G["rays"][ray_id]
    for cell in ray[1:4]:
        board[cell] = 1
    board[index(3, -3)] = 2
    board[index(3, 0)] = -2
    state = make_state(board, reserves=(10, 0), current=1, turn=-1, phase="push", ply=2)
    state.apply(ray_id)
    assert state.phase == "capture" and state.winner == 0
    state.apply(capture_action(3, 0))
    assert state.winner == 1


def test_conservation_holds_through_random_legal_play():
    state = gipf_engine.State()
    for _ in range(200):
        assert material(state, 1) == material(state, -1) == 18
        gipf_engine.State.from_dict(state.serialize()).validate()
        actions = state.legal_actions()
        if not actions:
            break
        state.apply(actions[len(actions) // 2])
        if state.winner:
            break
    assert material(state, 1) == material(state, -1) == 18
    gipf_engine.State.from_dict(state.serialize()).validate()


def test_from_dict_rejects_malformed_or_inconsistent_untrusted_state():
    good = gipf_engine.State().serialize()
    bad_length = dict(good, board=good["board"][:-1])
    with pytest.raises(ValueError):
        gipf_engine.State.from_dict(bad_length)
    bad_reserve = dict(good, reserves=[19, 12])
    with pytest.raises(ValueError):
        gipf_engine.State.from_dict(bad_reserve)
    bad_winner = dict(good, winner=1)
    with pytest.raises(ValueError):
        gipf_engine.State.from_dict(bad_winner)
    too_many_doubles = dict(good, board=[2, 2, 2, 2] + [0] * 33, reserves=[10, 18], captured=[0, 0])
    with pytest.raises(ValueError):
        gipf_engine.State.from_dict(too_many_doubles)
    row_deferred_to_push = dict(good, phase="push", ply=1, turn_player=1, current_player=-1)
    line = G["lines"][3]
    board = list(row_deferred_to_push["board"])
    for cell in line[:4]:
        board[cell] = 1
    row_deferred_to_push["board"] = board
    row_deferred_to_push["reserves"] = [8, 12]
    row_deferred_to_push["captured"] = [4, 0]
    with pytest.raises(ValueError):
        gipf_engine.State.from_dict(row_deferred_to_push)
