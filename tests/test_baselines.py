from __future__ import annotations

from copy import deepcopy
import json

from baselines.evaluate import EvaluationConfig, evaluate, summarize
from baselines.players import GreedyPlayer, MCTSPlayer, MinimaxPlayer, RandomPlayer, evaluate_state


class TreeState:
    """Tiny contract-shaped state with deliberate same-player continuations."""

    def __init__(self, node: str = "root", seed: int | None = None):
        self.node = node
        self.ply = 0
        self.board = [0] * 37
        self.reserves = [12, 12]
        self._set_node()

    def _set_node(self):
        # action -> target.  n1 is a capture-style extra decision for White.
        data = {
            "root": (1, "push", 0, {10: "n1", 20: "n2"}),
            "n1": (1, "capture", 0, {11: "white_win", 12: "black_win"}),
            "n2": (-1, "capture", 0, {21: "black_win"}),
            "white_win": (-1, "push", 1, {}),
            "black_win": (1, "push", -1, {}),
        }
        self.current_player, self.phase, self.winner, self._actions = data[self.node]

    @property
    def legal_actions(self):
        return list(self._actions)

    def clone(self):
        return deepcopy(self)

    def apply(self, action: int):
        self.node = self._actions[action]
        self.ply += 1
        self._set_node()


class WhiteWinsState:
    def __init__(self, seed: int | None = None):
        self.current_player, self.winner, self.ply = 1, 0, 0
        self.board, self.reserves = [0] * 37, [12, 12]

    @property
    def legal_actions(self):
        return [0] if not self.winner else []

    def clone(self):
        return deepcopy(self)

    def apply(self, action: int):
        assert action == 0
        self.winner = 1
        self.ply += 1


class EndlessState:
    def __init__(self, seed: int | None = None):
        self.current_player, self.winner, self.ply = 1, 0, 0
        self.board, self.reserves = [0] * 37, [12, 12]

    @property
    def legal_actions(self):
        return [0]

    def clone(self):
        return deepcopy(self)

    def apply(self, action: int):
        self.current_player *= -1
        self.ply += 1


class FirstAction:
    def choose_action(self, state, time_limit=None):
        return state.legal_actions[0]


def test_players_choose_legal_action_and_minimax_handles_extra_decision():
    state = TreeState()
    assert RandomPlayer(seed=7).choose_action(state) in state.legal_actions
    assert GreedyPlayer().choose_action(state) == 10
    # White must maximize again at n1, rather than blindly negating its value.
    assert MinimaxPlayer(depth=2).choose_action(state) == 10
    assert MCTSPlayer(simulations=12, seed=1).choose_action(state) in state.legal_actions


def test_evaluation_is_colour_symmetric_and_scores_terminal():
    white = TreeState("white_win")
    black = TreeState("black_win")
    assert evaluate_state(white, 1) > 0
    assert evaluate_state(white, -1) < 0
    assert evaluate_state(black, 1) < 0


def test_evaluation_balances_colours_and_resumes_jsonl(tmp_path):
    output = tmp_path / "heartbeats.jsonl"
    config = EvaluationConfig(games=2, max_insertion_plies=4, max_decisions=4)
    result = evaluate(WhiteWinsState, FirstAction(), FirstAction(), config, heartbeat_path=output)
    assert result["a_wins"] == result["a_losses"] == 1
    assert len(output.read_text().splitlines()) == 2

    # Completed ids are skipped when an interrupted run is resumed.
    extended = EvaluationConfig(games=4, max_insertion_plies=4, max_decisions=4)
    result = evaluate(WhiteWinsState, FirstAction(), FirstAction(), extended, heartbeat_path=output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["game_id"] for row in rows] == [0, 1, 2, 3]
    assert result["a_wins"] == result["a_losses"] == 2


def test_cutoffs_are_reported_separately_from_losses():
    config = EvaluationConfig(games=1, max_insertion_plies=2, max_decisions=10)
    result = evaluate(EndlessState, FirstAction(), FirstAction(), config)
    assert result["a_wins"] == result["a_losses"] == 0
    assert result["cutoffs"] == 1
