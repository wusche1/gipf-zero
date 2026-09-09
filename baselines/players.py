"""Deterministic, bounded reference players for the Gipf engine.

The engine deliberately owns all rules.  This module only relies on a state
having ``clone()``, mutating ``apply(action)``, ``legal_actions``,
``current_player`` and ``winner``.  In particular, a search does *not* assume
that applying an action changes the player: Gipf capture choices can leave the
same player in control.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Any, Protocol, Sequence


class GameState(Protocol):
    """The small engine-facing surface used by the reference players."""

    current_player: int
    winner: int
    board: Sequence[int]
    reserves: Sequence[int]

    def clone(self) -> "GameState": ...
    def apply(self, action: int) -> None: ...


# Kept together so experiment records can state exactly what was evaluated.
# ``material`` counts reserve plus board stones (a GIPF/double is two stones),
# while the other terms deliberately give a modest positional preference.
HEURISTIC_WEIGHTS = {
    "terminal": 100_000.0,
    "material": 100.0,
    "reserve": 4.0,
    "double_safety": 3.0,
    "threat": 12.0,
}
BASELINE_VERSION = "v1.2-paired-seeds"


def _read(state: Any, name: str) -> Any:
    value = getattr(state, name)
    return value() if callable(value) else value


def legal_actions(state: GameState) -> tuple[int, ...]:
    """Return a stable action tuple for engines exposing a property or method."""
    return tuple(int(action) for action in _read(state, "legal_actions"))


def terminal_winner(state: GameState) -> int:
    return int(_read(state, "winner"))


def decision_owner(state: GameState) -> int:
    owner = int(_read(state, "current_player"))
    if owner not in (-1, 1):
        raise ValueError(f"current_player must be +/-1, got {owner}")
    return owner


def _side_values(values: Sequence[int], player: int) -> tuple[int, int]:
    """Return player and opponent entries from [white, black] storage."""
    if len(values) != 2:
        raise ValueError("reserves must hold [white, black]")
    return (int(values[0]), int(values[1])) if player == 1 else (int(values[1]), int(values[0]))


def _board_counts(board: Sequence[int], player: int) -> tuple[int, int, int, int]:
    """Count own/opponent singles and doubles from the engine's +/-1/2 board."""
    own_single = own_double = enemy_single = enemy_double = 0
    for raw in board:
        stone = int(raw)
        if stone == player:
            own_single += 1
        elif stone == 2 * player:
            own_double += 1
        elif stone == -player:
            enemy_single += 1
        elif stone == -2 * player:
            enemy_double += 1
    return own_single, own_double, enemy_single, enemy_double


def _threats(state: GameState, player: int) -> float:
    """Read an engine-provided threat count when available.

    Threats are a four-in-a-row threat proxy: an open-ended run of at least
    three same-colour stones.  The fallback uses the public contract's fixed
    axial board ordering, while engine hooks can supply a more exact count.
    """
    for name in ("threat_count", "count_threats"):
        method = getattr(state, name, None)
        if callable(method):
            return float(method(player) - method(-player))
    threats = getattr(state, "threats", None)
    if threats is not None:
        values = threats() if callable(threats) else threats
        if isinstance(values, dict):
            return float(values.get(player, 0) - values.get(-player, 0))
        if len(values) == 2:
            own, enemy = _side_values(values, player)
            return float(own - enemy)
    board = tuple(int(x) for x in _read(state, "board"))
    return float(_line_threats(board, player) - _line_threats(board, -player))


def _contract_lines() -> tuple[tuple[int, ...], ...]:
    """The 21 board lines in the geometry order specified by engine/CONTRACT."""
    coords = [(q, r) for r in range(-3, 4) for q in range(max(-3, -r - 3), min(3, -r + 3) + 1)]
    index = {coord: i for i, coord in enumerate(coords)}
    lines: list[tuple[int, ...]] = []
    # Constant q: increasing r.  Constant r: increasing q.  Constant s:
    # increasing q, exactly matching the public geometry contract.
    for q in range(-3, 4):
        lines.append(tuple(index[(q, r)] for r in range(-3, 4) if (q, r) in index))
    for r in range(-3, 4):
        lines.append(tuple(index[(q, r)] for q in range(-3, 4) if (q, r) in index))
    for s in range(-3, 4):
        lines.append(tuple(index[(q, -s - q)] for q in range(-3, 4) if (q, -s - q) in index))
    return tuple(lines)


CONTRACT_LINES = _contract_lines()


def _line_threats(board: Sequence[int], player: int) -> int:
    if len(board) != 37:
        return 0
    threats = 0
    for line in CONTRACT_LINES:
        colours = [0 if board[i] == 0 else (1 if board[i] * player > 0 else -1) for i in line]
        start = 0
        while start < len(colours):
            if colours[start] != 1:
                start += 1
                continue
            end = start
            while end < len(colours) and colours[end] == 1:
                end += 1
            if end - start >= 3 and ((start > 0 and colours[start - 1] == 0) or (end < len(colours) and colours[end] == 0)):
                threats += 1
            start = end
    return threats


def evaluate_state(state: GameState, perspective: int) -> float:
    """Fixed, symmetric evaluation positive for ``perspective``.

    Components are terminal outcome, total material, unused reserve, number of
    doubles (a small safety proxy), and engine-provided threats.  It is a
    reference heuristic, not a claim that the last three terms are complete
    Gipf strategy.
    """
    if perspective not in (-1, 1):
        raise ValueError("perspective must be +/-1")
    winner = terminal_winner(state)
    if winner:
        return HEURISTIC_WEIGHTS["terminal"] if winner == perspective else -HEURISTIC_WEIGHTS["terminal"]

    own_single, own_double, enemy_single, enemy_double = _board_counts(_read(state, "board"), perspective)
    own_reserve, enemy_reserve = _side_values(_read(state, "reserves"), perspective)
    own_material = own_reserve + own_single + 2 * own_double
    enemy_material = enemy_reserve + enemy_single + 2 * enemy_double
    return (
        HEURISTIC_WEIGHTS["material"] * (own_material - enemy_material)
        + HEURISTIC_WEIGHTS["reserve"] * (own_reserve - enemy_reserve)
        + HEURISTIC_WEIGHTS["double_safety"] * (own_double - enemy_double)
        + HEURISTIC_WEIGHTS["threat"] * _threats(state, perspective)
    )


def _deadline(time_limit: float | None) -> float | None:
    if time_limit is None:
        return None
    return time.monotonic() + max(0.0, float(time_limit))


def _pair_rng(seed: int | None, pair_id: int, stream: int) -> random.Random:
    """A reproducible independent stream for one colour-balanced game pair."""
    # The constants separate pairs and A/B without Python's randomized hash.
    return random.Random((0 if seed is None else int(seed)) + 1_000_003 * pair_id + 97_409 * stream)


def _expired(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


class _SearchDeadline(Exception):
    """Internal sentinel: incomplete root iterations must not choose an action."""


def _check_deadline(deadline: float | None) -> None:
    if _expired(deadline):
        raise _SearchDeadline


def _is_capture_phase(state: GameState) -> bool:
    phase = getattr(state, "phase", None)
    phase = phase() if callable(phase) else phase
    return phase == "capture"


def _quiesce(
    state: GameState,
    root_player: int,
    remaining: int,
    alpha: float,
    beta: float,
    deadline: float | None,
) -> float:
    """Resolve forced capture choices before a static evaluation.

    A push can immediately enter one or more capture decisions, sometimes for
    different players.  This deliberately stops at six capture decisions so a
    nominally shallow baseline cannot grow into an unbounded tactical search.
    """
    _check_deadline(deadline)
    if terminal_winner(state) or remaining <= 0 or not _is_capture_phase(state):
        return evaluate_state(state, root_player)
    actions = legal_actions(state)
    if not actions:
        return evaluate_state(state, root_player)
    maximizing = decision_owner(state) == root_player
    if maximizing:
        value = -math.inf
        for action in actions:
            _check_deadline(deadline)
            child = state.clone()
            child.apply(action)
            value = max(value, _quiesce(child, root_player, remaining - 1, alpha, beta, deadline))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    value = math.inf
    for action in actions:
        _check_deadline(deadline)
        child = state.clone()
        child.apply(action)
        value = min(value, _quiesce(child, root_player, remaining - 1, alpha, beta, deadline))
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value


@dataclass
class RandomPlayer:
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def set_game_seed(self, pair_id: int, stream: int) -> None:
        self._rng = _pair_rng(self.seed, pair_id, stream)

    def choose_action(self, state: GameState, time_limit: float | None = None) -> int:
        actions = legal_actions(state)
        if not actions:
            raise ValueError("cannot choose an action from a terminal/no-action state")
        return self._rng.choice(actions)


@dataclass
class GreedyPlayer:
    """Pick the action with the best one-decision fixed heuristic."""

    seed: int | None = None
    capture_depth: int = 6

    def __post_init__(self) -> None:
        if not 0 <= self.capture_depth <= 6:
            raise ValueError("capture_depth must be between zero and six")
        self._rng = random.Random(self.seed)

    def set_game_seed(self, pair_id: int, stream: int) -> None:
        self._rng = _pair_rng(self.seed, pair_id, stream)

    def choose_action(self, state: GameState, time_limit: float | None = None) -> int:
        player = decision_owner(state)
        deadline = _deadline(time_limit)
        actions = list(legal_actions(state))
        if not actions:
            raise ValueError("cannot choose an action from a terminal/no-action state")
        best_action, best_value = actions[0], -math.inf
        for action in actions:
            if _expired(deadline):
                break
            try:
                child = state.clone()
                child.apply(action)
                value = _quiesce(child, player, self.capture_depth, -math.inf, math.inf, deadline)
            except _SearchDeadline:
                break
            if value > best_value or (value == best_value and self._rng.randrange(2)):
                best_action, best_value = action, value
        return best_action


@dataclass
class MinimaxPlayer:
    """Small alpha-beta search that maximizes/minimizes by actual decision owner."""

    depth: int = 2
    seed: int | None = None
    capture_depth: int = 6

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be at least one")
        if not 0 <= self.capture_depth <= 6:
            raise ValueError("capture_depth must be between zero and six")
        self._rng = random.Random(self.seed)

    def set_game_seed(self, pair_id: int, stream: int) -> None:
        self._rng = _pair_rng(self.seed, pair_id, stream)

    def choose_action(self, state: GameState, time_limit: float | None = None) -> int:
        root_player = decision_owner(state)
        actions = legal_actions(state)
        if not actions:
            raise ValueError("cannot choose an action from a terminal/no-action state")
        deadline = _deadline(time_limit)
        # Only publish a root choice from a whole depth iteration.  A timeout
        # in the middle cannot make early legal-action ordering look better.
        best_action = actions[0]
        for iteration_depth in range(1, self.depth + 1):
            try:
                candidate = self._root_iteration(state, root_player, actions, iteration_depth, deadline)
            except _SearchDeadline:
                break
            best_action = candidate
        return best_action

    def _root_iteration(
        self, state: GameState, root_player: int, actions: tuple[int, ...], depth: int, deadline: float | None
    ) -> int:
        best_action, best_value = actions[0], -math.inf
        alpha, beta = -math.inf, math.inf
        for action in actions:
            _check_deadline(deadline)
            child = state.clone()
            child.apply(action)
            value = self._search(child, root_player, depth - 1, alpha, beta, deadline)
            if value > best_value or (value == best_value and self._rng.randrange(2)):
                best_action, best_value = action, value
            alpha = max(alpha, best_value)
        return best_action

    def _search(
        self, state: GameState, root_player: int, depth: int, alpha: float, beta: float, deadline: float | None
    ) -> float:
        _check_deadline(deadline)
        if depth <= 0 or terminal_winner(state):
            return _quiesce(state, root_player, self.capture_depth, alpha, beta, deadline)
        actions = legal_actions(state)
        if not actions:
            return evaluate_state(state, root_player)
        maximizing = decision_owner(state) == root_player
        if maximizing:
            value = -math.inf
            for action in actions:
                _check_deadline(deadline)
                child = state.clone()
                child.apply(action)
                value = max(value, self._search(child, root_player, depth - 1, alpha, beta, deadline))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return _quiesce(state, root_player, self.capture_depth, alpha, beta, deadline) if value == -math.inf else value
        value = math.inf
        for action in actions:
            _check_deadline(deadline)
            child = state.clone()
            child.apply(action)
            value = min(value, self._search(child, root_player, depth - 1, alpha, beta, deadline))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return _quiesce(state, root_player, self.capture_depth, alpha, beta, deadline) if value == math.inf else value


@dataclass
class _MctsNode:
    state: GameState
    parent: "_MctsNode | None" = None
    action: int | None = None
    untried: list[int] | None = None
    children: list["_MctsNode"] | None = None
    visits: int = 0
    value_sum: float = 0.0

    def __post_init__(self) -> None:
        if self.untried is None:
            self.untried = list(legal_actions(self.state)) if not terminal_winner(self.state) else []
        if self.children is None:
            self.children = []


@dataclass
class MCTSPlayer:
    """UCT search with bounded simulation count and an optional tactical rollout."""

    simulations: int = 128
    exploration: float = 1.35
    rollout_depth: int = 2
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.simulations < 1:
            raise ValueError("simulations must be positive")
        self._rng = random.Random(self.seed)

    def set_game_seed(self, pair_id: int, stream: int) -> None:
        self._rng = _pair_rng(self.seed, pair_id, stream)

    def choose_action(self, state: GameState, time_limit: float | None = None) -> int:
        root_player = decision_owner(state)
        root = _MctsNode(state.clone())
        if not root.untried:
            raise ValueError("cannot choose an action from a terminal/no-action state")
        fallback = root.untried[0]
        deadline = _deadline(time_limit)
        for _ in range(self.simulations):
            if _expired(deadline):
                break
            node = root
            while not node.untried and node.children and not terminal_winner(node.state):
                node = self._select_child(node, root_player)
            if node.untried and not _expired(deadline):
                action = node.untried.pop(self._rng.randrange(len(node.untried)))
                child_state = node.state.clone()
                child_state.apply(action)
                child = _MctsNode(child_state, parent=node, action=action)
                node.children.append(child)
                node = child
            value = self._rollout_value(node.state, root_player, deadline)
            while node is not None:
                node.visits += 1
                node.value_sum += value
                node = node.parent
        if not root.children:
            return fallback
        # Visits are robust to occasional uncompleted/short rollouts.  Root is
        # always a maximizing decision for root_player.
        return max(root.children, key=lambda child: (child.visits, child.value_sum / child.visits)).action  # type: ignore[return-value]

    def _select_child(self, node: _MctsNode, root_player: int) -> _MctsNode:
        assert node.children
        log_parent = math.log(max(1, node.visits))
        actor_sign = 1.0 if decision_owner(node.state) == root_player else -1.0

        def uct(child: _MctsNode) -> float:
            if child.visits == 0:
                return math.inf
            mean = child.value_sum / child.visits
            return actor_sign * mean + self.exploration * math.sqrt(log_parent / child.visits)

        return max(node.children, key=uct)

    def _rollout_value(self, state: GameState, root_player: int, deadline: float | None) -> float:
        rollout = state.clone()
        for _ in range(self.rollout_depth):
            if terminal_winner(rollout) or _expired(deadline):
                break
            actions = legal_actions(rollout)
            if not actions:
                break
            # A guaranteed terminal action is preferable to a random rollout.
            action = self._rng.choice(actions)
            for candidate in actions:
                if _expired(deadline):
                    break
                probe = rollout.clone()
                probe.apply(candidate)
                if terminal_winner(probe) == decision_owner(rollout):
                    action = candidate
                    break
            rollout.apply(action)
        # Keep UCT values numerically tame; terminal values are still exact in sign.
        return math.tanh(evaluate_state(rollout, root_player) / HEURISTIC_WEIGHTS["material"])


Player = RandomPlayer | GreedyPlayer | MinimaxPlayer | MCTSPlayer


def make_player(name: str, *, seed: int | None = None, **kwargs: Any) -> Player:
    """Build one of the stable CLI/player-factory names."""
    normalized = name.lower().replace("-", "_")
    classes = {
        "random": RandomPlayer,
        "greedy": GreedyPlayer,
        "minimax": MinimaxPlayer,
        "mcts": MCTSPlayer,
    }
    try:
        return classes[normalized](seed=seed, **kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown baseline player {name!r}; choose one of {sorted(classes)}") from exc
