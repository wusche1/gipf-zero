"""Reproducible, resumable head-to-head evaluation for Gipf baseline players.

Example:
    python -m baselines.evaluate --white minimax --black mcts --games 40 \
      --move-seconds 0.05 --output artifacts/baselines.json
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import argparse
import importlib
import inspect
import json
from pathlib import Path
import time
import traceback
from typing import Any, Callable, Mapping, Protocol

from .players import BASELINE_VERSION, GameState, make_player, terminal_winner


class PlayerLike(Protocol):
    def choose_action(self, state: GameState, time_limit: float | None = None) -> int: ...

    def set_game_seed(self, pair_id: int, stream: int) -> None: ...


StateFactory = Callable[..., GameState]


@dataclass(frozen=True)
class EvaluationConfig:
    """Frozen settings recorded with every run.

    ``max_insertion_plies`` counts engine ``State.ply`` and consequently does
    not count capture decisions.  ``max_decisions`` is a separate hard bound
    on all calls to a player, including captures.
    """

    games: int = 20
    seed: int = 0
    max_insertion_plies: int = 400
    max_decisions: int = 2_000
    move_seconds: float | None = 0.05
    confidence: float = 0.95

    def __post_init__(self) -> None:
        if self.games < 1 or self.max_insertion_plies < 1 or self.max_decisions < 1:
            raise ValueError("games and both cutoffs must be positive")
        if self.move_seconds is not None and self.move_seconds < 0:
            raise ValueError("move_seconds cannot be negative")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must lie between zero and one")


def _state_ply(state: GameState) -> int:
    value = getattr(state, "ply")
    return int(value() if callable(value) else value)


def _new_state(factory: StateFactory, seed: int) -> GameState:
    """Permit seeded test factories while the native State() stays deterministic."""
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory()
    parameters = signature.parameters
    if "seed" in parameters:
        return factory(seed=seed)
    if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
        return factory(seed=seed)
    if any(p.kind is p.VAR_POSITIONAL for p in parameters.values()) or len(parameters) == 1:
        return factory(seed)
    return factory()


def _call_player(player: PlayerLike, state: GameState, limit: float | None) -> int:
    """Call modern players with a budget but permit a tiny legacy test player."""
    try:
        return int(player.choose_action(state, time_limit=limit))
    except TypeError as exc:
        # Do not mask TypeErrors raised within choose_action for normal players.
        try:
            signature = inspect.signature(player.choose_action)
        except (TypeError, ValueError):
            raise exc
        if "time_limit" not in signature.parameters:
            return int(player.choose_action(state))
        raise exc


def _prepare_players(player_a: PlayerLike, player_b: PlayerLike, pair_id: int) -> None:
    """Reset supported players by pair, making JSONL resume non-repeating."""
    for stream, player in enumerate((player_a, player_b)):
        reset = getattr(player, "set_game_seed", None)
        if callable(reset):
            reset(pair_id, stream)


def play_game(
    state_factory: StateFactory,
    white: PlayerLike,
    black: PlayerLike,
    config: EvaluationConfig,
    *,
    game_id: int,
    state_seed: int,
) -> dict[str, Any]:
    """Play one game, returning a JSON-ready record instead of raising.

    Errors are isolated to their game so a long evaluation keeps its useful
    results.  A cutoff is explicitly distinct from an engine winner.
    """
    started = time.monotonic()
    record: dict[str, Any] = {
        "kind": "game",
        "game_id": game_id,
        "state_seed": state_seed,
        "white": "A" if white is not black else "shared",
        "black": "B" if white is not black else "shared",
        "winner": 0,
        "reason": None,
        "decisions": 0,
        "insertion_plies": 0,
        "seconds": 0.0,
    }
    try:
        state = _new_state(state_factory, state_seed)
        while not terminal_winner(state):
            if _state_ply(state) >= config.max_insertion_plies:
                record["reason"] = "insertion_cutoff"
                break
            if record["decisions"] >= config.max_decisions:
                record["reason"] = "decision_cutoff"
                break
            owner = int(getattr(state, "current_player"))
            if owner not in (-1, 1):
                raise ValueError(f"engine returned invalid current_player {owner}")
            player = white if owner == 1 else black
            action = _call_player(player, state, config.move_seconds)
            state.apply(action)
            record["decisions"] += 1
        record["winner"] = terminal_winner(state)
        if record["winner"]:
            record["reason"] = "winner"
        elif record["reason"] is None:
            # A nonterminal state with no action is an engine/integration fault.
            record["reason"] = "no_winner"
    except Exception as exc:  # records are more valuable than aborting a tournament
        record["reason"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=8)
    record["seconds"] = round(time.monotonic() - started, 6)
    if "state" in locals():
        record["insertion_plies"] = _state_ply(state)
    return record


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> list[float]:
    """Wilson confidence interval, without a dependency on scipy."""
    if total <= 0:
        return [0.0, 1.0]
    # z=1.95996 for the standard 95% case.  NormalDist provides other values.
    from statistics import NormalDist

    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / total
    divisor = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / divisor
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / divisor
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def summarize(records: list[Mapping[str, Any]], config: EvaluationConfig) -> dict[str, Any]:
    """Summarize A-vs-B records; errors and cutoffs stay separate from losses."""
    games = [record for record in records if record.get("kind") == "game"]
    a_wins = sum(record.get("a_outcome") == "win" for record in games)
    b_wins = sum(record.get("a_outcome") == "loss" for record in games)
    cutoffs = sum(str(record.get("reason", "")).endswith("cutoff") for record in games)
    errors = sum(record.get("reason") == "error" for record in games)
    decisive = a_wins + b_wins
    return {
        "kind": "summary",
        "config": asdict(config),
        "games_recorded": len(games),
        "a_wins": a_wins,
        "a_losses": b_wins,
        "draws": sum(record.get("a_outcome") == "draw" for record in games),
        "cutoffs": cutoffs,
        "errors": errors,
        "decisive_games": decisive,
        "a_win_rate_decisive": (a_wins / decisive) if decisive else None,
        "a_win_rate_decisive_wilson": wilson_interval(a_wins, decisive, config.confidence),
        "confidence": config.confidence,
    }


def _load_heartbeats(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL heartbeat at {path}:{line_number}") from exc
            if record.get("kind") == "game":
                records.append(record)
    return records


def evaluate(
    state_factory: StateFactory,
    player_a: PlayerLike,
    player_b: PlayerLike,
    config: EvaluationConfig,
    *,
    heartbeat_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate paired colours and append one flushed JSONL line per game.

    Game ids alternate colours: even ids have A as White, odd ids have A as
    Black.  Pair members share a state seed for engines/factories with setup
    randomness.  Existing JSONL games are retained, so interrupted runs resume
    without replaying completed game ids.
    """
    path = Path(heartbeat_path) if heartbeat_path is not None else None
    records = _load_heartbeats(path) if path else []
    completed = {int(record["game_id"]) for record in records}
    if any(game_id < 0 or game_id >= config.games for game_id in completed):
        raise ValueError("heartbeat contains game ids outside this evaluation")
    writer = None
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = path.open("a", encoding="utf-8")
    try:
        for game_id in range(config.games):
            if game_id in completed:
                continue
            # Pair games 0/1, 2/3, ... retain their opening seed and swap A/B.
            state_seed = config.seed + game_id // 2
            _prepare_players(player_a, player_b, game_id // 2)
            white, black = (player_a, player_b) if game_id % 2 == 0 else (player_b, player_a)
            record = play_game(state_factory, white, black, config, game_id=game_id, state_seed=state_seed)
            record["a_colour"] = "white" if game_id % 2 == 0 else "black"
            if record["winner"]:
                a_won = (game_id % 2 == 0 and record["winner"] == 1) or (game_id % 2 == 1 and record["winner"] == -1)
                record["a_outcome"] = "win" if a_won else "loss"
            elif record["reason"] == "no_winner":
                record["a_outcome"] = "draw"
            else:
                record["a_outcome"] = None
            records.append(record)
            if writer:
                writer.write(json.dumps(record, sort_keys=True) + "\n")
                writer.flush()
    finally:
        if writer:
            writer.close()
    result = summarize(records, config)
    result["heartbeat_path"] = str(path) if path else None
    return result


def _load_factory(spec: str | None) -> StateFactory:
    if spec is None:
        return importlib.import_module("gipf_engine").State
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--state-factory must have the form module:callable")
    return getattr(importlib.import_module(module_name), attribute)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--white", default="minimax", choices=("random", "greedy", "minimax", "mcts"), help="baseline A")
    parser.add_argument("--black", default="mcts", choices=("random", "greedy", "minimax", "mcts"), help="baseline B")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-insertion-plies", type=int, default=400)
    parser.add_argument("--max-decisions", type=int, default=2000)
    parser.add_argument("--move-seconds", type=float, default=0.05)
    parser.add_argument("--state-factory", default=None, metavar="MODULE:CALLABLE")
    parser.add_argument("--output", type=Path, required=True, help="JSON summary path; heartbeat uses the same stem with .jsonl")
    parser.add_argument("--heartbeat", type=Path, default=None, help="optional JSONL heartbeat path (defaults beside --output)")
    parser.add_argument("--minimax-depth", type=int, default=2)
    parser.add_argument("--mcts-simulations", type=int, default=128)
    args = parser.parse_args(argv)
    config = EvaluationConfig(
        games=args.games,
        seed=args.seed,
        max_insertion_plies=args.max_insertion_plies,
        max_decisions=args.max_decisions,
        move_seconds=args.move_seconds,
    )
    a_kwargs = {"depth": args.minimax_depth} if args.white == "minimax" else {"simulations": args.mcts_simulations} if args.white == "mcts" else {}
    b_kwargs = {"depth": args.minimax_depth} if args.black == "minimax" else {"simulations": args.mcts_simulations} if args.black == "mcts" else {}
    result = evaluate(
        _load_factory(args.state_factory),
        make_player(args.white, seed=args.seed, **a_kwargs),
        make_player(args.black, seed=args.seed + 1, **b_kwargs),
        config,
        heartbeat_path=args.heartbeat or args.output.with_suffix(".jsonl"),
    )
    result["players"] = {
        "A": {"name": args.white, **a_kwargs},
        "B": {"name": args.black, **b_kwargs},
    }
    result["baseline_version"] = BASELINE_VERSION
    result["tie_breaking"] = "seeded_random_among_equal_root_values_per_colour_balanced_pair"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
