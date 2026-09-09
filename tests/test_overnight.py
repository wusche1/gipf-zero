from pathlib import Path

from ops.overnight import Runner


def report(wins, losses, cutoffs=0):
    return {"wins": wins, "losses": losses, "cutoffs": cutoffs, "unfinished": 0,
            "candidate_root_simulations": {"mean": 100},
            "champion_root_simulations": {"mean": 90}}


def bare_runner(tmp_path):
    runner = object.__new__(Runner)
    runner.c = {"seeds": [1, 2], "duels": {"games": 12, "seconds_cpu": 120,
                "seconds_sim": 45, "min_cpu_games_per_model": 12,
                "equal_cpu_time": {"batch": 1, "device": "cpu", "threads": 1,
                                   "simulations": 10000, "budget_ms": 50},
                "equal_simulations": {"batch": 64, "device": "cuda", "threads": 2,
                                      "simulations": 32, "budget_ms": 0}}}
    runner.reports = tmp_path
    runner.summary = {"duels": []}
    runner.emit = lambda *args, **kwargs: None
    runner.check_backend = lambda: None
    return runner


def test_duel_names_include_distinct_runs_seed_and_attempt(tmp_path):
    runner = bare_runner(tmp_path)
    seen = []
    def child(label, command, log, seconds, output):
        seen.append(output.name)
        output.write_text(__import__("json").dumps(report(8, 4)))
        return True
    runner.child = child
    left, right = Path("/tmp/mlp256-seed1"), Path("/tmp/squarecnn32-seed1")
    runner.duel(left, right, "equal_cpu_time", 970001)
    runner.duel(left, right, "equal_cpu_time", 970001, attempt=2)
    assert seen[0] != seen[1]
    assert "mlp256-seed1-vs-squarecnn32-seed1" in seen[0]


def test_rank_counts_both_sides_and_unplayed_games(tmp_path):
    runner = bare_runner(tmp_path)
    assert runner.rank(["a", "b", "c"], [("a", "b", report(10, 2)),
                                               ("a", "c", report(8, 4)),
                                               ("b", "c", report(7, 5))]) == "a"
    rows = {row["name"]: row for row in runner.summary["ranking"]["rows"]}
    assert rows["a"]["wins"] == 18 and rows["b"]["wins"] == 9
    assert rows["a"]["unplayed"] == 24


def test_train_retries_from_atomic_latest_after_failed_child(tmp_path):
    runner = bare_runner(tmp_path)
    runner.base = tmp_path
    runner.c["training_seconds"] = 600
    runner.c["selfplay"] = {"games": 64, "games_per_iteration": 32, "updates": 32,
                              "simulations": 32, "batch_size": 256, "lr": .001,
                              "max_ply": 300, "replay_size": 100000, "device": "cuda"}
    commands = []
    def child(label, command, log, seconds, output):
        commands.append(command)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"atomic")
        return len(commands) == 2
    runner.child = child
    runner.freeze = lambda name, source: source
    runner.summary["training"] = []
    run = runner.train({"name": "mlp", "kind": "mlp", "head": "flat", "width": 8, "blocks": 1}, 1)
    assert run is not None and "--resume" in commands[1]
