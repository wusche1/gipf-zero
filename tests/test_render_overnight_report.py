import json
from pathlib import Path

from ops import render_overnight_report as renderer


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def duel(wins, losses, unfinished=0):
    return {"wins": wins, "losses": losses, "cutoffs": 0, "unfinished": unfinished}


def test_primary_pool_uses_last_complete_retry_and_separates_tiebreak(tmp_path):
    report_dir = tmp_path / "reports" / "overnight" / "tag"
    write_json(report_dir / "summary.json", {"tag": "tag", "duels": [
        {"left": "a-seed1", "right": "b-seed1", "mode": "equal_cpu_time", "seed": 1, "attempt": 1, "complete": False, "result": None, "report": "a-attempt-1.json"},
        {"left": "a-seed1", "right": "b-seed1", "mode": "equal_cpu_time", "seed": 1, "attempt": 2, "complete": True, "report": "a-attempt-2.json"},
        {"left": "a-seed1", "right": "b-seed1", "mode": "equal_cpu_time", "seed": 1, "attempt": 3, "complete": True, "report": "a-attempt-3.json"},
    ]})
    write_json(report_dir / "a-attempt-1.json", duel(7, 0, unfinished=5))
    write_json(report_dir / "a-attempt-2.json", duel(8, 4))
    write_json(report_dir / "a-attempt-3.json", duel(50, 30))
    records = renderer.duel_records(report_dir, renderer.read_json(report_dir / "summary.json"))
    primary = renderer.primary_records(records, "equal_cpu_time")
    assert len(primary) == 1
    assert primary[0]["attempt"] == 2
    assert primary[0]["wins"] == 8
    assert len([record for record in records if record["tie_break"]]) == 1


def test_orphan_raw_duel_is_linked_but_cannot_enter_primary_pool(tmp_path):
    report_dir = tmp_path / "reports" / "overnight" / "tag"
    write_json(report_dir / "summary.json", {"tag": "tag", "duels": []})
    write_json(report_dir / "orphan-a-vs-b-equal_cpu_time-seed1.json", duel(12, 0))
    records = renderer.duel_records(report_dir, renderer.read_json(report_dir / "summary.json"))
    assert len(records) == 1 and records[0]["eligible"] is False
    assert renderer.primary_records(records, "equal_cpu_time") == []


def test_secondary_retry_replaces_only_matching_incomplete_fixed_record(tmp_path, monkeypatch):
    monkeypatch.setattr(renderer, "ROOT", tmp_path)
    report_dir = tmp_path / "reports" / "overnight" / "tag"
    write_json(report_dir / "summary.json", {"tag": "tag", "duels": [
        {"left": "a-seed1", "right": "b-seed1", "mode": "equal_cpu_time", "seed": 1, "attempt": 1, "complete": True, "report": "cpu.json"},
        {"left": "a-seed1", "right": "b-seed1", "mode": "equal_simulations", "seed": 1, "attempt": 1, "complete": False, "report": "fixed.json"},
    ]})
    write_json(report_dir / "cpu.json", {**duel(8, 4), "hashes": ["cpu-a", "cpu-b"]})
    write_json(report_dir / "fixed.json", {**duel(5, 2, unfinished=5), "hashes": ["a", "b"]})
    write_json(report_dir / "fixed-retry-complete.json", {**duel(9, 1), "hashes": ["a", "b"]})
    write_json(report_dir / "secondary-retries.json", {"all_complete": True, "entries": [{
        "original_report": "fixed.json", "retry_report": "fixed-retry-complete.json",
        "left": "a-seed1", "right": "b-seed1", "seed": 1, "frozen_hashes_match_original": True,
    }]})
    records = renderer.duel_records(report_dir, renderer.read_json(report_dir / "summary.json"))
    overlaid, applied = renderer.secondary_retry_overlay(records, report_dir)
    fixed = renderer.primary_records(overlaid, "equal_simulations")
    assert len(applied) == len(fixed) == 1
    assert (fixed[0]["wins"], fixed[0]["losses"], fixed[0]["unfinished"]) == (9, 1, 0)
    assert fixed[0]["original_path"].name == "fixed.json"
    # The CPU primary pool is never altered and the retry is not double-counted.
    assert renderer.primary_records(records, "equal_cpu_time") == renderer.primary_records(overlaid, "equal_cpu_time")


def test_renderer_lists_only_this_continuation_league_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(renderer, "ROOT", tmp_path)
    tag = "20260101T000000Z"; report_dir = tmp_path / "reports" / "overnight" / tag
    run = tmp_path / "runs" / "overnight" / tag / "winner-seed1"
    run.mkdir(parents=True); (run / "league-010203-duel.log").write_text("done")
    historical = tmp_path / "runs" / "final"; historical.mkdir(parents=True); (historical / "league-999999-duel.log").write_text("old")
    write_json(tmp_path / "reports" / "league-010203-duel.json", {**duel(7, 2), "models": [{"games": 1234}]})
    write_json(report_dir / "config.json", {"architectures": [], "duels": {}})
    write_json(report_dir / "summary.json", {"tag": tag, "continuation": {"run": f"runs/overnight/{tag}/winner-seed1"}})
    rows = renderer.continuation_league_reports(renderer.read_json(report_dir / "summary.json"), report_dir)
    assert len(rows) == 1 and rows[0]["path"].name == "league-010203-duel.json" and rows[0]["candidate_games"] == 1234
    text = renderer.render(report_dir / "summary.json").read_text()
    assert "Continuation league evaluations" in text and "1234" in text and "do not by themselves imply promotion" in text


def test_render_marks_incomplete_protocol_and_links_raw_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(renderer, "ROOT", tmp_path)
    tag = "20260101T000000Z"
    report_dir = tmp_path / "reports" / "overnight" / tag
    run = tmp_path / "runs" / "overnight" / tag / "a-seed1"
    write_json(report_dir / "config.json", {
        "training_seconds": 600, "seeds": [1],
        "architectures": [{"name": "a", "head": "flat"}, {"name": "b", "head": "query"}],
        "selfplay": {"native_forest": True}, "duels": {"games": 12, "seconds_cpu": 600, "seconds_sim": 45},
    })
    write_json(report_dir / "summary.json", {"tag": tag, "status": "running", "training": [{"name": "a-seed1", "run": f"runs/overnight/{tag}/a-seed1", "completed": False}], "duels": [
        {"left": "a-seed1", "right": "b-seed1", "mode": "equal_cpu_time", "seed": 1, "attempt": 1, "complete": True, "report": "a-vs-b.json"},
    ]})
    write_json(report_dir / "a-vs-b.json", duel(7, 5))
    write_json(run / "config.json", {"seed": 1, "effective_model": {"kind": "mlp"}})
    (run / "metrics.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (run / "metrics.jsonl").write_text(json.dumps({"event": "start", "parameters": 123}) + "\n" + json.dumps({"event": "stopped", "elapsed": 12.5, "games": 4, "updates": 2}) + "\n")
    output = renderer.render(report_dir / "summary.json")
    text = output.read_text()
    assert "incomplete or still running" in text
    assert "not a pure architecture ablation" in text
    assert "native forest" in text
    assert "a-vs-b.json" in text and "(a-vs-b.json)" in text
    assert "Unplayed" in text


def test_training_table_stops_at_initial_checkpoint_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(renderer, "ROOT", tmp_path)
    tag = "20260101T000000Z"
    report_dir = tmp_path / "reports" / "overnight" / tag
    run_root = tmp_path / "runs" / "overnight" / tag
    run = run_root / "a-seed1"
    write_json(report_dir / "config.json", {})
    write_json(report_dir / "summary.json", {"tag": tag, "status": "complete", "training": [{"name": "a-seed1", "run": f"runs/overnight/{tag}/a-seed1", "completed": True}]})
    write_json(run / "config.json", {"seed": 1, "effective_model": {"kind": "mlp"}})
    (run_root / "heartbeat.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (run_root / "heartbeat.jsonl").write_text(json.dumps({"event": "checkpoint_frozen", "name": "a-seed1", "time": 20}) + "\n")
    (run / "metrics.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in [
        {"event": "start", "time": 1, "parameters": 123},
        {"event": "train", "time": 10, "elapsed": 10, "games": 10, "updates": 2},
        {"event": "train", "time": 30, "elapsed": 200, "games": 110, "updates": 22},
    ]) + "\n")
    rows = renderer.training_rows(renderer.read_json(report_dir / "summary.json"), report_dir, run_root)
    assert rows[0]["games"] == 10
    assert rows[0]["updates"] == 2
    assert rows[0]["seconds"] == 10


def _recovery_fixture(tmp_path, *, recovery=True, continuation_complete=True, promotion=True, cpu_complete=True):
    monkeypatch_root = tmp_path
    tag = "20260101T000000Z"
    report_dir = monkeypatch_root / "reports" / "overnight" / tag
    run_root = monkeypatch_root / "runs" / "overnight" / tag
    training = []
    for model in ("a", "b"):
        name = f"{model}-seed1"
        run = run_root / name
        write_json(run / "config.json", {"seed": 1, "effective_model": {"kind": "resnet"}})
        (run / "metrics.jsonl").parent.mkdir(parents=True, exist_ok=True)
        (run / "metrics.jsonl").write_text(json.dumps({"event": "train", "elapsed": 600, "games": 12, "updates": 4}) + "\n")
        training.append({"name": name, "run": f"runs/overnight/{tag}/{name}", "completed": True})
    write_json(report_dir / "config.json", {
        "seeds": [1], "architectures": [{"name": "a", "head": "flat"}, {"name": "b", "head": "flat"}],
        "duels": {"games": 12, "seconds_cpu": 600, "seconds_sim": 45},
    })
    write_json(report_dir / "a-vs-b.json", duel(8, 4, unfinished=0 if cpu_complete else 2))
    summary = {
        "tag": tag, "status": "partial", "training": training,
        "continuation": {"completed": continuation_complete},
        "duels": [{"left": "a-seed1", "right": "b-seed1", "mode": "equal_cpu_time", "seed": 1,
                    "attempt": 1, "complete": cpu_complete, "report": "a-vs-b.json"}],
        "errors": [{"label": "huggingface-upload", "message": "upload failed"}],
    }
    if promotion:
        summary["promotion"] = {"passed": False, "duel_report": "a-vs-b.json"}
    write_json(report_dir / "summary.json", summary)
    if recovery:
        write_json(report_dir / "publication-recovery.json", {"event": "complete", "url": "https://example.invalid/recovered.pt"})
    return report_dir


def test_publication_recovery_clears_only_publication_incomplete_banner(tmp_path, monkeypatch):
    monkeypatch.setattr(renderer, "ROOT", tmp_path)
    report_dir = _recovery_fixture(tmp_path)
    summary_path = report_dir / "summary.json"
    text = renderer.render(summary_path).read_text()
    assert "training/evaluation complete; publication recovered" in text
    assert "incomplete or still running" not in text
    assert "**Status:** `partial` (training/evaluation complete; publication recovered)" in text
    assert "upload failed" in text
    assert "publication-recovery.json" in text


def test_publication_recovery_never_claims_completion_without_all_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(renderer, "ROOT", tmp_path)
    for index, kwargs in enumerate((
        {"recovery": False}, {"continuation_complete": False}, {"promotion": False}, {"cpu_complete": False},
    )):
        report_dir = _recovery_fixture(tmp_path / str(index), **kwargs)
        text = renderer.render(report_dir / "summary.json").read_text()
        assert "training/evaluation complete; publication recovered" not in text
        assert "incomplete or still running" in text
