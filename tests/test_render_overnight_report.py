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
