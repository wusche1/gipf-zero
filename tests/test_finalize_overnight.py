import json
import hashlib
from pathlib import Path

from ops import finalize_overnight as finalizer


def test_collects_only_reports_named_by_continuation_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(finalizer, "ROOT", tmp_path)
    run = tmp_path / "runs" / "overnight" / "tag" / "winner-seed1"; run.mkdir(parents=True)
    (run / "league-010203-duel.log").write_text("complete")
    (run / "league-010203-greedy.log").write_text("complete")
    (tmp_path / "reports").mkdir(); (tmp_path / "reports" / "league-010203-duel.json").write_text(json.dumps({"wins": 1}))
    (tmp_path / "reports" / "league-010203-greedy.json").write_text(json.dumps({"wins": 2}))
    (tmp_path / "reports" / "league-999999-duel.json").write_text(json.dumps({"wins": 3}))
    report_dir = tmp_path / "reports" / "overnight" / "tag"; report_dir.mkdir(parents=True)
    copied = finalizer.collect_continuation_reports({"continuation": {"run": "runs/overnight/tag/winner-seed1"}}, report_dir)
    assert [path.name for path in copied] == ["league-010203-duel.json", "league-010203-greedy.json"]
    assert not (report_dir / "league-999999-duel.json").exists()


def test_served_metadata_records_the_actual_artifact_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(finalizer, "ROOT", tmp_path)
    (tmp_path / "checkpoints").mkdir(); (tmp_path / "reports").mkdir()
    artifact = tmp_path / "checkpoints" / "champion.pt"; artifact.write_bytes(b"served artifact")
    (tmp_path / "reports" / "champion.json").write_text(json.dumps({"source_checkpoint_sha256": "source"}))
    metadata = finalizer.served_metadata()
    assert metadata["served_artifact_sha256"] == hashlib.sha256(b"served artifact").hexdigest()
    assert "Served `champion.pt` artifact SHA-256" in finalizer.model_card(metadata)
