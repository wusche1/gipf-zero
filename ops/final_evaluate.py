"""Deadline-bounded final checkpoint selection and held-out evaluation.

This script is intentionally inert until invoked.  It stops only the final
training/league services, snapshots every moving checkpoint before use, and
continues after individual evaluation failures so its JSON summary is durable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from baselines.evaluate import wilson_interval
from ops.promote import promote


ROOT = Path(__file__).resolve().parents[1]
FINALISTS = ("final", "final_mlp")
TRAINING_SERVICES = ("gipf_league", "gipf_final_training", "gipf_final_mlp")
DUEL_SEEDS = {"final": 941001, "final_mlp": 941002}
HELDOUT_SEEDS = {"random": 942001, "greedy": 942002, "mcts": 942003, "minimax": 942004}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_tag() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable target already exists: {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


class FinalRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tag = utc_tag()
        self.archive = ROOT / "checkpoints" / "final" / self.tag
        self.report_dir = ROOT / "reports" / "final" / self.tag
        self.archive.mkdir(parents=True, exist_ok=False)
        self.report_dir.mkdir(parents=True, exist_ok=False)
        self.heartbeat_path = self.report_dir / "heartbeat.jsonl"
        self.summary_path = self.report_dir / "summary.json"
        self.summary: dict[str, Any] = {
            "tag": self.tag,
            "start_epoch": args.start,
            "deadline_epoch": args.deadline,
            "started_epoch": time.time(),
            "archive": str(self.archive.relative_to(ROOT)),
            "reports": [],
            "snapshots": {},
            "steps": [],
            "errors": [],
            "chosen_source": None,
            "completed": False,
        }

    def remaining(self) -> float:
        return self.args.deadline - time.time()

    def emit(self, event: str, **fields: Any) -> None:
        record = {"time": time.time(), "remaining_seconds": max(0.0, self.remaining()), "event": event, **fields}
        with self.heartbeat_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
        self.summary["steps"].append(record)
        temporary = self.summary_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.summary_path)
        print(json.dumps(record, sort_keys=True), flush=True)

    def error(self, label: str, exc: Exception) -> None:
        detail = {"label": label, "type": type(exc).__name__, "message": str(exc)}
        self.summary["errors"].append(detail)
        self.emit("error", **detail)

    def wait_until_start(self) -> bool:
        while self.remaining() > 0 and time.time() < self.args.start:
            self.emit("waiting_for_start", start_epoch=self.args.start)
            time.sleep(min(30.0, self.args.start - time.time(), self.remaining()))
        if self.remaining() <= 0:
            self.emit("deadline_before_start")
            return False
        return True

    def stop_services(self) -> None:
        for service in TRAINING_SERVICES:
            if self.remaining() <= 0:
                self.emit("deadline_before_service_stop", service=service)
                return
            try:
                result = subprocess.run(
                    ["supervisorctl", "stop", service], text=True, capture_output=True,
                    timeout=min(55, max(1, int(self.remaining()))), check=False,
                )
                self.emit("service_stop", service=service, returncode=result.returncode,
                          stdout=result.stdout.strip(), stderr=result.stderr.strip())
            except Exception as exc:
                self.error(f"stop:{service}", exc)

    def freeze(self, label: str, source: Path) -> Path | None:
        try:
            data = source.read_bytes()
            digest = sha256_bytes(data)
            target = self.archive / f"{label}-{digest[:12]}.pt"
            atomic_bytes(target, data)
            self.summary["snapshots"][label] = {"source": str(source.relative_to(ROOT)), "path": str(target.relative_to(ROOT)), "sha256": digest}
            self.emit("snapshot", label=label, path=str(target.relative_to(ROOT)), sha256=digest)
            return target
        except Exception as exc:
            self.error(f"freeze:{label}", exc)
            return None

    def child(self, label: str, command: list[str], output: Path, internal_limit: int, outer_limit: int) -> bool:
        """Run one child process with group termination and durable progress."""
        # Keep enough wall time to terminate a process group and persist the
        # final report before the absolute global deadline.
        if self.remaining() <= 40:
            self.emit("skipped_deadline", label=label)
            return False
        allowed = min(outer_limit, max(1, int(self.remaining() - 35)))
        command = [*command, "--seconds", str(min(internal_limit, allowed))]
        log = self.report_dir / f"{label}.log"
        self.emit("child_start", label=label, command=command, log=str(log.relative_to(ROOT)), outer_seconds=allowed)
        process: subprocess.Popen[str] | None = None
        try:
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen([sys.executable, "-u", *command], cwd=ROOT, stdout=stream,
                                           stderr=subprocess.STDOUT, text=True, start_new_session=True)
                launched = time.monotonic()
                next_heartbeat = launched + 30
                while process.poll() is None:
                    elapsed = time.monotonic() - launched
                    if elapsed >= allowed or self.remaining() <= 0:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=10)
                        raise TimeoutError(f"{label} exceeded its {allowed}s outer limit")
                    if time.monotonic() >= next_heartbeat:
                        self.emit("child_running", label=label, elapsed_seconds=round(elapsed, 1),
                                  output_exists=output.exists())
                        next_heartbeat += 30
                    time.sleep(2)
            if process.returncode != 0:
                raise RuntimeError(f"{label} exited {process.returncode}; see {log}")
            if not output.exists():
                raise RuntimeError(f"{label} exited without {output}")
            self.emit("child_complete", label=label, output=str(output.relative_to(ROOT)))
            self.summary["reports"].append(str(output.relative_to(ROOT)))
            return True
        except Exception as exc:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            self.error(f"child:{label}", exc)
            return False

    def read_report(self, path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.error(f"report:{path.name}", exc)
            return None

    def duel_gate(self, report: dict[str, Any]) -> bool:
        total = report.get("wins", 0) + report.get("losses", 0) + report.get("cutoffs", 0)
        lower = wilson_interval(int(report.get("wins", 0)), int(total))[0] if total else 0.0
        passed = report.get("unfinished", 1) == 0 and total == 80 and lower > 0.5
        self.emit("duel_gate", wins=report.get("wins"), losses=report.get("losses"), cutoffs=report.get("cutoffs"),
                  total=total, wilson_all_games_lower=lower, passed=passed)
        return passed

    def evaluate_finalist(self, name: str, candidate: Path) -> None:
        if self.remaining() <= 1:
            return
        champion = self.freeze(f"champion-before-{name}", ROOT / "checkpoints" / "champion.pt")
        if champion is None:
            return
        duel = self.report_dir / f"{name}-vs-champion-duel.json"
        command = ["-m", "training.duel", "--candidate", str(candidate), "--champion", str(champion),
                   "--games", "80", "--batch", "1", "--device", "cpu", "--threads", "1", "--budget-ms", "50",
                   "--simulations", "10000", "--seed", str(DUEL_SEEDS[name]), "--max-ply", "300", "--output", str(duel)]
        if not self.child(f"{name}-duel", command, duel, 600, 650):
            return
        duel_report = self.read_report(duel)
        if duel_report is None or not self.duel_gate(duel_report):
            return
        greedy = self.report_dir / f"{name}-vs-greedy.json"
        command = ["-m", "training.evaluate", "--checkpoint", str(candidate), "--opponent", "greedy", "--games", "40",
                   "--batch", "64", "--device", "cuda", "--threads", "2", "--simulations", "128", "--seed", str(943000 + DUEL_SEEDS[name]),
                   "--max-ply", "300", "--output", str(greedy)]
        if not self.child(f"{name}-greedy", command, greedy, 600, 650):
            return
        greedy_report = self.read_report(greedy)
        passed = greedy_report is not None and greedy_report.get("unfinished", 1) == 0 and greedy_report.get("wins", 0) >= 32
        self.emit("greedy_gate", finalist=name, wins=None if greedy_report is None else greedy_report.get("wins"), passed=passed)
        if not passed:
            return
        try:
            metadata = promote(candidate, f"GIPF Zero — {greedy_report['model_games']:,} games", [duel, greedy])
            self.emit("promoted", finalist=name, source_hash=metadata["source_checkpoint_sha256"])
        except Exception as exc:
            self.error(f"promote:{name}", exc)

    def source_for_current_champion(self) -> Path | None:
        try:
            champion_meta = json.loads((ROOT / "reports" / "champion.json").read_text(encoding="utf-8"))
            source_hash = champion_meta["source_checkpoint_sha256"]
        except Exception as exc:
            self.error("champion_metadata", exc)
            return None
        candidates = list(self.archive.glob("*.pt")) + list((ROOT / "checkpoints" / "league").glob("*.pt"))
        for path in candidates:
            try:
                if sha256_bytes(path.read_bytes()) == source_hash:
                    self.emit("chosen_source", path=str(path.relative_to(ROOT)), sha256=source_hash)
                    self.summary["chosen_source"] = str(path.relative_to(ROOT))
                    return path
            except OSError:
                continue
        self.error("champion_source", FileNotFoundError(f"no immutable full checkpoint for {source_hash}"))
        return None

    def heldout(self, checkpoint: Path) -> None:
        plans = (
            ("random", 200, ["--opponent", "random", "--batch", "64", "--device", "cuda", "--threads", "2", "--simulations", "128"]),
            ("greedy", 200, ["--opponent", "greedy", "--batch", "64", "--device", "cuda", "--threads", "2", "--simulations", "128"]),
            ("mcts", 80, ["--opponent", "mcts", "--batch", "1", "--device", "cpu", "--threads", "1", "--simulations", "10000", "--neural-budget-ms", "50", "--move-seconds", "0.05", "--baseline-simulations", "10000"]),
            ("minimax", 80, ["--opponent", "minimax", "--batch", "1", "--device", "cpu", "--threads", "1", "--simulations", "10000", "--neural-budget-ms", "50", "--move-seconds", "0.05", "--baseline-depth", "6"]),
        )
        for opponent, games, extra in plans:
            output = self.report_dir / f"heldout-vs-{opponent}.json"
            command = ["-m", "training.evaluate", "--checkpoint", str(checkpoint), "--games", str(games), "--seed", str(HELDOUT_SEEDS[opponent]), "--max-ply", "300", "--output", str(output), *extra]
            self.child(f"heldout-{opponent}", command, output, 600 if opponent in ("mcts", "minimax") else 300, 630 if opponent in ("mcts", "minimax") else 330)

    def run(self) -> int:
        self.emit("runner_created")
        if not self.wait_until_start():
            return 1
        self.stop_services()
        frozen = {name: self.freeze(name, ROOT / "runs" / name / "latest.pt") for name in FINALISTS}
        self.freeze("champion-initial", ROOT / "checkpoints" / "champion.pt")
        for name in FINALISTS:
            if frozen[name] is not None:
                self.evaluate_finalist(name, frozen[name])
        source = self.source_for_current_champion()
        if source is not None:
            self.heldout(source)
        self.summary["completed"] = self.remaining() > 0
        self.summary["finished_epoch"] = time.time()
        self.emit("runner_finished", completed=self.summary["completed"])
        return 0 if self.summary["completed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=float, default=1788988500)
    parser.add_argument("--deadline", type=float, default=1788990900)
    args = parser.parse_args()
    if args.deadline <= args.start:
        parser.error("--deadline must be later than --start")
    return FinalRunner(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
