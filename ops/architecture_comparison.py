"""Inert, supervised fresh-start architecture comparison runner.

It never promotes a checkpoint or touches the served champion.  Invoke only
after the native-forest benchmark named in the immutable JSON configuration is
present.  The runner starts one training process at a time and freezes every
completed candidate before running paired architecture duels.

The result is a dedicated-wall-time comparison: all candidates receive the
same training allocation and each CPU match receives the same per-decision
budget.  It intentionally does not claim equal parameter counts or FLOPs.
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

import gipf_engine


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {"count": len(values), "mean": sum(values) / len(values), "min": min(values), "max": max(values)}


class Runner:
    def __init__(self, config: dict[str, Any], deadline: float) -> None:
        self.config = config
        self.deadline = deadline
        self.tag = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = ROOT / "runs" / "architecture-comparison" / self.tag
        self.checkpoint_dir = ROOT / "checkpoints" / "architecture-comparison" / self.tag
        self.report_dir = ROOT / "reports" / "architecture-comparison" / self.tag
        for folder in (self.run_dir, self.checkpoint_dir, self.report_dir):
            folder.mkdir(parents=True, exist_ok=False)
        self.heartbeat = self.run_dir / "heartbeat.jsonl"
        self.summary_path = self.report_dir / "summary.json"
        self.backend = self.backend_revision()
        self.summary: dict[str, Any] = {
            "version": config["version"], "tag": self.tag, "deadline": deadline,
            "backend_revision": self.backend, "training": [], "duels": [], "errors": [],
            "status": "running", "paths": {"runs": str(self.run_dir.relative_to(ROOT)),
            "checkpoints": str(self.checkpoint_dir.relative_to(ROOT)), "reports": str(self.report_dir.relative_to(ROOT))},
        }

    @staticmethod
    def backend_revision() -> dict[str, str]:
        native_path = Path(gipf_engine.__file__).resolve()
        files = [native_path, ROOT / "training" / "native_search.py", ROOT / "training" / "search.py"]
        return {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): digest(path) for path in files}

    def assert_backend_unchanged(self) -> None:
        if self.backend_revision() != self.backend:
            raise RuntimeError("native/search backend changed during comparison; aborting to preserve fairness")

    def remaining(self) -> float:
        return self.deadline - time.time()

    def emit(self, event: str, **extra: Any) -> None:
        row = {"event": event, "time": time.time(), "remaining_seconds": max(0.0, self.remaining()), **extra}
        with self.heartbeat.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
        temporary = self.summary_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.summary_path)
        print(json.dumps(row, sort_keys=True), flush=True)

    def record_error(self, label: str, error: Exception) -> None:
        row = {"label": label, "type": type(error).__name__, "message": str(error)}
        self.summary["errors"].append(row)
        self.emit("error", **row)

    def child(self, label: str, command: list[str], log: Path, outer_seconds: int) -> bool:
        """Bound one child group, emitting durable progress and preserving logs."""
        if self.remaining() <= 20:
            self.emit("skipped_deadline", label=label)
            return False
        allowed = min(outer_seconds, max(1, int(self.remaining() - 15)))
        log.parent.mkdir(parents=True, exist_ok=True)
        self.emit("child_start", label=label, command=command, log=str(log.relative_to(ROOT)), outer_seconds=allowed)
        process: subprocess.Popen[str] | None = None
        try:
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen([sys.executable, "-u", *command], cwd=ROOT, stdout=stream,
                                           stderr=subprocess.STDOUT, text=True, start_new_session=True)
                started = time.monotonic()
                next_heartbeat = started + 30
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    if elapsed >= allowed or self.remaining() <= 0:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=12)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=8)
                        raise TimeoutError(f"{label} exceeded {allowed}s")
                    if time.monotonic() >= next_heartbeat:
                        self.emit("child_running", label=label, elapsed_seconds=round(elapsed, 1))
                        next_heartbeat += 30
                    time.sleep(2)
            if process.returncode:
                raise RuntimeError(f"{label} exited {process.returncode}; see {log}")
            self.emit("child_complete", label=label)
            return True
        except Exception as error:
            self.record_error(label, error)
            return False

        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=8)

    def freeze(self, name: str, source: Path) -> Path | None:
        try:
            raw = source.read_bytes()
            hash_value = hashlib.sha256(raw).hexdigest()
            target = self.checkpoint_dir / f"{name}-{hash_value[:12]}.pt"
            if target.exists():
                raise FileExistsError(target)
            temporary = target.with_suffix(".tmp")
            temporary.write_bytes(raw)
            os.replace(temporary, target)
            self.emit("checkpoint_frozen", name=name, source=str(source.relative_to(ROOT)),
                      checkpoint=str(target.relative_to(ROOT)), sha256=hash_value)
            return target
        except Exception as error:
            self.record_error(f"freeze:{name}", error)
            return None

    def train_one(self, architecture: dict[str, Any], seed: int) -> Path | None:
        self.assert_backend_unchanged()
        name = f"{architecture['name']}-seed{seed}"
        folder = self.run_dir / name
        train = self.config["training"]
        started = time.monotonic()
        ok = False
        for attempt in range(2):
            remaining_run = int(self.config["training_seconds"] - (time.monotonic() - started))
            if remaining_run <= 0:
                break
            command = ["-m", "training.train", "--run", str(folder), "--kind", architecture["kind"],
                       "--head", train["head"], "--width", str(architecture["width"]), "--blocks", str(architecture["blocks"]),
                       "--seed", str(seed), "--seconds", str(remaining_run), "--games", str(train["games"]),
                       "--games-per-iteration", str(train["games_per_iteration"]), "--updates", str(train["updates"]),
                       "--simulations", str(train["simulations"]), "--batch-size", str(train["batch_size"]), "--lr", str(train["lr"]),
                       "--max-ply", str(train["max_ply"]), "--replay-size", str(train["replay_size"]), "--device", train["device"]]
            if attempt:
                resume = folder / "latest.pt"
                if not resume.exists():
                    break
                command.extend(["--resume", str(resume)])
                self.emit("training_recovery", name=name, attempt=attempt + 1, resume=str(resume.relative_to(ROOT)))
            if train["native_forest"]:
                command.append("--native-forest")
            ok = self.child(f"train-{name}-attempt{attempt + 1}", command, folder / f"attempt{attempt + 1}.log", remaining_run + 30)
            if ok:
                break
        latest = folder / "latest.pt"
        frozen = self.freeze(name, latest) if latest.exists() else None
        self.summary["training"].append({"name": name, "completed": ok, "checkpoint": None if frozen is None else str(frozen.relative_to(ROOT))})
        return frozen

    def duel(self, candidate: Path, champion: Path, mode: str, seed: int) -> None:
        self.assert_backend_unchanged()
        spec = self.config["comparisons"][mode]
        stem = f"{candidate.stem}-vs-{champion.stem}-{mode}-seed{seed}"
        output = self.report_dir / f"{stem}.json"
        command = ["-m", "training.duel", "--candidate", str(candidate), "--champion", str(champion),
                   "--games", str(spec["games"]), "--batch", str(spec["batch"]), "--device", spec["device"],
                   "--threads", str(spec["threads"]), "--simulations", str(spec["simulations"]),
                   "--budget-ms", str(spec["budget_ms"]), "--seed", str(seed), "--max-ply", "300", "--output", str(output),
                   "--seconds", str(spec["seconds"])]
        complete = self.child(f"duel-{stem}", command, self.report_dir / f"{stem}.log", spec["seconds"] + 30)
        record: dict[str, Any] = {"candidate": str(candidate.relative_to(ROOT)), "champion": str(champion.relative_to(ROOT)),
                                  "mode": mode, "seed": seed, "report": str(output.relative_to(ROOT)), "completed": complete}
        if complete and output.exists():
            try:
                data = json.loads(output.read_text(encoding="utf-8"))
                record["result"] = {key: data.get(key) for key in ("wins", "losses", "cutoffs", "unfinished", "wilson_95_decisive")}
            except Exception as error:
                self.record_error(f"read:{stem}", error)
        self.summary["duels"].append(record)

    def validate(self) -> None:
        expected = {"simulations": 32, "games": 64, "games_per_iteration": 32, "updates": 32,
                    "batch_size": 256, "lr": 0.001, "max_ply": 300, "replay_size": 100000}
        for key, value in expected.items():
            if self.config["training"].get(key) != value:
                raise ValueError(f"configuration must retain {key}={value}")
        if len(self.config["architectures"]) != 4 or len(self.config["seeds"]) != 2:
            raise ValueError("comparison requires exactly four architectures and two seeds")
        benchmark = ROOT / self.config["required_benchmark_report"]
        if not benchmark.exists():
            raise FileNotFoundError(f"native-forest benchmark must complete before launch: {benchmark}")
        json.loads(benchmark.read_text(encoding="utf-8"))

    def run(self) -> int:
        try:
            self.validate()
        except Exception as error:
            self.record_error("validation", error)
            self.summary["status"] = "blocked"
            self.emit("blocked")
            return 1
        frozen: dict[tuple[str, int], Path] = {}
        for architecture in self.config["architectures"]:
            for seed in self.config["seeds"]:
                if self.remaining() <= 20:
                    self.emit("deadline_before_training")
                    break
                try:
                    candidate = self.train_one(architecture, seed)
                    if candidate is not None:
                        frozen[(architecture["name"], seed)] = candidate
                except Exception as error:
                    self.record_error(f"train:{architecture['name']}:{seed}", error)
        # All training is intentionally complete before any pairwise evidence.
        architectures = self.config["architectures"]
        for left_index, left in enumerate(architectures):
            for right in architectures[left_index + 1:]:
                for seed_index, seed in enumerate(self.config["seeds"]):
                    candidate = frozen.get((left["name"], seed))
                    champion = frozen.get((right["name"], seed))
                    if candidate is None or champion is None:
                        self.emit("duel_skipped_missing_checkpoint", candidate=left["name"], champion=right["name"], seed=seed)
                        continue
                    for mode_index, mode in enumerate(("equal_simulations", "equal_cpu_time")):
                        if self.remaining() <= 20:
                            self.emit("deadline_before_duel")
                            break
                        try:
                            self.duel(candidate, champion, mode, 970000 + left_index * 1000 + seed_index * 10 + mode_index)
                        except Exception as error:
                            self.record_error(f"duel:{left['name']}:{right['name']}:{seed}:{mode}", error)
        self.summary["status"] = "complete" if self.remaining() > 0 else "deadline"
        self.summary["finished_epoch"] = time.time()
        self.emit("runner_finished", status=self.summary["status"])
        return 0 if self.summary["status"] == "complete" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "ops" / "architecture_comparison_config.json")
    parser.add_argument("--deadline", type=float, required=True, help="absolute Unix epoch; runner never exceeds it")
    args = parser.parse_args()
    if args.deadline <= time.time():
        parser.error("--deadline must be in the future")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    return Runner(config, args.deadline).run()


if __name__ == "__main__":
    raise SystemExit(main())
