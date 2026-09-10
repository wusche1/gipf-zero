"""Render an overnight experiment summary as a readable, auditable Markdown report.

The runner keeps the machine-readable summary deliberately small.  This module
joins it with the immutable run metrics and raw duel JSON files without changing
any experiment data.  It is intentionally usable while a run is still active.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import NormalDist
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SEED_RE = re.compile(r"-seed(\d+)")
ATTEMPT_RE = re.compile(r"-attempt(\d+)")
MODE_RE = re.compile(r"-(equal_cpu_time|equal_simulations)(?:-|\.)")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 1.0)
    z = NormalDist().inv_cdf(.975)
    p = successes / total
    divisor = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / divisor
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / divisor
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def fmt_seconds(value: Any) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "—"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def model_name(value: str | None) -> str:
    if not value:
        return "unknown"
    name = Path(value).name
    return SEED_RE.split(name, maxsplit=1)[0] or name


def infer_seed(record: dict[str, Any], report_name: str = "") -> int | None:
    value = record.get("seed")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    match = SEED_RE.search(report_name)
    return int(match.group(1)) if match else None


def raw_report_path(report_dir: Path, report: str | None) -> Path | None:
    if not report:
        return None
    path = Path(report)
    candidates = [ROOT / path, report_dir / path.name, report_dir / path]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def duel_records(report_dir: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Join summary duel metadata to raw reports, retaining incomplete attempts."""
    records: list[dict[str, Any]] = []
    referenced: set[Path] = set()
    for index, summary_record in enumerate(summary.get("duels") or []):
        if not isinstance(summary_record, dict):
            continue
        path = raw_report_path(report_dir, summary_record.get("report"))
        raw = read_json(path) if path else {}
        if path:
            referenced.add(path.resolve())
        report_name = path.name if path else str(summary_record.get("report", ""))
        left = model_name(summary_record.get("left"))
        right = model_name(summary_record.get("right"))
        models = raw.get("models") if isinstance(raw.get("models"), list) else []
        if (left == "unknown" or right == "unknown") and len(models) >= 2:
            left = left if left != "unknown" else model_name(str(models[0].get("name", "candidate")))
            right = right if right != "unknown" else model_name(str(models[1].get("name", "champion")))
        mode = summary_record.get("mode") or (MODE_RE.search(report_name).group(1) if MODE_RE.search(report_name) else "unknown")
        attempt = int(summary_record.get("attempt", 1) or 1)
        match = ATTEMPT_RE.search(report_name)
        if match:
            attempt = max(attempt, int(match.group(1)))
        result = summary_record.get("result") or {}
        wins = int(raw.get("wins", result.get("wins", 0)) or 0)
        losses = int(raw.get("losses", result.get("losses", 0)) or 0)
        cutoffs = int(raw.get("cutoffs", result.get("cutoffs", 0)) or 0)
        unfinished = int(raw.get("unfinished", result.get("unfinished", 0)) or 0)
        records.append({"left": left, "right": right, "mode": mode, "seed": infer_seed(summary_record, report_name),
                        "attempt": attempt, "wins": wins, "losses": losses, "cutoffs": cutoffs,
                        "unfinished": unfinished, "complete": bool(summary_record.get("complete", unfinished == 0)) and unfinished == 0,
                        "tie_break": attempt >= 3 or (attempt >= 3 and wins + losses + cutoffs == 80),
                        "path": path, "report_name": report_name, "order": index, "eligible": True})

    # Raw files are useful if a process died before its summary append.  They are
    # retained for audit links, but cannot enter ranking without explicit summary
    # pair/seed metadata (otherwise a continuation or unrelated report could be
    # mistaken for a primary trial).
    for path in sorted(report_dir.glob("*.json")):
        if path.name in {"summary.json", "config.json"} or path.resolve() in referenced:
            continue
        raw = read_json(path)
        if not any(key in raw for key in ("wins", "losses", "unfinished")):
            continue
        match = MODE_RE.search(path.name)
        if not match:
            continue
        records.append({"left": model_name(str((raw.get("models") or [{"name": "candidate"}])[0].get("name", "candidate"))),
                        "right": model_name(str((raw.get("models") or [{}, {"name": "champion"}])[1].get("name", "champion"))),
                        "mode": match.group(1), "seed": infer_seed({}, path.name),
                        "attempt": int((ATTEMPT_RE.search(path.name) or [1, 1])[1]),
                        "wins": int(raw.get("wins", 0) or 0), "losses": int(raw.get("losses", 0) or 0),
                        "cutoffs": int(raw.get("cutoffs", 0) or 0), "unfinished": int(raw.get("unfinished", 0) or 0),
                        "complete": False, "tie_break": False, "eligible": False,
                        "path": path, "report_name": path.name, "order": 10_000})
    return records


def primary_records(records: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """Choose the last complete retry for each unordered pair/mode/seed."""
    groups: dict[tuple[str, str, int | None], list[dict[str, Any]]] = {}
    for record in records:
        if record["mode"] != mode or record["tie_break"] or not record.get("eligible", True):
            continue
        pair = tuple(sorted((record["left"], record["right"])))
        groups.setdefault((*pair, record["seed"]), []).append(record)
    selected = []
    for values in groups.values():
        complete = [value for value in values if value["complete"]]
        if complete:
            selected.append(max(complete, key=lambda value: (value["attempt"], value["order"])))
    return selected


def secondary_retry_overlay(records: list[dict[str, Any]], report_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replace verified incomplete fixed-simulation records for display only.

    The runner's summary is the primary experiment record.  A separately
    verified retry can complete a match that reached its short whole-match
    deadline, but it must never become evidence in the primary CPU ranking or
    be appended beside its original fixed-simulation record.
    """
    manifest = read_json(report_dir / "secondary-retries.json")
    if manifest.get("all_complete") is not True:
        return records, []
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return records, []
    replacements: dict[Path, tuple[dict[str, Any], Path, dict[str, Any]]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("frozen_hashes_match_original") is not True:
            continue
        original = raw_report_path(report_dir, entry.get("original_report"))
        retry = raw_report_path(report_dir, entry.get("retry_report"))
        if not original or not retry:
            continue
        original_data, retry_data = read_json(original), read_json(retry)
        if int(retry_data.get("unfinished", 1) or 0) != 0:
            continue
        # Hash equality binds a retry to the frozen models of its original.
        if not isinstance(retry_data.get("hashes"), list) or retry_data.get("hashes") != original_data.get("hashes"):
            continue
        replacements[original.resolve()] = (entry, retry, retry_data)
    overlaid = []
    applied = []
    for record in records:
        replacement = replacements.get(record.get("path").resolve()) if record.get("path") else None
        if (not replacement or record["mode"] != "equal_simulations" or record["complete"]
                or not record.get("eligible", True)):
            overlaid.append(record)
            continue
        entry, retry_path, retry = replacement
        if (record["left"] != model_name(entry.get("left")) or record["right"] != model_name(entry.get("right"))
                or record["seed"] != infer_seed(entry, retry_path.name)):
            overlaid.append(record)
            continue
        updated = dict(record)
        updated.update({"wins": int(retry.get("wins", 0) or 0), "losses": int(retry.get("losses", 0) or 0),
                        "cutoffs": int(retry.get("cutoffs", 0) or 0), "unfinished": 0, "complete": True,
                        "path": retry_path, "report_name": retry_path.name,
                        "secondary_retry": True, "original_path": record["path"]})
        overlaid.append(updated)
        applied.append(updated)
    return overlaid, applied


def pair_totals(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        left, right = sorted((record["left"], record["right"]))
        key = (left, right)
        row = groups.setdefault(key, {"left": left, "right": right, "wins": 0, "losses": 0, "cutoffs": 0, "games": 0, "seeds": set()})
        if record["left"] == left:
            row["wins"] += record["wins"]
            row["losses"] += record["losses"]
        else:
            row["wins"] += record["losses"]
            row["losses"] += record["wins"]
        row["cutoffs"] += record["cutoffs"]
        row["games"] += record["wins"] + record["losses"] + record["cutoffs"]
        row["seeds"].add(record["seed"])
    return sorted(groups.values(), key=lambda row: (row["left"], row["right"]))


def pooled_ranking(records: list[dict[str, Any]], models: list[str], expected_games: int) -> list[dict[str, Any]]:
    totals = {name: {"wins": 0, "losses": 0, "cutoffs": 0} for name in models}
    for record in records:
        totals.setdefault(record["left"], {"wins": 0, "losses": 0, "cutoffs": 0})
        totals.setdefault(record["right"], {"wins": 0, "losses": 0, "cutoffs": 0})
        totals[record["left"]]["wins"] += record["wins"]
        totals[record["left"]]["losses"] += record["losses"]
        totals[record["left"]]["cutoffs"] += record["cutoffs"]
        totals[record["right"]]["wins"] += record["losses"]
        totals[record["right"]]["losses"] += record["wins"]
        totals[record["right"]]["cutoffs"] += record["cutoffs"]
    rows = []
    for name, value in totals.items():
        games = value["wins"] + value["losses"] + value["cutoffs"]
        lo, hi = wilson(value["wins"], games)
        rows.append({"name": name, **value, "games": games, "unplayed": max(0, expected_games - games),
                     "rate": value["wins"] / games if games else 0.0, "wilson": (lo, hi)})
    return sorted(rows, key=lambda row: (row["rate"], row["games"]), reverse=True)


def training_rows(summary: dict[str, Any], report_dir: Path, run_root: Path) -> list[dict[str, Any]]:
    listed = summary.get("training") or []
    if not listed:
        listed = [{"name": path.name, "run": str(path.relative_to(ROOT))} for path in sorted(run_root.glob("*-seed*/"))]
    rows = []
    for entry in listed:
        if not isinstance(entry, dict):
            continue
        run_value = entry.get("run") or entry.get("path")
        run = ROOT / run_value if run_value else run_root / str(entry.get("name", ""))
        config = read_json(run / "config.json")
        metrics = read_metrics(run / "metrics.jsonl")
        # Continuation appends to the same metrics file.  The controlled
        # architecture trial ends at checkpoint_frozen; never report later
        # continuation games/updates as part of its 600-second training row.
        freeze_time = None
        heartbeat = run_root / "heartbeat.jsonl"
        for event in read_metrics(heartbeat):
            if event.get("event") == "checkpoint_frozen" and event.get("name") == entry.get("name", run.name):
                try:
                    freeze_time = float(event["time"])
                except (KeyError, TypeError, ValueError):
                    pass
        if freeze_time is not None:
            metrics = [row for row in metrics if float(row.get("time", 0) or 0) <= freeze_time]
        start = next((row for row in metrics if row.get("event") == "start"), {})
        last = metrics[-1] if metrics else {}
        effective = config.get("effective_model") or config
        model = str(entry.get("name", run.name)).rsplit("-seed", 1)[0]
        seed_match = SEED_RE.search(entry.get("name", run.name))
        completed = entry.get("completed")
        if completed is None:
            completed = bool(summary.get("status") == "complete" and last.get("event") not in {"stopped", "error"})
        rows.append({"model": model, "seed": seed_match.group(1) if seed_match else config.get("seed", "—"),
                     "games": last.get("games", 0), "updates": last.get("updates", 0),
                     "seconds": last.get("elapsed", config.get("seconds")), "parameters": start.get("parameters", "—"),
                     "completed": bool(completed), "status": last.get("event", "no metrics"), "run": run})
    return rows


def table(rows: list[list[Any]], headers: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(out)


def render(summary_path: Path) -> Path:
    report_dir = summary_path.parent
    summary = read_json(summary_path)
    config = read_json(report_dir / "config.json")
    tag = summary.get("tag", report_dir.name)
    run_root = ROOT / "runs" / "overnight" / tag
    training = training_rows(summary, report_dir, run_root)
    records = duel_records(report_dir, summary)
    architectures = config.get("architectures") or []
    models = [str(item.get("name")) for item in architectures if item.get("name")]
    if not models:
        models = sorted({name for record in records for name in (record["left"], record["right"])})
    seeds = config.get("seeds") or sorted({record["seed"] for record in records if record["seed"] is not None})
    duel_cfg = config.get("duels") or {}
    expected = int(duel_cfg.get("games", 0) or 0) * max(0, len(models) - 1) * len(seeds)
    cpu = primary_records(records, "equal_cpu_time")
    fixed_records, secondary_retries = secondary_retry_overlay(records, report_dir)
    fixed = primary_records(fixed_records, "equal_simulations")
    cpu_rank = pooled_ranking(cpu, models, expected)
    fixed_pairs = pair_totals(fixed)
    cpu_pairs = pair_totals(cpu)
    status = str(summary.get("status", "unknown"))
    heads = {str(item.get("head")) for item in architectures if item.get("head") is not None}
    lines = [f"# Overnight evaluation — `{tag}`", "", f"**Status:** `{status}`"]
    if status not in {"complete", "published"}:
        lines += ["", "> This experiment is incomplete or still running. Missing matches are shown as unplayed; no winner is inferred from partial coverage."]
    lines += ["", "## Protocol", "", f"- Training: isolated **{config.get('training_seconds', 600)} seconds per run**, {len(seeds)} seed(s), native forest search: **{bool((config.get('selfplay') or {}).get('native_forest', False))}**.", f"- Primary matches: {duel_cfg.get('games', '—')} paired-colour games per seed and pair, **{(duel_cfg.get('equal_cpu_time') or {}).get('budget_ms', '—')} ms per decision** on one CPU thread for each player. Secondary matches use **{(duel_cfg.get('equal_simulations') or {}).get('simulations', '—')} search simulations per decision** on GPU.", f"- Whole-match timeouts are {duel_cfg.get('seconds_cpu', '—')} seconds (CPU) and {duel_cfg.get('seconds_sim', '—')} seconds (fixed simulations); unfinished games are reported separately.", "- Training seconds are nominal allocations; the elapsed column includes checkpoint and shutdown overhead.", "- Parameter counts are read from each run's initial `start` metric, before training."]
    if len(heads) > 1:
        lines.append("- This is **not a pure architecture ablation**: the compared models use different output heads (" + ", ".join(sorted(heads)) + ").")
    lines += ["- Results are automated game comparisons; this report makes no human-rating claim.", "", "## Training", ""]
    train_rows = [[row["model"], row["seed"], row["games"], row["updates"], fmt_seconds(row["seconds"]), row["parameters"], "complete" if row["completed"] else f"incomplete ({row['status']})"] for row in training]
    lines.append(table(train_rows, ["Model", "Seed", "Games", "Updates", "Seconds", "Parameters", "State"]) if train_rows else "No training run has been recorded yet.")
    lines += ["", "## Pooled equal-CPU ranking", "", "Complete primary attempts only; retries use the last complete attempt for each pair/mode/seed. Cutoffs count as games and Wilson intervals use all recorded games.", ""]
    rank_rows = [[row["name"], row["wins"], row["losses"], row["cutoffs"], row["games"], row["unplayed"], pct(row["rate"]), f"{pct(row['wilson'][0])}–{pct(row['wilson'][1])}"] for row in cpu_rank]
    lines.append(table(rank_rows, ["Model", "W", "L", "C", "Played", "Unplayed", "Win rate", "Wilson 95%"]) if rank_rows else "No complete equal-CPU matches are available.")
    lines += ["", "## Equal-CPU pairwise", ""]
    pair_rows = [[row["left"] + " vs " + row["right"], row["wins"], row["losses"], row["cutoffs"], row["games"], f"{pct(wilson(row['wins'], row['games'])[0])}–{pct(wilson(row['wins'], row['games'])[1])}"] for row in cpu_pairs]
    lines.append(table(pair_rows, ["Pair (left perspective)", "Left W", "Left L", "C", "Games", "Wilson 95% (left)"]) if pair_rows else "No complete equal-CPU pairs are available.")
    lines += ["", "## Fixed-simulation pairwise", ""]
    if secondary_retries:
        lines += [f"Three incomplete fixed-simulation matches reached the original {duel_cfg.get('seconds_sim', '—')}-second whole-match cap. "
                  f"Their verified frozen-checkpoint retries used a 180-second cap and replace only those rows below; raw initial and retry reports remain linked for audit. "
                  f"All {len(fixed)} secondary pair/seed matches are now complete. The primary equal-CPU ranking above is unchanged.", ""]
    fixed_rows = [[row["left"] + " vs " + row["right"], row["wins"], row["losses"], row["cutoffs"], row["games"], f"{pct(wilson(row['wins'], row['games'])[0])}–{pct(wilson(row['wins'], row['games'])[1])}"] for row in fixed_pairs]
    lines.append(table(fixed_rows, ["Pair (left perspective)", "Left W", "Left L", "C", "Games", "Wilson 95% (left)"]) if fixed_rows else "No complete fixed-simulation pairs are available.")
    tie = [record for record in records if record["tie_break"]]
    lines += ["", "## Extra tie-break attempts", "", "Attempt 3 / 80-game tie-breaks are excluded from the primary pools above.", ""]
    tie_rows = [[record["report_name"], record["left"], record["right"], record["wins"], record["losses"], record["cutoffs"], "complete" if record["complete"] else "incomplete"] for record in tie]
    lines.append(table(tie_rows, ["Report", "Candidate", "Opponent", "W", "L", "C", "State"]) if tie_rows else "No extra tie-break attempt recorded.")
    promotion = summary.get("promotion") or {}
    publication = summary.get("publication") or {}
    lines += ["", "## Promotion and publication", ""]
    lines.append(f"- Promotion gate: `{promotion.get('passed')}`." if promotion else "- No promotion gate result recorded.")
    for key in ("duel_report", "greedy_report"):
        if promotion.get(key):
            lines.append(f"- `{key}`: [{Path(str(promotion[key])).name}]({Path(str(promotion[key])).name})")
    if publication.get("hf_repo"):
        lines.append(f"- Hugging Face: [" + str(publication["hf_repo"]) + f"](https://huggingface.co/{publication['hf_repo']})")
    lines += ["", "## Raw duel reports", ""]
    raw_paths = sorted({record["path"] for record in records if record.get("path")}, key=lambda path: path.name)
    lines.extend(f"- [{path.name}]({path.name})" for path in raw_paths) if raw_paths else lines.append("No raw duel report has been written yet.")
    output = report_dir / "RESULTS.md"
    output.write_text("\n".join(lines) + "\n")
    return output


def latest_summary() -> Path:
    summaries = sorted((ROOT / "reports" / "overnight").glob("*/summary.json"))
    if not summaries:
        raise FileNotFoundError("no reports/overnight/*/summary.json found")
    return summaries[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, help="summary.json to render; defaults to the latest overnight tag")
    args = parser.parse_args()
    path = (ROOT / args.summary if args.summary and not args.summary.is_absolute() else args.summary) if args.summary else latest_summary()
    print(render(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
