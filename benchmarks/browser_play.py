"""Run local browser-WASM GIPF games against deterministic native baselines.

This is a release benchmark, deliberately not a normal test: it launches a
real Chromium worker and can take several minutes.  All browser traffic other
than the temporary local static server is aborted and recorded.
"""
from __future__ import annotations

import argparse
import functools
import json
import random
import statistics
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from playwright.sync_api import sync_playwright

import gipf_engine as ge
from baselines import GreedyPlayer, RandomPlayer
from training.model import load_model
from training.native_search import NativeBatchedMCTS, NativeNode
from training.search import BatchedMCTS, Node, choose_action

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


RPC_SETUP = """() => {
  window.worker?.terminate(); window.worker = new Worker('/ai-worker.js'); let serial = 0;
  window.rpc = (body) => new Promise((resolve, reject) => {
    const id = ++serial; const timeout = setTimeout(() => { worker.removeEventListener('message', receive); reject(Error('worker timeout')); }, 60000);
    function receive(e) { if (e.data.id !== id) return; clearTimeout(timeout); worker.removeEventListener('message', receive); resolve(e.data); }
    worker.addEventListener('message', receive); worker.postMessage({...body, id});
  });
}"""


class BrowserPlayer:
    def __init__(self, page, simulations: int, budget_ms: int):
        self.page, self.simulations, self.budget_ms = page, simulations, budget_ms
        self.calls: list[dict] = []

    def choose_action(self, state):
        response = self.page.evaluate("(request) => rpc(request)", {
            "type": "search", "state": state.serialize(), "simulations": self.simulations, "budgetMs": self.budget_ms,
        })
        if response.get("error"):
            raise RuntimeError(response["error"])
        action, stats = int(response["action"]), response["stats"]
        if action not in state.legal_actions():
            raise RuntimeError(f"browser selected illegal action {action}")
        self.calls.append({"phase": state.phase, "action": action, "stats": stats})
        return action


def native_result(model, state, native_forest: bool) -> dict:
    if native_forest:
        root = NativeNode(state.clone()); search = NativeBatchedMCTS(model, "cpu", seed=0)
    else:
        root = Node(state.clone()); search = BatchedMCTS(model, "cpu", seed=0)
    policy = search.search([root], simulations=128, noise=False)[0]
    return {"action": int(root.actions[int(np.argmax(policy))]), "actions": [int(x) for x in root.actions],
            "visits": [int(x) for x in root.n], "root_visits": int(root.n.sum())}


def fixture_comparison(page, model, fixtures: list[dict]) -> list[dict]:
    rows = []
    for fixture in fixtures:
        response = page.evaluate("(request) => rpc(request)", {"type": "search", "state": fixture["state"], "simulations": 128, "budgetMs": 10_000})
        if response.get("error"):
            raise RuntimeError(response["error"])
        browser = {"action": int(response["action"]), **response["stats"]}
        state = ge.State.from_dict(fixture["state"])
        python_tree = native_result(model, state, native_forest=False)
        forest = native_result(model, state, native_forest=True)
        if browser["action"] not in fixture["legal_actions"]:
            raise RuntimeError(f"browser fixture {fixture['id']} selected illegal action")
        rows.append({"fixture": fixture["id"], "phase": fixture["phase"], "browser": browser,
                     "python_batched": python_tree, "native_forest": forest,
                     "action_parity": {"browser_python": browser["action"] == python_tree["action"],
                                       "browser_forest": browser["action"] == forest["action"]},
                     "root_visit_equal": {"browser_python": browser.get("visits") == python_tree["visits"],
                                          "browser_forest": browser.get("visits") == forest["visits"]}})
    return rows


def keep_double_capture_check(fixtures: list[dict]) -> dict:
    """Exercise the engine's legal zero-mask choice, which retains doubles."""
    for fixture in fixtures:
        if fixture["phase"] != "capture":
            continue
        state = ge.State.from_dict(fixture["state"])
        action = next((int(value) for value in state.legal_actions() if value >= 42 and (value - 42) % 128 == 0), None)
        if action is None:
            continue
        before = sum(abs(piece) == 2 for piece in state.board)
        state.apply(action)
        after = sum(abs(piece) == 2 for piece in state.board)
        if before != after:
            raise RuntimeError("zero-mask capture unexpectedly removed a double")
        return {"fixture": fixture["id"], "action": action, "doubles_before": before, "doubles_after": after}
    raise RuntimeError("no fixture offered a zero-mask capture action")


def play_game(browser: BrowserPlayer, baseline, browser_color: int, game_id: int, seed: int, deadline: float) -> dict:
    if hasattr(baseline, "set_game_seed"):
        baseline.set_game_seed(game_id // 2, 1)
    state = ge.State(); started = time.monotonic(); decisions = 0; captures = []
    while not state.winner and state.ply < 300 and decisions < 2000:
        if time.monotonic() >= deadline:
            return {"game": game_id, "seed": seed, "browser_color": browser_color, "winner": 0, "reason": "global_timeout",
                    "ply": state.ply, "decisions": decisions, "captures": captures, "seconds": time.monotonic() - started}
        actor = browser if state.current_player == browser_color else baseline
        action = browser.choose_action(state) if actor is browser else int(actor.choose_action(state))
        if action not in state.legal_actions():
            raise RuntimeError(f"illegal action {action} during game {game_id}")
        if state.phase == "capture":
            captures.append({"action": action, "mask": (action - 42) % 128, "kept_all_optional_doubles": (action - 42) % 128 == 0})
        state.apply(action); decisions += 1
    reason = "winner" if state.winner else "insertion_cutoff" if state.ply >= 300 else "decision_cutoff"
    return {"game": game_id, "seed": seed, "browser_color": browser_color, "winner": state.winner, "reason": reason,
            "ply": state.ply, "decisions": decisions, "captures": captures, "seconds": time.monotonic() - started}


def match(browser, opponent: str, games: int, seed: int, deadline: float) -> dict:
    records = []
    for game_id in range(games):
        if time.monotonic() >= deadline:
            break
        baseline = RandomPlayer(seed=seed) if opponent == "random" else GreedyPlayer(seed=seed)
        records.append(play_game(browser, baseline, 1 if game_id % 2 == 0 else -1, game_id, seed + game_id // 2, deadline))
    wins = sum(row["winner"] == row["browser_color"] for row in records)
    losses = sum(row["winner"] and row["winner"] != row["browser_color"] for row in records)
    cutoffs = sum(row["winner"] == 0 and row["reason"] != "global_timeout" for row in records)
    timeouts = sum(row["reason"] == "global_timeout" for row in records)
    kept = sum(item["kept_all_optional_doubles"] for row in records for item in row["captures"])
    return {"opponent": opponent, "games_requested": games, "games_completed": len(records), "wins": wins, "losses": losses,
            "cutoffs": cutoffs, "global_timeouts": timeouts, "capture_decisions": sum(len(row["captures"]) for row in records),
            "keep_all_optional_double_choices": kept, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports/browser-play.json")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--budget-ms", type=int, default=10_000)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    fixtures = json.loads((ROOT / "web/models/champion-fixtures.json").read_text())["fixtures"]
    model, _ = load_model(ROOT / "checkpoints/champion.pt", "cpu"); model.eval(); torch.set_num_threads(1)
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(ROOT / "web")))
    threading.Thread(target=server.serve_forever, daemon=True).start(); base = f"http://127.0.0.1:{server.server_port}"
    result = {"schema": "gipf-browser-play-v1", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "protocol": {"browser": "Chromium incognito context", "external_requests": "blocked", "simulations": args.simulations,
                           "budget_ms": args.budget_ms, "games_per_opponent": args.games, "max_insertion_plies": 300,
                           "global_timeout_seconds": args.timeout_seconds}, "blocked_requests": []}
    deadline = time.monotonic() + args.timeout_seconds
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True); context = browser.new_context(); page = context.new_page(); page.set_default_timeout(60_000)
            def route(route):
                if route.request.url.startswith(base): route.continue_()
                else: result["blocked_requests"].append(route.request.url); route.abort()
            page.route("**/*", route); page.goto(base + "/models/champion-metadata.json"); page.evaluate(RPC_SETUP)
            init = page.evaluate("() => rpc({type:'init', backend:'wasm'})")
            if init.get("error"):
                raise RuntimeError(init["error"])
            result["browser"] = {"version": browser.version, "init": init}
            result["keep_double_capture_check"] = keep_double_capture_check(fixtures)
            result["fixtures"] = fixture_comparison(page, model, fixtures)
            browser_player = BrowserPlayer(page, args.simulations, args.budget_ms)
            result["matches"] = [match(browser_player, "random", args.games, 730001, deadline),
                                 match(browser_player, "greedy", args.games, 730101, deadline)]
            result["browser_search_calls"] = len(browser_player.calls)
            result["terminal_games"] = sum(row["winner"] != 0 for match_result in result["matches"] for row in match_result["records"])
            context.close(); browser.close()
        result["completed"] = all(row["games_completed"] == args.games for row in result["matches"])
    except Exception as error:
        result["completed"] = False; result["error"] = f"{type(error).__name__}: {error}"
    finally:
        server.shutdown()
    result["elapsed_seconds"] = args.timeout_seconds - max(0, deadline - time.monotonic())
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result.get(key) for key in ("completed", "error", "elapsed_seconds")}), flush=True)
    if not result["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
