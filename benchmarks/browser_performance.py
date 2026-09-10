"""Matched shipped-WASM and native-PyTorch search timings for the champion.

This is a benchmark tool, not a CI test: timings are intentionally sequential
to avoid CPU contention and each sample builds a new MCTS tree.
"""
from __future__ import annotations

import functools
import hashlib
import json
import platform
import statistics
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from playwright.sync_api import sync_playwright

from training.model import load_model
from training.search import choose_action

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_IDS = (0, 3, 8, 15)
SAMPLES = 5
SIMULATIONS = 128


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def native(model, fixtures: list[dict]) -> dict:
    torch.set_num_threads(1)
    rows = []
    for fixture in fixtures:
        state = __import__("gipf_engine").State.from_dict(fixture["state"])
        # Same code path and a whole search warmup, excluded from samples.
        choose_action(model, state, "cpu", simulations=SIMULATIONS, budget_ms=30_000)
        samples = []
        for _ in range(SAMPLES):
            started = time.perf_counter()
            action, detail = choose_action(model, state, "cpu", simulations=SIMULATIONS, budget_ms=30_000)
            samples.append((time.perf_counter() - started) * 1000)
            if action not in fixture["legal_actions"] or detail["simulations"] != SIMULATIONS:
                raise AssertionError("native search did not finish a legal 128-simulation tree")
        rows.append({"fixture": fixture["id"], "phase": fixture["phase"], "samples_ms": samples,
                     "median_ms": median(samples), "action": action, "simulations": detail["simulations"]})
    feature = np.asarray(fixtures[0]["encoded_input"], dtype=np.float32)[None]
    with torch.inference_mode():
        x = torch.from_numpy(feature)
        for _ in range(5): model(x)
        forward = []
        for _ in range(50):
            started = time.perf_counter(); model(x); forward.append((time.perf_counter() - started) * 1000)
    return {"search": rows, "forward_batch1": {"warmup": 5, "repeats": 50, "samples_ms": forward, "median_ms": median(forward)}}


def browser(fixtures: list[dict]) -> tuple[dict, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(ROOT / "web")))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    blocked: list[str] = []
    try:
        with sync_playwright() as pw:
            chromium = pw.chromium.launch(headless=True)
            page = chromium.new_page(); page.set_default_timeout(90_000)
            page.route("**/*", lambda route: route.continue_() if route.request.url.startswith(base) else (blocked.append(route.request.url), route.abort())[1])
            page.goto(base + "/models/champion-metadata.json")
            page.evaluate("""() => { window.worker=new Worker('/ai-worker.js'); let serial=0;
              window.rpc=(body)=>new Promise((resolve,reject)=>{const id=++serial;const timeout=setTimeout(()=>reject(Error('worker timeout')),60000);
              function recv(e){if(e.data.id!==id)return;clearTimeout(timeout);worker.removeEventListener('message',recv);resolve(e.data)}
              worker.addEventListener('message',recv);worker.postMessage({...body,id})}) }""")
            ready = page.evaluate("() => rpc({type:'init', backend:'wasm', modelVariant:'dynamic'})")
            if ready.get("error"): raise RuntimeError(ready["error"])
            rows = []
            for fixture in fixtures:
                request = {"type": "search", "state": fixture["state"], "simulations": SIMULATIONS, "budgetMs": 10_000}
                page.evaluate("request => rpc(request)", request)  # excluded warm tree
                samples = []; worker_elapsed = []; response = None
                for _ in range(SAMPLES):
                    started = time.perf_counter(); response = page.evaluate("request => rpc(request)", request)
                    samples.append((time.perf_counter() - started) * 1000)
                    if response.get("error"): raise RuntimeError(response["error"])
                    if response["action"] not in fixture["legal_actions"] or response["stats"]["simulations"] != SIMULATIONS:
                        raise AssertionError("browser search did not finish a legal 128-simulation tree")
                    worker_elapsed.append(response["stats"]["elapsedMs"])
                rows.append({"fixture": fixture["id"], "phase": fixture["phase"], "samples_ms": samples,
                             "median_ms": median(samples), "worker_elapsed_samples_ms": worker_elapsed, "action": response["action"],
                             "simulations": response["stats"]["simulations"]})
            feature = [fixtures[0]["encoded_input"]]
            forward = page.evaluate("request => rpc(request)", {"type": "benchmark", "features": feature, "repeats": 50})
            if forward.get("error"): raise RuntimeError(forward["error"])
            chromium.close()
            return ({"ready": ready, "search": rows, "forward_batch1": {"warmup": 5, "repeats": 50,
                    "samples_ms": forward["timesMs"], "median_ms": forward["medianMs"]}, "blocked_requests": blocked}, pw.chromium.executable_path)
    finally:
        server.shutdown()


def main() -> None:
    fixtures_file = ROOT / "web/models/champion-fixtures.json"
    data = json.loads(fixtures_file.read_text())
    fixtures = [data["fixtures"][i] for i in FIXTURE_IDS]
    model, checkpoint = load_model(ROOT / "checkpoints/champion.pt", "cpu")
    native_result = native(model, fixtures)
    browser_result, executable = browser(fixtures)
    native_by_id = {r["fixture"]: r for r in native_result["search"]}
    ratios = []
    for row in browser_result["search"]:
        reference = native_by_id[row["fixture"]]
        if row["action"] != reference["action"]:
            raise AssertionError(f"action mismatch at fixture {row['fixture']}")
        ratios.append(row["median_ms"] / reference["median_ms"])
    report = {"schema": "gipf-browser-performance-v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "protocol": {"variant": "dynamic", "backend": "wasm", "simulations": SIMULATIONS, "samples": SAMPLES,
                           "warmup_searches_per_fixture": 1, "fresh_tree_per_sample": True, "sequential": True,
                           "fixtures": list(FIXTURE_IDS), "native_threads": 1, "browser_threads": 1},
              "model": {"checkpoint": "checkpoints/champion.pt", "checkpoint_sha256": sha256(ROOT / "checkpoints/champion.pt"),
                        "onnx_sha256": sha256(ROOT / "web/models/champion.onnx"), "architecture": checkpoint["config"]},
              "runtime": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                          "browser_executable": executable, "browser_runtime": browser_result["ready"]},
              "native_pytorch_choose_action": native_result, "browser_wasm_search": browser_result,
              "browser_to_native_search_median_ratios": ratios, "median_of_position_ratios": median(ratios)}
    out = ROOT / "reports/browser-performance-final.json"; out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(out), "ratios": ratios, "median_ratio": report["median_of_position_ratios"]}, indent=2))


if __name__ == "__main__":
    main()
