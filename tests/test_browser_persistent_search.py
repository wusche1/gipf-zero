"""The real worker: tree reuse, pondering, model swap, and complete games."""
import json
import random
import shutil
import subprocess
import sys
import time

import gipf_engine as ge
from test_browser_mapping import ROOT, WEB, QuietStaticHandler, browser, site_url
from test_browser_offline import versioned_site_url
from ops.web_model import positions

RPC = """() => {
  window.worker = new Worker('/ai-worker.js'); window.replies = {}; let serial = 0;
  worker.onmessage = (e) => { if (!e.data.progress) window.replies[e.data.id] = e.data; else window.progress = (window.progress || 0) + 1; };
  window.send = (body) => { const id = ++serial; worker.postMessage({...body, id}); return id; };
  window.reply = (id) => new Promise((resolve) => { const poll = () => window.replies[id] ? resolve(window.replies[id]) : setTimeout(poll, 5); poll(); });
  window.rpc = (body) => reply(send(body));
}"""


def worker_page(browser, url, model=None):
    page = browser.new_page()
    page.set_default_timeout(120_000)
    page.goto(url + "/config.json")
    page.evaluate(RPC)
    assert page.evaluate("(model) => rpc({type:'init', model})", model)["ready"]
    return page


def rpc(page, **body):
    response = page.evaluate("(body) => rpc(body)", body)
    assert "error" not in response, response
    return response


def test_search_reuses_the_subtree_of_the_played_move(browser, site_url):
    page = worker_page(browser, site_url)
    try:
        state = ge.State()
        first = rpc(page, type="search", state=state.serialize(), simulations=200)["stats"]
        assert first["reusedVisits"] == 0 and first["root_visits"] == 200
        assert first["pv"][0] == first["action"] and first["evalsPerSec"] > 0
        state.apply(first["action"])
        second = rpc(page, type="search", state=state.serialize(), simulations=20)["stats"]
        assert second["reusedVisits"] == max(first["visits"]) - 1
        assert second["root_visits"] == second["reusedVisits"] + 20
        assert second["action"] in state.legal_actions()
        assert second["white_value"] == -second["value"]
        # Undo: the old position is not below the new root, so the tree restarts.
        undone = rpc(page, type="search", state=ge.State().serialize(), simulations=5)["stats"]
        assert undone["reusedVisits"] == 0
        fresh = rpc(page, type="search", state=ge.State().serialize(), simulations=5, fresh=True)["stats"]
        assert fresh["reusedVisits"] == 0 and fresh["root_visits"] == 5
    finally:
        page.close()


def test_onnx_champion_still_searches_through_the_same_worker(browser, site_url):
    page = worker_page(browser, site_url, "champion")
    try:
        state = ge.State()
        first = rpc(page, type="search", state=state.serialize(), simulations=120)
        assert "43,869" in first["model"] and first["stats"]["root_visits"] == 120 == first["stats"]["simulations"]
        assert first["stats"]["evaluations"] == 121
        state.apply(first["action"])
        second = rpc(page, type="search", state=state.serialize(), budgetMs=200)["stats"]
        assert second["reusedVisits"] == max(first["stats"]["visits"]) - 1 and second["action"] in state.legal_actions()
        assert 0 < second["elapsedMs"] < 2_000
    finally:
        page.close()


def test_time_management_stops_early_and_skips_forced_decisions(browser, site_url):
    page = worker_page(browser, site_url)
    try:
        # A position where one move takes most of the visits: its lead grows
        # faster than the remaining budget could overturn it.
        for state in positions(40, seed=2):
            visits = sorted(rpc(page, type="search", state=state.serialize(), simulations=1500, fresh=True)["stats"]["visits"])
            if visits[-1] - (visits[-2] if len(visits) > 1 else 0) >= 1000 and len(visits) > 1:
                break
        else:
            raise AssertionError("no decisive position")
        settled = rpc(page, type="search", state=state.serialize(), budgetMs=3_000)["stats"]
        assert settled["reusedVisits"] == 1500 and 0 < settled["simulations"] and settled["elapsedMs"] < 2_000
        state = ge.State()
        rng = random.Random(3)
        while len(state.legal_actions()) != 1:
            if state.winner:
                state = ge.State()
            state.apply(rng.choice(state.legal_actions()))
        forced = rpc(page, type="search", state=state.serialize(), budgetMs=10_000)
        assert forced["action"] == state.legal_actions()[0]
        assert forced["stats"]["simulations"] == 0 and forced["stats"]["elapsedMs"] < 1_000
    finally:
        page.close()


def test_ponder_yields_to_search_and_to_stop(browser, site_url):
    page = worker_page(browser, site_url)
    try:
        state = ge.State().serialize()
        started = time.monotonic()
        ponder = page.evaluate("(state) => send({type:'ponder', state, progress:true})", state)
        page.wait_for_function("window.progress >= 2")
        search = rpc(page, type="search", state=state, simulations=10)
        pondered = page.evaluate("(id) => reply(id)", ponder)["stats"]
        assert time.monotonic() - started < 15
        assert pondered["ponder"] and 0 < pondered["root_visits"] < 20_000
        assert search["stats"]["reusedVisits"] == pondered["root_visits"]

        ponder = page.evaluate("(state) => send({type:'ponder', state})", state)
        page.wait_for_timeout(300)
        page.evaluate("send({type:'stop', ponder:true})")
        assert page.evaluate("(id) => reply(id)", ponder)["stats"]["elapsedMs"] < 5_000
        # A ponder that was already superseded when it reached the queue never starts.
        _, stale, last = page.evaluate("(state) => [send({type:'search', state, simulations:200}), send({type:'ponder', state}), send({type:'search', state, simulations:1})]", state)
        assert page.evaluate("(id) => reply(id)", stale)["error"] == "Search stopped."
        assert page.evaluate("(id) => reply(id)", last)["action"] in ge.State().legal_actions()
    finally:
        page.close()


def test_full_games_against_random_are_legal_and_finish(browser, site_url):
    page = worker_page(browser, site_url)
    rng = random.Random(11)
    try:
        for game in range(3):
            state, ai, decisions, reused = ge.State(), 1 if game % 2 == 0 else -1, 0, 0
            while not state.winner:
                if state.current_player == ai:
                    response = rpc(page, type="search", state=state.serialize(), budgetMs=60)
                    assert response["action"] in state.legal_actions()
                    reused += response["stats"]["reusedVisits"] > 0
                    decisions += 1
                    state.apply(response["action"])
                    if not state.winner and state.current_player != ai:
                        page.evaluate("(state) => send({type:'ponder', state})", state.serialize())
                else:
                    state.apply(rng.choice(state.legal_actions()))
                assert state.ply < 400
            assert state.winner == ai
            assert reused
    finally:
        page.close()


def test_placeholder_model_swaps_in_through_config(browser, versioned_site_url, tmp_path):
    web = tmp_path / "web"
    subprocess.run([sys.executable, "-m", "ops.export_onnx", "--kind", "hex", "--width", "32", "--blocks", "2", "--tokens",
                    "--output", str(web / "models/placeholder.onnx")], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "ops.web_model", "placeholder", "--title", "Placeholder net", "--cpuct", "2.0",
                    "--fpu", "0.2", "--web", str(web)], cwd=ROOT, check=True)
    config = json.loads((web / "config.json").read_text())
    (web / "config.json").write_text(json.dumps({**config, "model": "placeholder"}))
    page = browser.new_page(viewport={"width": 390, "height": 844})
    page.set_default_timeout(60_000)
    try:
        page.goto(versioned_site_url, wait_until="domcontentloaded")
        page.locator("#mode-select").select_option("ai")
        page.locator("#difficulty").select_option("fast")
        page.locator("#hints").check()
        page.locator('.ray-hit[data-ray-id="1"]').click()
        page.wait_for_function("document.querySelector('#move-count').textContent === 'MOVE 02'")
        assert page.locator("#ai-status").text_content() == "Placeholder net ready"
        assert "pos/s" in page.locator("#think-readout").text_content()
        # Pondering on the human's turn feeds the candidate-move overlay.
        page.locator("#hint-layer .hint-label").first.wait_for()
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.locator("#hints").uncheck()
        assert page.locator("#hint-layer .hint-label").count() == 0
    finally:
        page.close()


def test_kernel_checkpoint_of_another_shape_swaps_in_with_the_tool(browser, versioned_site_url, tmp_path):
    import torch
    from training import net
    web = tmp_path / "web"
    config = {"kind": "former", "width": 64, "blocks": 2, "heads": 2, "ffmul": 3}
    torch.manual_seed(0)
    model = net.build(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    torch.save({"model": model.state_dict(), "config": config}, tmp_path / "other.pt")
    subprocess.run([sys.executable, "-m", "ops.web_model", "other", "--checkpoint", str(tmp_path / "other.pt"), "--title", "Other net",
                    "--cpuct", "2.0", "--fpu", "0.2", "--web", str(web)], cwd=ROOT, check=True)
    meta = json.loads((web / "models/other.json").read_text())
    assert meta["format"] == "kernel" and meta["arch"] == {"width": 64, "blocks": 2, "heads": 2, "ff": 192, "relu": False, "int16": False}
    assert (meta["cpuct"], meta["fpu"]) == (2.0, 0.2)
    (web / "config.json").write_text(json.dumps({**json.loads((web / "config.json").read_text()), "model": "other"}))
    page = browser.new_page()
    page.set_default_timeout(60_000)
    try:
        page.goto(versioned_site_url, wait_until="domcontentloaded")
        page.locator("#engine-status").get_by_text("Ready to play").wait_for()
        page.locator("#mode-select").select_option("ai")
        page.locator("#difficulty").select_option("fast")
        page.locator('.ray-hit[data-ray-id="1"]').click()
        page.wait_for_function("document.querySelector('#move-count').textContent === 'MOVE 02'")
        assert page.locator("#ai-status").text_content() == "Other net ready"
    finally:
        page.close()
