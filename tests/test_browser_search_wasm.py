"""Browser NativeForest wrapper agrees with the native C++ tree on a capture state."""
import hashlib
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import gipf_engine as ge


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = json.loads((ROOT / "web/config.json").read_text())["model"]
META = json.loads((ROOT / f"web/models/{DEFAULT}.json").read_text())
CHECKPOINT = os.environ.get("GIPF_KERNEL_CHECKPOINT", "/workspace/tmp/gipf-work/snap/former-L1.pt")


def capture_state():
    state, rng = ge.State(), random.Random(89)
    while state.phase != "capture":
        state.apply(rng.choice(state.legal_actions()))
    return state


def terminal_capture_state():
    """The only capture ends the game by removing the losing final double."""
    return {
        "board": [-1, -1, -1, 0, 1, 0, 1, 2, 0, -1, -1, -1, 0, -1, -1, -1, 1,
                  -2, -1, 0, 2, 0, -1, 1, 1, 0, 1, -1, 1, 1, 2, 0, 1, 1, 1, -1, -2],
        "reserves": [1, 0], "captured": [0, 1], "current_player": 1,
        "turn_player": 1, "phase": "capture", "winner": 0, "ply": 35,
    }


def test_browser_search_zero_logits_matches_native_capture_backup():
    wasm = ROOT / "web" / "gipf_engine.js"
    if not wasm.exists() or not shutil.which("node"):
        pytest.skip("WASM artifact or Node unavailable")
    state = capture_state()
    expected_features = ge.encode_batch([state]).reshape(-1)
    root = ge.NativeNode(state)
    forest = ge.NativeForest([root])
    zeros = np.zeros((1, 2730), np.float32)
    assert forest.expand_unexpanded_root_pending() == 1
    forest.finish(zeros, np.zeros(1, np.float32))
    for _ in range(8):
        if forest.select_pending(1.5):
            forest.finish(zeros, np.zeros(1, np.float32))
    program = r'''
const create = require(process.argv[1]);
const state = JSON.parse(require('fs').readFileSync(0, 'utf8'));
create().then(Module => {
  const search = new Module.BrowserSearch(JSON.stringify(state), 8);
  if (!search.prepare()) throw new Error('initial prepare failed');
  let features = search.features();
  if (features.length !== 441) throw new Error('expected [1,9,7,7] initial features');
  const initialFeatures = Array.from(features);
  search.complete(new Float32Array(2730), 0);
  while (search.step(1.5)) {
    features = search.features();
    if (features.length !== 441) throw new Error('expected one pending leaf');
    search.complete(new Float32Array(2730), 0);
  }
  console.log(JSON.stringify({result: search.result(), initialFeatures})); search.delete();
}).catch(error => { console.error(error); process.exit(1); });
'''
    completed = subprocess.run(["node", "-e", program, "./web/gipf_engine.js"], input=json.dumps(state.serialize()),
                               text=True, cwd=ROOT, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    response = json.loads(completed.stdout)
    browser = response["result"]
    np.testing.assert_array_equal(np.asarray(response["initialFeatures"], np.float32), expected_features)
    assert browser["actions"] == list(root.actions)
    assert browser["visits"] == list(root.visits)
    assert browser["action"] == int(root.actions[int(np.argmax(root.visits))])


def test_browser_search_terminal_capture_returns_after_one_unbounded_step():
    """A solved terminal child cannot monopolize the worker event loop."""
    wasm = ROOT / "web" / "gipf_engine.js"
    if not wasm.exists() or not shutil.which("node"):
        pytest.skip("WASM artifact or Node unavailable")
    program = r'''
const create = require(process.argv[1]);
const state = JSON.parse(require('fs').readFileSync(0, 'utf8'));
create().then(Module => {
  const search = new Module.BrowserSearch(JSON.stringify(state));
  if (!search.prepare()) throw new Error('initial prepare failed');
  search.complete(new Float32Array(2730), 0);
  const pending = search.step(1.5);
  const result = search.result(); search.delete();
  console.log(JSON.stringify({pending, simulations: result.simulations, terminal: result.terminal}));
}).catch(error => { console.error(error); process.exit(1); });
'''
    completed = subprocess.run(["node", "-e", program, "./web/gipf_engine.js"],
                               input=json.dumps(terminal_capture_state()), text=True, cwd=ROOT,
                               capture_output=True, timeout=2)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"pending": False, "simulations": 1, "terminal": False}


def test_reroot_keeps_child_statistics_and_rejects_unrelated_states():
    wasm = ROOT / "web" / "gipf_engine.js"
    if not wasm.exists() or not shutil.which("node"):
        pytest.skip("WASM artifact or Node unavailable")
    program = r"""
const create = require(process.argv[1]);
create().then(Module => {
  const run = (search, n) => { for (let i = 0; i < n; i++) if (search.step(1.5)) search.complete(new Float32Array(2730), 0); };
  const start = new Module.State();
  const search = new Module.BrowserSearch(JSON.stringify(start.serialize()));
  search.set_fpu(0.2);
  search.prepare(); search.complete(new Float32Array(2730), 0);
  run(search, 400);
  const before = search.result();
  const child = start.clone(); child.apply(before.action);
  const grandchild = child.clone(); grandchild.apply(before.pv[1]);
  const far = grandchild.clone(); for (let i = 0; i < 2; i++) far.apply(far.legal_actions().get(0));
  const out = {before, margin: search.margin(), choices: search.choices()};
  out.farFound = search.reroot(JSON.stringify(far.serialize()));
  out.childFound = search.reroot(JSON.stringify(child.serialize()));
  out.child = search.result(); out.childVisits = search.root_visits();
  out.grandchildFound = search.reroot(JSON.stringify(grandchild.serialize()));
  out.grandchildVisits = search.root_visits();
  run(search, 50);
  out.after = search.result();
  const legal = grandchild.legal_actions(); out.legal = Array.from({length: legal.size()}, (_, i) => legal.get(i));
  out.undoFound = search.reroot(JSON.stringify(start.serialize()));
  console.log(JSON.stringify(out)); search.delete();
}).catch(error => { console.error(error); process.exit(1); });
"""
    completed = subprocess.run(["node", "-e", program, "./web/gipf_engine.js"], text=True, cwd=ROOT, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    out = json.loads(completed.stdout)
    before = out["before"]
    best = max(before["visits"])
    assert before["root_visits"] == 400 and out["choices"] == len(before["actions"]) == 42
    assert out["margin"] == best - sorted(before["visits"])[-2]
    assert before["pv"][0] == before["action"] and before["actor"] == 1
    assert before["white_value"] == before["value"] and -1 <= before["value"] <= 1
    assert not out["farFound"] and not out["undoFound"]
    # The first visit to a child expands it; every later one passes through.
    assert out["childFound"] and out["childVisits"] == best - 1
    assert out["child"]["simulations"] == 0 and out["child"]["actor"] == -1
    assert out["child"]["white_value"] == -out["child"]["value"]
    assert out["child"]["pv"] == before["pv"][1:]
    assert out["grandchildFound"] and out["grandchildVisits"] == max(out["child"]["visits"]) - 1
    assert out["after"]["root_visits"] == out["grandchildVisits"] + 50
    assert out["after"]["actions"] == out["legal"] and out["after"]["action"] in out["legal"]


def test_kernel_search_matches_pytorch_search_on_the_shipped_weights():
    """think() inside WASM against the Python search driving the PyTorch model."""
    if META["format"] != "kernel" or not Path(CHECKPOINT).exists() or not shutil.which("node"):
        pytest.skip("kernel model, its checkpoint (GIPF_KERNEL_CHECKPOINT) or Node unavailable")
    import torch
    from ops.dump_former import dump
    from ops.web_model import positions
    from training import net
    from training.selfplay import search
    model = net.load(CHECKPOINT)[0]
    if hashlib.sha256(dump(model).tobytes()).hexdigest() != META["sha256"]:
        pytest.skip("GIPF_KERNEL_CHECKPOINT is not the checkpoint of the shipped weights")
    torch.set_num_threads(1)
    states = [ge.State(), capture_state(), *positions(6, seed=5)]
    program = r"""
const [engine, weights, meta] = process.argv.slice(1), fs = require('fs');
require(engine)().then((m) => {
  const w = fs.readFileSync(weights), {arch, cpuct, fpu} = JSON.parse(meta);
  if (!m.load_kernel(new Float32Array(w.buffer, w.byteOffset, w.length / 4), arch)) throw new Error('bad weights');
  console.log(JSON.stringify(JSON.parse(fs.readFileSync(0, 'utf8')).map((state) => {
    const search = new m.BrowserSearch(JSON.stringify(state)); search.set_fpu(fpu); search.set_cpuct(cpuct);
    let slices = 0;
    for (let n = 0; n < 256; slices++) n += search.think(256 - n, 20);
    return {...search.result(), slices};
  })));
}).catch(error => { console.error(error); process.exit(1); });
"""
    completed = subprocess.run(["node", "-e", program, "./web/gipf_engine.js", f"web/models/{META['path']}", json.dumps(META)],
                               input=json.dumps([state.serialize() for state in states]), text=True, cwd=ROOT, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    exact = 0
    for state, browser in zip(states, json.loads(completed.stdout)):
        root = ge.NativeNode(state)
        search(model, [root], 256, "cpu", META["cpuct"], META["fpu"])
        visits = np.asarray(root.visits)
        assert browser["actions"] == list(root.actions) and browser["root_visits"] == 256 == browser["simulations"]
        assert browser["slices"] > 1
        assert browser["action"] == int(root.actions[int(visits.argmax())])
        assert abs(browser["value"] - sum(root.values) / 256) < 1e-3
        # Float rounding may flip a near-tie between two children, nothing more.
        assert np.abs(visits - browser["visits"]).sum() <= 8
        exact += list(visits) == browser["visits"]
    assert exact >= len(states) - 1
