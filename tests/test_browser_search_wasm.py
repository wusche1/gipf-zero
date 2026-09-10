"""Browser NativeForest wrapper agrees with the native C++ tree on a capture state."""
import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import gipf_engine as ge


ROOT = Path(__file__).resolve().parents[1]


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
    """A solved terminal child cannot monopolize the classic-worker event loop."""
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
