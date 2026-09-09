"""Differential check of the native pybind module and the embind WASM build."""

import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

import gipf_engine


ROOT = Path(__file__).resolve().parents[1]
WASM_JS = ROOT / "web" / "gipf_engine.js"


def test_wasm_matches_native_for_1000_random_state_action_pairs():
    if not WASM_JS.exists() or not shutil.which("node"):
        pytest.skip("WASM artifact or Node is not available")

    rng = random.Random(0x61F)  # fixed, portable fixture seed
    state = gipf_engine.State()
    fixture = []
    for _ in range(1000):
        if state.winner or not state.legal_actions():
            state = gipf_engine.State()
            reset = True
        else:
            reset = False
        action = rng.choice(state.legal_actions())
        state.apply(action)
        fixture.append({"reset": reset, "action": action, "expected": state.serialize()})

    node_program = r'''
const createGipfEngine = require(process.argv[1]);
const fixture = JSON.parse(require('fs').readFileSync(0, 'utf8'));
createGipfEngine().then(Module => {
  let state = new Module.State();
  for (let i = 0; i < fixture.length; ++i) {
    const item = fixture[i];
    if (item.reset) { state.delete(); state = new Module.State(); }
    state.apply(item.action);
    const actual = state.serialize();
    if (JSON.stringify(actual) !== JSON.stringify(item.expected)) {
      console.error(`mismatch at ${i}: ${JSON.stringify(actual)} != ${JSON.stringify(item.expected)}`);
      process.exit(1);
    }
  }
  state.delete();
});
'''
    completed = subprocess.run(
        ["node", "-e", node_program, str(WASM_JS)],
        input=json.dumps(fixture),
        text=True,
        cwd=ROOT,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
