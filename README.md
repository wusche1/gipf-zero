# GIPF Zero

Play **standard GIPF** in your browser: two people sharing a screen, or a public AI opponent. The game engine is shared between native training and browser WebAssembly. The learned opponent is trained from random initialization through self-play and policy/value Monte Carlo tree search.

**[Play the game](https://wusche1.github.io/gipf-zero/)** · [Official GIPF rules](https://www.gipf.com/gipf/rules/complete_rules.html)

## Current experiment

This is an active, time-bounded experiment on one NVIDIA A100 40 GB. Baselines, short candidate runs, evaluation, and checkpoint promotion precede the main training run. Published results distinguish wins, losses, and game-length cutoffs. Beating random is a smoke test, not evidence of expert play.

The website starts with a fixed heuristic lookahead opponent. A learned checkpoint is promoted only after evaluation. The current opponent is identified by the API and interface. Local two-player mode runs entirely in your browser; remote AI requires the inference host to remain online.

## Run locally

Python 3.12 and a C++17 compiler:

```sh
python -m pip install pybind11 setuptools numpy torch fastapi uvicorn pytest
python setup.py build_ext --inplace
python -m pytest -q
python -m http.server 8000 --directory web
```

Open `http://localhost:8000` for local play. For a local inference server:

```sh
python -m uvicorn server.app:app --host 127.0.0.1 --port 17100
```

Set the API base URL in `web/config.json`. The public service permits the GitHub Pages origin; configure CORS for your own frontend when hosting elsewhere.

## Train and evaluate

```sh
python -m training.train --run runs/example --kind mlp --width 128 --blocks 2 \
  --games 64 --simulations 64 --seconds 900
python -m training.evaluate --checkpoint runs/example/latest.pt \
  --opponent greedy --games 200 --simulations 128 --output reports/example.json
```

Training emits JSONL metrics and a heartbeat, writes atomic checkpoints, and saves its replay buffer for recovery. `--resume` reloads a checkpoint; `--seconds` and `--deadline` bound a run. SIGTERM requests a clean checkpoint and exit. Evaluation uses paired colors and seeded opening diversity; unfinished/cutoff games are reported separately.

- [Rules and action contract](engine/CONTRACT.md)
- [Fixed baseline definitions](baselines/README.md)
- `training/`: network, symmetry augmentation, batched PUCT, self-play and evaluation
- `reports/`: measured experiment results
- `ops/`: managed service and experiment launch scripts

## Rebuild the browser engine

Install Emscripten, activate its environment, then run:

```sh
engine/build_wasm.sh
```

The generated JavaScript and WASM are committed so playing the game does not require a build toolchain. Differential tests compare native and browser state transitions.

## Credits

GIPF was designed by Kris Burm. This is an independent, unofficial implementation, not affiliated with or endorsed by the game's designer or publisher. GIPF and associated marks belong to their respective owners. Board graphics and interface assets here are original; the official rules are linked as the reference.
