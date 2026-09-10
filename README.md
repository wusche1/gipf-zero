# GIPF Zero

Play **standard GIPF** in your browser: two people sharing a screen, or a public AI opponent. The game engine is shared between native training and browser WebAssembly. The learned opponent is trained from random initialization through self-play and policy/value Monte Carlo tree search.

**[Play the game](https://wusche1.github.io/gipf-zero/)** · [Official GIPF rules](https://www.gipf.com/gipf/rules/complete_rules.html)

## Original deadline result

The completed 9 September 2026 selection run chose a square 3x3 **ResNet-32
CNN** trained for 43,869 self-play games. It won its 80-game CPU, one-thread,
50 ms-per-decision promotion duel 51--29 (Wilson lower bound 0.528), and passed
the greedy gate 40--0. The later MLP-256 checkpoint trained for 103,668 games
tied the CNN 40--40 and did not clear that promotion gate.

The frozen CNN then scored 200--0 against random, 200--0 against greedy, 79--1
against MCTS, and 80--0 against depth-6 minimax, all with no cutoffs. The first
two are fixed-128-simulation GPU evaluations; MCTS and minimax use CPU, batch 1,
and equal 50 ms decision budgets. [Raw reports and conditions](reports/RESULTS.md)
include checkpoint hashes, seeds, actual search measurements, and confidence
intervals.

## Overnight comparison and final model

The follow-up compared MLP256, square CNN32, hex CNN32, and two query
transformers (256-wide/two-layer and 164-wide/five-layer), with two seeds and
600 seconds of dedicated training per seed. All 20 equal-CPU and 20
fixed-simulation pair/seed comparisons completed, including three bounded
fixed-simulation retries.

Square CNN led the primary round robin with 80 wins in 96 games; hex CNN had
74. Their extra direct match was 42–38 for square, an inconclusive result.
The predeclared search-throughput tie rule selected hex for continued training.
It reached **93,916 games**, then scored **38–42** against the existing champion
with no cutoffs. It did not qualify for promotion, so the **43,869-game square
CNN remains the public model**. The query transformers underperformed under
these budgets; the different output heads mean this is not a pure architecture
ablation.

[Full comparison, uncertainty, and raw results](reports/overnight/20260909T235853Z/RESULTS.md)
· [Download the verified public model and comparison checkpoints](https://huggingface.co/wuschelschulz/gipf-zero)

The website supports local two-player play entirely in your browser. Remote AI
requires the inference host to remain online. Automated wins against these
opponents do not establish a human or expert rating.

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
- [Original deadline results and raw reports](reports/RESULTS.md)
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
