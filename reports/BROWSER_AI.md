# Browser AI migration — 10 September 2026

The published game is a static GitHub Pages application. Its learned opponent
runs on the visitor's device; the training/inference instance is not required.
It uses the same 43,869-game square CNN champion, exported as FP32 ONNX, plus
the existing native PUCT search compiled to WebAssembly with SIMD. No model
quantization, replacement policy, random fallback, or retraining was used.

## Model and hosting

- Source served checkpoint SHA-256:
  `d5a7e3bda2966a79d22cf34f131976907260e679683d0e2680d935312f41c2f6`.
- Default ONNX SHA-256:
  `3d657aea6d89f73e84f189f919c17bdd8ee3f2d69d5a1ef84c6f722ee0a17a73`.
- ONNX model: 4,664,337 bytes, FP32, 1,163,523 network parameters.
- All runtime, model, worker, and engine assets are served from the Pages site.
  The app makes no inference-server or Hugging Face requests.
- ONNX Runtime Web 1.29.0, single-thread CPU/WASM by default. No special hosting
  headers or SharedArrayBuffer requirement. AI loads lazily when needed.
- CPU thinking budgets: 0.5 seconds, 1.5 seconds (default), or 5 seconds; at most
  10,000 simulations. Cold model/runtime initialization is separate.
- The default runtime WASM is about 14 MB before HTTP compression, in addition
  to the model. First-load time depends on the connection; later loads can cache.

## Measured speed

Same host, one thread each, actual production PyTorch `choose_action` versus
browser WASM. Each position had one excluded warmup, then five fresh search trees
with exactly 128 simulations. Browser figures include the worker round trip.
Model downloads and initialization are excluded from these warmed search times.

| Position | Native server path median | Browser median |
|---|---:|---:|
| Opening push | 84.8 ms | 81.9 ms |
| Later push | 74.4 ms | 77.9 ms |
| Capture A | 83.8 ms | 76.6 ms |
| Capture B | 84.9 ms | 74.1 ms |

The median browser/native ratio across positions was 0.94. Given ordinary timing
variation, treat this as roughly comparable speed, not a demonstrated browser
speed advantage. Earlier cold/noisier runs were slower. Batch-one forward medians
were 0.496 ms for PyTorch and 0.500 ms for browser WASM. These measurements do not
predict laptop or phone performance. [Raw samples and protocol](browser-performance-final.json)
include runtime versions and model hashes.

Native ONNX Runtime CPU was also measured at about 0.182 ms per forward. That is
an alternative native runtime, not the existing server path used for the table.
Static-batch and arithmetic-normalization exports preserved outputs but did not
establish a consistent browser speed improvement; the dynamic export stays default.

## Correctness and gameplay

- All 16 reachable reference positions (8 push, 8 capture) matched native chosen
  actions **and every root visit count**, with 128 simulations. This includes the
  actor-aware capture backup semantics.
- Export numerical errors were below 5e-6 absolute in the tested fixtures.
- Browser AI won 8/8 games against random and 8/8 against greedy, alternating
  colours, with 64 simulations per decision. All games ended normally: no illegal
  moves, timeouts, unfinished games, or move-limit cutoffs. Capture and keeping
  double pieces were exercised. These small baseline checks are not expert ratings.
- Full games blocked every request outside the static origin.
- Real UI checks passed in Chromium desktop/mobile layouts, Firefox 153.0, and
  WebKit 26.5: model initialization, human move, AI reply, and undo. Chromium also
  covered side changes, AI opening, deep-search cancellation, and stale replies.
- A model-download failure can be retried without losing the current board.
- Search steps return control even for terminal leaves. Worker initialization and
  search watchdogs prevent an indefinitely stuck UI.

[Native/browser parity and complete games](browser-play.json) ·
[UI/browser checks](browser-ui.json).

## WebGPU investigation

This instance exposes no normal hardware WebGPU adapter. We additionally tested
Chromium's SwiftShader software adapter. It initialized, but model outputs failed
reference checks. Static shape, explicit arithmetic normalization, and disabling
ORT graph optimizations did not fix that result. Therefore those outputs are
never used for play, and software-GPU timing is not presented as hardware speed.

The default is CPU/WASM. The experimental GPU path checks all 16 reference
positions before accepting a GPU backend and rejects incorrect/nonfinite outputs.
A real laptop GPU remains unmeasured here. The self-contained
[device benchmark](https://wusche1.github.io/gipf-zero/benchmark.html) lets a visitor
check their actual browser. [GPU diagnostic evidence](browser-webgpu-diagnosis.json).

## Reproduction

- `ops/export_browser_model.py` exports/verifies the original champion.
- `benchmarks/browser_performance.py` repeats the matched CPU/browser benchmark.
- `benchmarks/browser_play.py` checks complete games and native search parity.
- `benchmarks/browser_inference.py` compares model/runtime variants.
- `tests/test_browser_local_ai.py` exercises real browser AI with no external host.
- `tests/test_browser_search_wasm.py` checks feature/backup parity and terminal-step liveness.
- `tests/test_browser_offline.py` exercises cached offline play.

The repository retains the native training pipeline for reproducibility. All
training processes were stopped; browser inference does not consume OpenAI credits.

## Public deployment verification

The public Pages deployment at revision `77e1119` was tested after stopping both
`gipf_api` and its public tunnel. A fresh Chromium context loaded the actual public
site with every non-Pages origin blocked, played a human move and a champion reply,
and undid the turn. The same context then switched completely offline, reloaded
the benchmark page and game, and played another AI reply. No page errors occurred.
Only optional Google Fonts requests were blocked; no inference endpoint was used.
[Public test evidence](browser-public-deployment.json).

Offline play requires first loading the AI while online and browser cache storage
remaining available. A brand-new visitor still needs internet access to GitHub
Pages to download the application and model. The GPU instance is unnecessary in
both cases. All 89 local tests passed, including native rules/search and real
browser AI, cancellation, retry, and versioned offline-cache checks.
