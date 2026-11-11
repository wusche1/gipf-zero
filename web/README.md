# Static GIPF with a local AI

Serve this directory over HTTP or HTTPS (for example `python -m http.server
8000 --directory web`). Both local hotseat and the learned AI run in the browser.
There is no inference endpoint, account token, CDN runtime, or Hugging Face fetch.
Deleting the training instance does not affect the GitHub Pages deployment.
Opening `index.html` via `file://` is not supported by browser worker/fetch rules.

`game.js` adapts the authoritative C++ rules engine compiled to WebAssembly.
`local-ai.js` starts `ai-worker.js` lazily. `config.json` names the model
(`"model": "champion"`); the worker reads `models/<name>.json` for the ONNX file,
its SHA-256, `cpuct`, `fpu` (negative = off), thinking budgets and ponder caps, and
runs it with self-hosted ONNX Runtime Web. The native PUCT tree and actor-aware
backups are also compiled to WASM. All inference uses one CPU thread so normal
Pages hosting needs no cross-origin isolation headers. Search runs off the UI thread.

The worker keeps one search tree alive. A request for a position below the
current root (the machine's move, your reply, capture choices) re-roots onto that
subtree; anything else (undo, new game, side switch) starts a fresh tree. While it
is your turn the worker ponders the same tree until the model's cap, a hidden tab,
or your move stops it. Strength levels are Fast, Normal (default) and Strong
(0.3 s, 1 s, 3 s by default). A search ends early when the runner-up cannot catch
the most-visited move in the time left, returns at once when only one action is
legal, and gives capture choices a quarter budget. The move played is the
most-visited root action. The board shows the search's win estimate, simulations,
positions per second and principal-variation depth; candidate moves with their
visit share are an optional overlay. Cancelling stops the search without
discarding the worker; initialization and searches still have watchdogs, and a
failed model download can be retried without discarding the game.

To ship another network (same I/O contract):

    python -m ops.export_onnx --checkpoint CKPT --tokens --output web/models/NAME.onnx
    python -m ops.web_model NAME --title "Shown name" --cpuct 1.5 --fpu -1
    # then set "model": "NAME" in web/config.json

`benchmark.html` measures inference and full search on the visitor's actual device.
WebGPU is experimental: the worker verifies all 16 original reference positions
before allowing GPU play, and rejects an incorrect or unavailable backend. The
production default is the verified CPU/WASM path.

## Reproduce the artifacts

- `ops/export_browser_model.py`: champion export, numerical parity, native benchmarks.
- `benchmarks/browser_inference.py`: browser WASM/WebGPU inference and search timing.
- `benchmarks/browser_play.py`: native search parity and complete browser games.
- `engine/build_wasm.sh`: rules and native-tree search, compiled with SIMD.
- `ops/version_web_assets.py`: consistent cache versions for Pages deployment.

Runtime files under `vendor/onnxruntime/` are pinned to `onnxruntime-web@1.29.0`.
Its manifest records SHA-256 hashes; upstream MIT license and third-party notices
are included. Install that exact npm package to reproduce the copied `dist/`
JavaScript, `.mjs`, and `.wasm` files. Only the selected backend's files are fetched.
