# Static GIPF with a local AI

Serve this directory over HTTP or HTTPS (for example `python -m http.server
8000 --directory web`). Both local hotseat and the learned AI run in the browser.
There is no inference endpoint, account token, CDN runtime, or Hugging Face fetch.
Deleting the training instance does not affect the GitHub Pages deployment.
Opening `index.html` via `file://` is not supported by browser worker/fetch rules.

`game.js` adapts the authoritative C++ rules engine compiled to WebAssembly.
`local-ai.js` starts `ai-worker.js` lazily and terminates it on cancellation.
The worker loads the exact promoted champion from `models/champion.onnx` using
self-hosted ONNX Runtime Web. The native PUCT tree and actor-aware backups are
also compiled to WASM. All inference uses one CPU thread so normal Pages hosting
needs no cross-origin isolation headers. Search runs off the UI thread.

AI tempos are 0.5, 1.5 (default), and 5 seconds. The simulation limit is 10,000;
slower devices complete fewer simulations within the selected time. Initialization
has a 60-second watchdog, and searches have a separate deadline/watchdog. A failed
model download can be retried without discarding the game.

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
