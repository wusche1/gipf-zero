/*
 * Small client for the browser-local AI worker.
 *
 * The worker is deliberately lazy: local hotseat can start as soon as the
 * rules engine is ready, while AI mode pays the model startup cost only when
 * it actually needs a move.  The worker keeps its search tree between
 * requests, so cancelling asks it to stop rather than terminating it; a stale
 * reply is dropped here because its request is no longer pending.
 */

function abortError(reason = 'cancelled') {
  const error = new Error(reason);
  error.name = 'AbortError';
  return error;
}

export class LocalAiClient {
  constructor({ onStatus = () => {}, backend = 'wasm', initTimeoutMs = 60_000, searchTimeoutMs = 30_000 } = {}) {
    this.onStatus = onStatus;
    this.backend = backend;
    this.initTimeoutMs = initTimeoutMs;
    this.searchTimeoutMs = searchTimeoutMs;
    this.worker = null;
    this.readyPromise = null;
    this.isReady = false;
    this.pending = new Map();
    this.nextId = 1;
    this.generation = 0;
    this.epoch = 0;
  }

  async ready() {
    if (this.readyPromise) return this.readyPromise;
    const generation = this.generation;
    const worker = this.#startWorker(generation);
    this.onStatus('loading');
    const init = { type: 'init' };
    if (this.backend) init.backend = this.backend;
    this.readyPromise = this.#request(worker, generation, init, this.initTimeoutMs).then((message) => {
      if (!message.ready) throw new Error('Local AI worker did not become ready');
      this.isReady = true;
      this.onStatus('ready', message);
      return message;
    }).catch((error) => {
      if (this.generation === generation) {
        this.onStatus('error', error);
        this.#discardWorker();
      }
      throw error;
    });
    return this.readyPromise;
  }

  search(state, options) { return this.#run('search', state, options, this.searchTimeoutMs); }

  // Background thinking on the opponent's turn. It ends by itself at the
  // model's cap, or as soon as any later search/ponder/stop arrives.
  ponder(state, options) { return this.#run('ponder', state, options, 0); }

  stopPonder() { this.worker?.postMessage({ type: 'stop', ponder: true, id: this.nextId++ }); }

  async #run(type, state, { level, budgetMs, simulations, onProgress } = {}, timeoutMs) {
    const epoch = this.epoch;
    const ready = await this.ready();
    if (epoch !== this.epoch) throw abortError();
    const message = { type, state, level, budgetMs, simulations, progress: Boolean(onProgress) };
    const result = await this.#request(this.worker, this.generation, message, timeoutMs, onProgress);
    return { ...result, model: result.model || ready.model, backend: result.backend || ready.backend };
  }

  cancel(reason = 'cancelled') {
    this.epoch += 1;
    for (const [id, pending] of this.pending) {
      if (pending.init) continue;
      clearTimeout(pending.timer);
      pending.reject(abortError(reason));
      this.pending.delete(id);
    }
    this.worker?.postMessage({ type: 'stop', id: this.nextId++ });
    this.onStatus('cancelled');
  }

  close() {
    this.#invalidate(abortError('closed'));
    this.onStatus('closed');
  }

  #startWorker(generation) {
    if (this.worker) return this.worker;
    const worker = new Worker(new URL('./ai-worker.js', import.meta.url));
    this.worker = worker;
    worker.addEventListener('message', (event) => {
      if (generation !== this.generation) return;
      const message = event.data || {};
      const pending = this.pending.get(message.id);
      if (!pending) return;
      if (message.progress) { pending.onProgress?.(message.stats); return; }
      this.pending.delete(message.id);
      clearTimeout(pending.timer);
      if (message.error) pending.reject(new Error(String(message.error)));
      else pending.resolve(message);
    });
    worker.addEventListener('error', (event) => {
      if (generation !== this.generation) return;
      const error = new Error(event.message || 'Local AI worker failed');
      this.#invalidate(error);
      this.onStatus('error', error);
    });
    return worker;
  }

  #request(worker, generation, message, timeoutMs, onProgress) {
    if (!worker || generation !== this.generation) return Promise.reject(abortError());
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = timeoutMs ? setTimeout(() => {
        if (!this.pending.has(id) || generation !== this.generation) return;
        const error = new Error(`${message.type === 'init' ? 'Local AI startup' : 'Local AI search'} timed out`);
        error.name = 'TimeoutError';
        this.#invalidate(error);
        this.onStatus('error', error);
      }, timeoutMs) : null;
      this.pending.set(id, { resolve, reject, timer, onProgress, init: message.type === 'init' });
      worker.postMessage({ ...message, id });
    });
  }

  #invalidate(reason) {
    this.generation += 1;
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(reason);
    }
    this.pending.clear();
    this.#discardWorker();
  }

  #discardWorker() {
    const worker = this.worker;
    this.worker = null;
    this.readyPromise = null;
    this.isReady = false;
    worker?.terminate();
  }
}

export function createLocalAiClient(options) {
  return new LocalAiClient(options);
}
