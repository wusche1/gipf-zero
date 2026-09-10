/*
 * Small client for the browser-local AI worker.
 *
 * The worker is deliberately lazy: local hotseat can start as soon as the
 * rules engine is ready, while AI mode pays the model startup cost only when
 * it actually needs a move.  Cancelling terminates the worker so a stale
 * search can never post a move into a newer position.
 */

function abortError(reason = 'cancelled') {
  const error = new Error(reason);
  error.name = 'AbortError';
  return error;
}

export class LocalAiClient {
  constructor({ onStatus = () => {}, backend = 'wasm', initTimeoutMs = 60_000, searchTimeoutExtraMs = 10_000 } = {}) {
    this.onStatus = onStatus;
    this.backend = backend;
    this.initTimeoutMs = initTimeoutMs;
    this.searchTimeoutExtraMs = searchTimeoutExtraMs;
    this.worker = null;
    this.readyPromise = null;
    this.pending = new Map();
    this.nextId = 1;
    this.generation = 0;
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
      this.onStatus('ready', message);
      return message;
    }).catch((error) => {
      if (this.generation === generation) {
        this.onStatus('error', error);
        this.#discardWorker(generation);
      }
      throw error;
    });
    return this.readyPromise;
  }

  async search(state, { budgetMs = 250, simulations } = {}) {
    const generation = this.generation;
    const ready = await this.ready();
    if (generation !== this.generation) throw abortError();
    const message = { type: 'search', state, budgetMs };
    if (simulations != null) message.simulations = simulations;
    const result = await this.#request(this.worker, generation, message, Math.max(1, Number(budgetMs) || 250) + this.searchTimeoutExtraMs);
    if (result?.error) throw new Error(String(result.error));
    return { ...result, model: result.model || ready.model, backend: result.backend || ready.backend };
  }

  cancel(reason = 'cancelled') {
    const oldGeneration = this.generation;
    this.generation += 1;
    for (const [id, pending] of this.pending) {
      if (pending.generation === oldGeneration) {
        clearTimeout(pending.timer);
        pending.reject(abortError(reason));
      }
      this.pending.delete(id);
    }
    this.#discardWorker(oldGeneration);
    this.onStatus('cancelled');
  }

  close() {
    this.cancel('closed');
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
      this.pending.delete(message.id);
      clearTimeout(pending.timer);
      if (message.error) pending.reject(new Error(String(message.error)));
      else pending.resolve(message);
    });
    worker.addEventListener('error', (event) => {
      if (generation !== this.generation) return;
      const error = new Error(event.message || 'Local AI worker failed');
      for (const [id, pending] of this.pending) {
        if (pending.generation === generation) {
          clearTimeout(pending.timer);
          pending.reject(error);
          this.pending.delete(id);
        }
      }
      this.onStatus('error', error);
      this.#discardWorker(generation);
    });
    return worker;
  }

  #request(worker, generation, message, timeoutMs) {
    if (!worker || generation !== this.generation) return Promise.reject(abortError());
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (!this.pending.has(id) || generation !== this.generation) return;
        const error = new Error(`${message.type === 'init' ? 'Local AI startup' : 'Local AI search'} timed out`);
        error.name = 'TimeoutError';
        this.#invalidateGeneration(generation, error);
        this.onStatus('error', error);
      }, Math.max(1, timeoutMs));
      this.pending.set(id, { resolve, reject, generation, timer });
      try {
        worker.postMessage({ ...message, id });
      } catch (error) {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(error);
      }
    });
  }

  #invalidateGeneration(generation, reason) {
    if (generation !== this.generation) return;
    this.generation += 1;
    for (const [id, pending] of this.pending) {
      if (pending.generation === generation) {
        clearTimeout(pending.timer);
        pending.reject(reason);
      }
      this.pending.delete(id);
    }
    const worker = this.worker;
    this.worker = null;
    this.readyPromise = null;
    worker?.terminate();
  }

  #discardWorker(generation) {
    if (generation !== this.generation && this.worker == null) return;
    const worker = this.worker;
    this.worker = null;
    this.readyPromise = null;
    worker?.terminate();
  }
}

export function createLocalAiClient(options) {
  return new LocalAiClient(options);
}
