/* Offline shell and same-origin runtime cache for the static GIPF site. */
'use strict';

const VERSION = '__ASSET_VERSION__';
const SCOPE_KEY = new URL('./', self.location).pathname.replace(/[^a-z0-9]+/gi, '_');
const CACHE_PREFIX = `gipf-${SCOPE_KEY}-`;
const SHELL_CACHE = `${CACHE_PREFIX}shell-${VERSION}`;
const RUNTIME_CACHE = `${CACHE_PREFIX}runtime-${VERSION}`;
const NAVIGATION_TIMEOUT_MS = 4000;
const REVISION_QUERY = VERSION.startsWith('__') ? '' : `?v=${VERSION}`;

// Keep this list small. The model and ORT files are filled into the runtime
// cache only after the worker actually uses them.
const SHELL_ASSETS = [
  './',
  './index.html',
  './styles.css',
  './app.js',
  './game.js',
  './local-ai.js',
  './offline.js',
  './service-worker.js',
  './config.json',
  './gipf_engine.js',
  './gipf_engine.wasm',
  './benchmark.html',
  './benchmark.js'
];
const REVISIONED_SHELL_ASSETS = REVISION_QUERY ? [
  `./styles.css${REVISION_QUERY}`,
  `./app.js${REVISION_QUERY}`,
  `./game.js${REVISION_QUERY}`,
  `./local-ai.js${REVISION_QUERY}`,
  `./offline.js${REVISION_QUERY}`,
  `./service-worker.js${REVISION_QUERY}`,
  `./ai-worker.js${REVISION_QUERY}`,
  `./gipf_engine.js${REVISION_QUERY}`,
  `./gipf_engine.wasm${REVISION_QUERY}`,
  `./benchmark.js${REVISION_QUERY}`
] : [];

const sameOrigin = (request) => new URL(request.url).origin === self.location.origin;
const isNavigation = (request) => request.mode === 'navigate' || request.destination === 'document';
const isConfig = (url) => url.pathname.endsWith('/config.json');
const isMetadata = (url) => url.pathname.endsWith('/champion-metadata.json');
const isStableAsset = (request, url) => {
  if (request.method !== 'GET' || !sameOrigin(request) || isConfig(url) || isMetadata(url)) return false;
  return request.destination === 'script'
    || request.destination === 'style'
    || request.destination === 'worker'
    || request.destination === 'wasm'
    || /\.(?:js|mjs|css|wasm|onnx|json)$/i.test(url.pathname)
    || url.pathname.includes('/models/')
    || url.pathname.includes('/vendor/');
};

async function fetchWithTimeout(request, timeoutMs = NAVIGATION_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(request, { signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function putBestEffort(cacheName, request, response) {
  if (!response || !response.ok) return;
  try {
    const cache = await caches.open(cacheName);
    await cache.put(request, response.clone());
  } catch {
    // Quota and private-mode failures must never turn an online response into
    // a game failure.
  }
}

async function cacheShell(cache) {
  try {
    await cache.addAll([...SHELL_ASSETS, ...REVISIONED_SHELL_ASSETS]);
    // The Pages build adds a revision query to script/style/engine references.
    // Cache those exact URLs too; stable asset lookups intentionally do not
    // ignore query strings because model and export hashes are meaningful.
    const index = await fetch(new URL('./index.html', self.location).href);
    if (index.ok) {
      const html = await index.text();
      const references = [...html.matchAll(/(?:src|href)=["']([^"']+\.(?:js|css)(?:\?[^"']+)?)['"]/gi)]
        .map((match) => new URL(match[1], self.location).href);
      await cache.addAll(references);
    }
    const benchmark = await fetch(new URL('./benchmark.html', self.location).href);
    if (benchmark.ok) {
      const html = await benchmark.text();
      const references = [...html.matchAll(/(?:src|href)=["']([^"']+\.(?:js|css)(?:\?[^"']+)?)['"]/gi)]
        .map((match) => new URL(match[1], self.location).href);
      await cache.addAll(references);
    }
  } catch {
    // A partial shell is still useful online, and runtime fetches can fill in
    // anything that was unavailable during installation.
  }
}

async function cacheFirst(request) {
  try {
    const cache = await caches.open(RUNTIME_CACHE);
    const shell = await caches.open(SHELL_CACHE);
    const cached = await cache.match(request)
      || await shell.match(request);
    if (cached) return cached;
  } catch {
    // Continue to the network when storage is unavailable.
  }
  const response = await fetch(request);
  await putBestEffort(RUNTIME_CACHE, request, response);
  return response;
}

async function networkFirst(request, fallback = null) {
  try {
    const response = await fetchWithTimeout(request);
    await putBestEffort(RUNTIME_CACHE, request, response);
    return response;
  } catch {
    try {
      const cache = await caches.open(RUNTIME_CACHE);
      const shell = await caches.open(SHELL_CACHE);
      return await cache.match(request)
        || await shell.match(request)
        || (fallback && (await cache.match(fallback) || await shell.match(fallback)))
        || Response.error();
    } catch {
      return Response.error();
    }
  }
}

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE)
      .then((cache) => cacheShell(cache))
      .catch(() => undefined)
  );
  // Deliberately do not skipWaiting: an active game should finish on one
  // coherent asset revision.
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((names) => Promise.all(names
        .filter((name) => name.startsWith(CACHE_PREFIX) && ![SHELL_CACHE, RUNTIME_CACHE].includes(name))
        .map((name) => caches.delete(name))))
      .catch(() => undefined)
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET' || !sameOrigin(request)) return;
  const url = new URL(request.url);
  if (isNavigation(request)) {
    const rootPath = url.pathname.endsWith('/') || url.pathname.endsWith('/index.html');
    const fallback = rootPath ? new Request(new URL('./index.html', self.location).href, { method: 'GET' }) : null;
    event.respondWith(networkFirst(request, fallback));
  } else if (isConfig(url) || isMetadata(url)) {
    event.respondWith(networkFirst(request));
  } else if (isStableAsset(request, url)) {
    event.respondWith(cacheFirst(request));
  }
});
