/* Optional static-site service-worker registration. */
export async function registerOffline({ scope = './', onStatus = () => {} } = {}) {
  if (!('serviceWorker' in navigator)) {
    onStatus('unsupported');
    return null;
  }
  try {
    const deadline = Date.now() + 2000;
    const url = new URL('./service-worker.js', import.meta.url);
    const timeout = Symbol('offline registration timeout');
    const raceUntilDeadline = async (promise) => {
      const remaining = Math.max(0, deadline - Date.now());
      let timer;
      const timeoutPromise = new Promise((resolve) => {
        timer = setTimeout(() => resolve(timeout), remaining);
      });
      try {
        return await Promise.race([promise, timeoutPromise]);
      } finally {
        clearTimeout(timer);
      }
    };
    const registration = await raceUntilDeadline(navigator.serviceWorker.register(url, { scope }));
    if (registration === timeout) {
      onStatus('unavailable', new Error('Service worker registration timed out'));
      return null;
    }
    onStatus('registered', registration);
    const ready = await raceUntilDeadline(navigator.serviceWorker.ready.catch(() => null));
    let controllerListener;
    try {
      if (ready !== timeout && !navigator.serviceWorker.controller) {
        const controllerChanged = new Promise((resolve) => {
          controllerListener = () => resolve(true);
          navigator.serviceWorker.addEventListener('controllerchange', controllerListener);
        });
        await raceUntilDeadline(controllerChanged);
      }
    } finally {
      if (controllerListener) navigator.serviceWorker.removeEventListener('controllerchange', controllerListener);
    }
    if (ready === timeout || !navigator.serviceWorker.controller) onStatus('pending', registration);
    else onStatus('ready', registration);
    return registration;
  } catch (error) {
    // Private browsing, quota policy, and unsupported static hosts should
    // leave the online game fully usable.
    onStatus('unavailable', error);
    return null;
  }
}

export function offlineSupported() {
  return 'serviceWorker' in navigator && 'caches' in window;
}
