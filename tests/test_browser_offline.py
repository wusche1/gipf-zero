"""Offline cache smoke for the real static engine and browser-local AI."""

import functools
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def site_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(WEB)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


@pytest.fixture
def versioned_site_url(tmp_path):
    web = tmp_path / "web"
    shutil.copytree(WEB, web)
    subprocess.run(
        [sys.executable, str(ROOT / "ops/version_web_assets.py")],
        cwd=tmp_path,
        env={**os.environ, "ASSET_VERSION": "offline-smoke"},
        check=True,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(web)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


@pytest.fixture(scope="module")
def browser():
    if sync_playwright is None:
        pytest.skip("Playwright is not installed")
    with sync_playwright() as playwright:
        try:
            instance = playwright.chromium.launch(headless=True)
        except Exception as error:
            pytest.skip(f"Chromium is not installed: {error}")
        yield instance
        instance.close()


def block_fonts(page):
    page.route("https://fonts.googleapis.com/**", lambda route: route.abort())
    page.route("https://fonts.gstatic.com/**", lambda route: route.abort())


def test_real_ai_assets_survive_offline_reload(browser, site_url):
    context = browser.new_context(service_workers="allow", viewport={"width": 1000, "height": 900})
    page = context.new_page()
    page.set_default_timeout(120_000)
    block_fonts(page)
    try:
        page.goto(site_url, wait_until="domcontentloaded")
        expect(page.locator("#engine-status")).to_have_text("Ready to play", timeout=30_000)
        page.evaluate("navigator.serviceWorker.ready.then(() => true)")
        page.reload(wait_until="domcontentloaded")
        expect(page.locator("#engine-status")).to_have_text("Ready to play", timeout=30_000)

        # Hotseat remains usable before any local model request.
        worker_requests = []
        page.on("request", lambda request: worker_requests.append(request.url) if request.url.endswith("/ai-worker.js") else None)
        page.locator("#mode-select").select_option("local")
        page.locator('.ray-hit[data-ray-id="0"]').click()
        expect(page.locator("#move-count")).to_have_text("MOVE 01")
        assert worker_requests == []
        page.locator("#undo").click()

        # Warm the actual model and runtime assets while online.
        page.locator("#mode-select").select_option("ai")
        page.locator('.ray-hit[data-ray-id="0"]').click()
        page.wait_for_function("document.querySelector('#move-count').textContent === 'MOVE 02'", timeout=120_000)
        assert "ready" in page.locator("#ai-status").text_content().lower()

        context.set_offline(True)
        page.reload(wait_until="domcontentloaded")
        expect(page.locator("#engine-status")).to_have_text("Ready to play", timeout=30_000)
        page.locator("#mode-select").select_option("ai")
        page.locator('.ray-hit[data-ray-id="0"]').click()
        page.wait_for_function("document.querySelector('#move-count').textContent === 'MOVE 02'", timeout=120_000)
        assert "ready" in page.locator("#ai-status").text_content().lower()
    finally:
        context.close()


def test_hung_registration_does_not_block_hotseat(browser, site_url):
    context = browser.new_context(service_workers="allow", viewport={"width": 1000, "height": 900})
    page = context.new_page()
    page.set_default_timeout(10_000)
    block_fonts(page)
    page.add_init_script("""
      if (navigator.serviceWorker) {
        navigator.serviceWorker.register = () => new Promise(() => {});
      }
    """)
    started = time.monotonic()
    try:
        page.goto(site_url, wait_until="domcontentloaded")
        expect(page.locator("#engine-status")).to_have_text("Ready to play", timeout=5_000)
        page.locator("#mode-select").select_option("local")
        page.locator('.ray-hit[data-ray-id="0"]').click()
        expect(page.locator("#move-count")).to_have_text("MOVE 01")
        assert time.monotonic() - started < 2
    finally:
        context.close()


def test_versioned_build_caches_hashed_assets_and_preserves_benchmark_navigation(browser, versioned_site_url):
    context = browser.new_context(service_workers="allow", viewport={"width": 1000, "height": 900})
    page = context.new_page()
    page.set_default_timeout(120_000)
    block_fonts(page)
    try:
        page.goto(versioned_site_url, wait_until="domcontentloaded")
        expect(page.locator("#engine-status")).to_have_text("Ready to play", timeout=30_000)
        page.evaluate("navigator.serviceWorker.ready.then(() => true)")

        page.locator("#mode-select").select_option("ai")
        page.locator('.ray-hit[data-ray-id="0"]').click()
        page.wait_for_function("document.querySelector('#move-count').textContent === 'MOVE 02'", timeout=120_000)
        keys = page.evaluate("""async () => {
          const names = await caches.keys();
          const urls = [];
          for (const name of names) for (const request of await (await caches.open(name)).keys()) urls.push(request.url);
          return urls;
        }""")
        assert any("gipf_engine.js?v=offline-smoke" in url for url in keys)
        assert any("gipf_engine.wasm?v=offline-smoke" in url for url in keys)
        assert any("ai-worker.js?v=offline-smoke" in url for url in keys)
        assert any("models/champion.onnx?" in url for url in keys)

        page.goto(f"{versioned_site_url}/benchmark.html", wait_until="domcontentloaded")
        expect(page.locator("h1")).to_have_text("How fast is your device?")
        context.set_offline(True)
        page.goto(f"{versioned_site_url}/benchmark.html", wait_until="domcontentloaded")
        expect(page.locator("h1")).to_have_text("How fast is your device?")
        page.goto(versioned_site_url, wait_until="domcontentloaded")
        expect(page.locator("#engine-status")).to_have_text("Ready to play", timeout=30_000)
        page.locator("#mode-select").select_option("ai")
        page.locator('.ray-hit[data-ray-id="0"]').click()
        page.wait_for_function("document.querySelector('#move-count').textContent === 'MOVE 02'", timeout=120_000)
    finally:
        context.close()
