"""Browser checks for human undo semantics around machine turns."""

import json

from test_browser_mapping import GEO, block_external_fonts, browser, site_url

import gipf_engine as ge


def fake_engine(initial):
    return f'''
      var createGipfEngine = async function() {{
        const initial = {json.dumps(initial)};
        const geometry = {json.dumps({"rays": GEO["rays"]})};
        class State {{
          constructor() {{ Object.assign(this, JSON.parse(JSON.stringify(initial))); }}
          clone() {{ const other = new State(); Object.assign(other, JSON.parse(JSON.stringify(this))); return other; }}
          legal_actions() {{ return [0]; }}
          apply(action) {{ window.applied = (window.applied || []).concat([action]); this.ply += 1; this.current_player = this.current_player === 1 ? -1 : 1; }}
          serialize() {{ return JSON.parse(JSON.stringify(this)); }}
        }}
        return {{ State, geometry: () => geometry }};
      }};
    '''


def local_worker(*, delay_ms=0):
    """Return a deterministic classic worker matching the browser AI protocol."""
    return f'''
      let searches = 0;
      self.onmessage = (event) => {{
        const request = event.data;
        if (request.type === 'init') {{
          self.postMessage({{ id: request.id, ready: true, model: 'test-local', backend: 'wasm' }});
          return;
        }}
        if (request.type === 'search') {{
          searches += 1;
          const reply = () => self.postMessage({{ id: request.id, action: 0, model: `test-local-${{searches}}` }});
          {"setTimeout(reply, %d);" % delay_ms if delay_ms else "reply();"}
        }}
      }};
    '''


def configure(page, site_url, script, *, worker=None, worker_requests=None):
    block_external_fonts(page)
    page.route("**/gipf_engine.js", lambda route: route.fulfill(content_type="application/javascript", body=script))
    page.route(
        "**/config.json",
        lambda route: route.fulfill(
            content_type="application/json",
            body=json.dumps({"aiMode": "browser", "engineUrl": "./gipf_engine.js"}),
        ),
    )
    def serve_worker(route):
        if worker_requests is not None:
            worker_requests.append(route.request.url)
        route.fulfill(content_type="application/javascript", body=worker if worker is not None else local_worker())
    page.route("**/ai-worker.js", serve_worker)
    page.goto(site_url, wait_until="domcontentloaded")


def test_ai_undo_restores_latest_human_decision_without_replaying_ai(browser, site_url):
    page = browser.new_page(viewport={"width": 1000, "height": 900})
    page.set_default_timeout(5_000)
    initial = ge.State().serialize()
    configure(page, site_url, fake_engine(initial))
    try:
        page.locator("#mode-select").select_option("ai")
        page.locator('.ray-hit[data-ray-id="0"]').click()
        page.wait_for_function("document.querySelector('#move-count').textContent === 'MOVE 02'")
        assert page.locator("#ai-status").text_content() == "test-local-1 ready"
        page.locator("#undo").click()
        assert page.locator("#move-count").text_content() == "MOVE 00"
        assert page.locator("#undo").is_disabled()
        page.wait_for_timeout(100)
        assert page.locator("#ai-status").text_content() == "test-local-1 ready"
    finally:
        page.close()


def test_local_undo_reverts_one_decision(browser, site_url):
    page = browser.new_page(viewport={"width": 1000, "height": 900})
    page.set_default_timeout(5_000)
    initial = ge.State().serialize()
    worker_requests = []
    configure(page, site_url, fake_engine(initial), worker_requests=worker_requests)
    try:
        page.locator('.ray-hit[data-ray-id="0"]').click()
        assert page.locator("#move-count").text_content() == "MOVE 01"
        page.locator("#undo").click()
        assert page.locator("#move-count").text_content() == "MOVE 00"
        assert page.locator("#undo").is_disabled()
        assert worker_requests == []
    finally:
        page.close()


def test_ai_white_opening_has_no_human_undo(browser, site_url):
    page = browser.new_page(viewport={"width": 1000, "height": 900})
    page.set_default_timeout(5_000)
    initial = ge.State().serialize()
    configure(page, site_url, fake_engine(initial))
    try:
        page.locator("#mode-select").select_option("ai")
        page.locator("#ai-color").select_option("white")
        page.wait_for_function("document.querySelector('#move-count').textContent === 'MOVE 01'")
        assert page.locator("#undo").is_disabled()
    finally:
        page.close()


def test_pending_ai_reply_after_undo_is_ignored(browser, site_url):
    page = browser.new_page(viewport={"width": 1000, "height": 900})
    page.set_default_timeout(5_000)
    block_external_fonts(page)
    initial = ge.State().serialize()
    script = fake_engine(initial)
    page.route("**/gipf_engine.js", lambda route: route.fulfill(content_type="application/javascript", body=script))
    page.route("**/config.json", lambda route: route.fulfill(content_type="application/json", body='{"aiMode":"browser","engineUrl":"./gipf_engine.js"}'))
    page.route("**/ai-worker.js", lambda route: route.fulfill(content_type="application/javascript", body=local_worker(delay_ms=250)))
    try:
        page.goto(site_url, wait_until="domcontentloaded")
        page.locator("#mode-select").select_option("ai")
        page.locator('.ray-hit[data-ray-id="0"]').click()
        page.wait_for_function("document.querySelector('#undo').disabled === false")
        page.locator("#undo").click()
        page.wait_for_timeout(400)
        assert page.locator("#move-count").text_content() == "MOVE 00"
        assert page.evaluate("window.applied || []") == [0]
    finally:
        page.close()
