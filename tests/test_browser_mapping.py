"""Rendered-control checks for the static board against the native rule engine."""

import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import gipf_engine as ge

try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:  # pragma: no cover - developer environments may omit browsers
    sync_playwright = None


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
GEO = ge.geometry()


class QuietStaticHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def site_url():
    handler = functools.partial(QuietStaticHandler, directory=str(WEB))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
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
            browser = playwright.chromium.launch(headless=True)
        except Exception as error:
            pytest.skip(f"Chromium is not installed: {error}")
        yield browser
        browser.close()


def browser_board(page):
    pieces = page.locator("circle.piece").evaluate_all(
        "elements => elements.map(element => ({x:+element.getAttribute('cx'), y:+element.getAttribute('cy'), className:element.getAttribute('class')}))"
    )
    counts = {}
    for piece in pieces:
        q, r = min(
            (tuple(point) for point in GEO["coordinates"]),
            key=lambda point: (400 + 63 * (point[0] + point[1] * 0.5) - piece["x"]) ** 2
            + (400 + 63 * (3 ** 0.5) / 2 * point[1] - piece["y"]) ** 2,
        )
        key = (q, r, 1 if "ivory" in piece["className"] else -1)
        counts[key] = counts.get(key, 0) + 1
    return {(q, r): colour * (2 if count == 2 else 1) for (q, r, colour), count in counts.items()}


def expected_board(state):
    return {
        tuple(GEO["coordinates"][index]): piece
        for index, piece in enumerate(state.board)
        if piece
    }


def block_external_fonts(page):
    """Keep browser tests deterministic when Google Fonts is unavailable."""
    page.route("https://fonts.googleapis.com/**", lambda route: route.abort())
    page.route("https://fonts.gstatic.com/**", lambda route: route.abort())


def ready_page(browser, site_url):
    page = browser.new_page(viewport={"width": 1000, "height": 900})
    page.set_default_timeout(5_000)
    block_external_fonts(page)
    page.route("**/config.json", lambda route: route.fulfill(content_type="application/json", body='{"aiEndpoint":"","engineUrl":"./gipf_engine.js"}'))
    page.goto(site_url, wait_until="domcontentloaded")
    expect(page.locator("#engine-status")).to_have_text("Ready to play", timeout=5_000)
    return page


def test_every_rendered_push_ray_applies_its_native_action(browser, site_url):
    page = ready_page(browser, site_url)
    try:
        for action in range(42):
            if action:
                page.locator("#new-game").click()
            assert page.locator(f'.ray-hit[data-ray-id="{action}"]').count() == 1
            page.locator(f'.ray-hit[data-ray-id="{action}"]').click()
            assert page.locator("#move-count").text_content() == "MOVE 01"
            expected = ge.State()
            expected.apply(action)
            assert browser_board(page) == expected_board(expected)
    finally:
        page.close()


def test_optional_capture_toggle_preserves_the_engine_action_encoding(browser, site_url):
    line = GEO["lines"][3]
    board = [0] * 37
    for cell in line[:4]:
        board[cell] = 1
    board[line[4]] = 2
    board[GEO["coordinates"].index([3, -3])] = 2
    board[GEO["coordinates"].index([3, 0])] = -2
    state = ge.State.from_dict({"board": board, "reserves": [8, 10], "captured": [2, 6], "current_player": 1, "turn_player": 1, "phase": "capture", "winner": 0, "ply": 1})
    actions = state.legal_actions()
    selected = 42 + 3 * 128 + (1 << 4)
    assert selected in actions
    fake_engine = f'''
      var createGipfEngine = async function() {{
        const initial = {json.dumps(state.serialize())};
        const actions = {json.dumps(actions)};
        const geometry = {json.dumps({"rays": GEO["rays"], "lines": GEO["lines"]})};
        class State {{
          constructor() {{ Object.assign(this, JSON.parse(JSON.stringify(initial))); }}
          clone() {{ const other = new State(); Object.assign(other, JSON.parse(JSON.stringify(this))); return other; }}
          legal_actions() {{ return this.phase === 'capture' ? actions : []; }}
          apply(action) {{ window.captureActions = (window.captureActions || []).concat([action]); this.phase = 'push'; this.current_player = -1; }}
          serialize() {{ return JSON.parse(JSON.stringify(this)); }}
        }}
        return {{ State, geometry: () => geometry }};
      }};
    '''
    page = browser.new_page(viewport={"width": 1000, "height": 900})
    page.set_default_timeout(5_000)
    block_external_fonts(page)
    page.route("**/gipf_engine.js", lambda route: route.fulfill(content_type="application/javascript", body=fake_engine))
    page.route("**/config.json", lambda route: route.fulfill(content_type="application/json", body='{"aiEndpoint":"","engineUrl":"./gipf_engine.js"}'))
    try:
        page.goto(site_url, wait_until="domcontentloaded")
        page.get_by_role("button", name="Line 4").click()
        page.get_by_role("button", name="Keep D5").click()
        assert page.locator(".capture-row-line").count() > 0
        assert page.locator("circle.double-piece").count() >= 2
        assert page.locator("circle.double-ring").count() >= 1
        assert page.locator(".piece-visual.capture-remove").first.is_visible()
        assert page.locator("circle.single-piece.capture-remove").count() == 0
        assert page.get_by_text("D5").count() >= 1
        assert page.get_by_role("button", name="Remove D5").count() == 1
        page.get_by_role("button", name="Remove D5").click()
        assert page.locator(".piece-visual.capture-keep").first.is_visible()
        assert page.get_by_role("button", name="Keep D5").count() == 1
        page.get_by_role("button", name="Keep D5").click()
        page.locator("#confirm-capture").click()
        assert page.evaluate("window.captureActions") == [selected]
    finally:
        page.close()


def test_ai_resolves_every_capture_decision_in_a_chain(browser, site_url):
    first, second = 42 + 3 * 128, 42 + 10 * 128
    initial = ge.State().serialize()
    initial.update(current_player=-1, turn_player=-1, phase="capture", ply=2)
    fake_engine = f'''
      var createGipfEngine = async function() {{
        const initial = {json.dumps(initial)};
        const geometry = {json.dumps({"rays": GEO["rays"]})};
        class State {{
          constructor() {{ Object.assign(this, JSON.parse(JSON.stringify(initial))); this.stage = 0; }}
          clone() {{ const other = new State(); Object.assign(other, JSON.parse(JSON.stringify(this))); return other; }}
          legal_actions() {{ return this.phase === 'capture' ? (this.stage ? [{second}] : [{first}]) : [0]; }}
          apply(action) {{ window.aiApplied = (window.aiApplied || []).concat([action]); this.stage += 1; if (this.stage === 1) {{ this.phase = 'capture'; this.current_player = -1; }} else {{ this.phase = 'push'; this.current_player = 1; }} }}
          serialize() {{ return JSON.parse(JSON.stringify(this)); }}
        }}
        return {{ State, geometry: () => geometry }};
      }};
    '''
    calls = []
    page = browser.new_page(viewport={"width": 1000, "height": 900})
    page.set_default_timeout(5_000)
    block_external_fonts(page)
    page.route("**/gipf_engine.js", lambda route: route.fulfill(content_type="application/javascript", body=fake_engine))
    page.route("**/config.json", lambda route: route.fulfill(content_type="application/json", body='{"aiEndpoint":"https://ai.test","engineUrl":"./gipf_engine.js"}'))

    def ai_route(route):
        calls.append(route.request.post_data_json)
        action = first if len(calls) == 1 else second
        route.fulfill(content_type="application/json", body=json.dumps({"action": action, "model": "test"}))

    page.route("https://ai.test/api/move", ai_route)
    page.route("https://ai.test/api/status", lambda route: route.fulfill(content_type="application/json", body='{"model":"test"}'))
    try:
        page.goto(site_url, wait_until="domcontentloaded")
        page.locator("#mode-select").select_option("ai")
        page.wait_for_function("window.aiApplied && window.aiApplied.length === 2")
        assert page.evaluate("window.aiApplied") == [first, second]
        assert len(calls) == 2
        assert page.locator("#capture-card").is_hidden()
    finally:
        page.close()


def test_stale_ai_response_cannot_mutate_a_new_game(browser, site_url):
    initial = ge.State().serialize()
    initial.update(current_player=-1, turn_player=-1, phase="push", ply=0)
    fake_engine = f'''
      var createGipfEngine = async function() {{
        const initial = {json.dumps(initial)};
        const geometry = {json.dumps({"rays": GEO["rays"]})};
        class State {{
          constructor() {{ Object.assign(this, JSON.parse(JSON.stringify(initial))); }}
          clone() {{ const other = new State(); Object.assign(other, JSON.parse(JSON.stringify(this))); return other; }}
          legal_actions() {{ return [0]; }}
          apply(action) {{ window.aiApplied = (window.aiApplied || []).concat([action]); this.ply += 1; }}
          serialize() {{ return JSON.parse(JSON.stringify(this)); }}
        }}
        return {{ State, geometry: () => geometry }};
      }};
    '''
    page = browser.new_page(viewport={"width": 1000, "height": 900})
    page.set_default_timeout(5_000)
    block_external_fonts(page)
    page.add_init_script("""
      window.aiResolvers = [];
      const nativeFetch = window.fetch.bind(window);
      window.fetch = (input, init) => {
        const url = typeof input === 'string' ? input : input.url;
        if (url === 'https://ai.test/api/move') {
          return new Promise(resolve => window.aiResolvers.push(() => resolve(new Response(JSON.stringify({action: 0, model: 'test'}), {status: 200, headers: {'content-type': 'application/json'}}))));
        }
        return nativeFetch(input, init);
      };
    """)
    page.route("**/gipf_engine.js", lambda route: route.fulfill(content_type="application/javascript", body=fake_engine))
    page.route("**/config.json", lambda route: route.fulfill(content_type="application/json", body='{"aiEndpoint":"https://ai.test","engineUrl":"./gipf_engine.js"}'))
    page.route("https://ai.test/api/status", lambda route: route.fulfill(content_type="application/json", body='{"model":"test"}'))
    try:
        page.goto(site_url, wait_until="domcontentloaded")
        page.locator("#mode-select").select_option("ai")
        page.wait_for_function("window.aiResolvers.length >= 1")
        page.locator("#new-game").click()
        page.wait_for_function("window.aiResolvers.length >= 2")
        page.evaluate("window.aiResolvers[0]()")
        page.wait_for_timeout(100)
        assert page.locator("#move-count").text_content() == "MOVE 00"
        assert page.evaluate("window.aiApplied || []") == []
    finally:
        page.close()
