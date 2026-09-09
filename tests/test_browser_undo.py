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


def configure(page, site_url, script, endpoint="https://ai.test"):
    block_external_fonts(page)
    page.route("**/gipf_engine.js", lambda route: route.fulfill(content_type="application/javascript", body=script))
    page.route(
        "**/config.json",
        lambda route: route.fulfill(
            content_type="application/json",
            body=json.dumps({"aiEndpoint": endpoint, "engineUrl": "./gipf_engine.js"}),
        ),
    )
    if endpoint:
        page.route(f"{endpoint}/api/status", lambda route: route.fulfill(content_type="application/json", body='{"model":"test"}'))
    page.goto(site_url, wait_until="domcontentloaded")


def test_ai_undo_restores_latest_human_decision_without_replaying_ai(browser, site_url):
    page = browser.new_page(viewport={"width": 1000, "height": 900})
    page.set_default_timeout(5_000)
    calls = []
    initial = ge.State().serialize()
    configure(page, site_url, fake_engine(initial))

    def ai_route(route):
        calls.append(route.request.post_data_json)
        route.fulfill(content_type="application/json", body='{"action":0,"model":"test"}')

    page.route("https://ai.test/api/move", ai_route)
    try:
        page.locator("#mode-select").select_option("ai")
        page.locator('.ray-hit[data-ray-id="0"]').click()
        page.wait_for_function("document.querySelector('#move-count').textContent === 'MOVE 02'")
        assert len(calls) == 1
        page.locator("#undo").click()
        assert page.locator("#move-count").text_content() == "MOVE 00"
        assert page.locator("#undo").is_disabled()
        assert len(calls) == 1
    finally:
        page.close()


def test_local_undo_reverts_one_decision(browser, site_url):
    page = browser.new_page(viewport={"width": 1000, "height": 900})
    page.set_default_timeout(5_000)
    initial = ge.State().serialize()
    configure(page, site_url, fake_engine(initial), endpoint="")
    try:
        page.locator('.ray-hit[data-ray-id="0"]').click()
        assert page.locator("#move-count").text_content() == "MOVE 01"
        page.locator("#undo").click()
        assert page.locator("#move-count").text_content() == "MOVE 00"
        assert page.locator("#undo").is_disabled()
    finally:
        page.close()


def test_ai_white_opening_has_no_human_undo(browser, site_url):
    page = browser.new_page(viewport={"width": 1000, "height": 900})
    page.set_default_timeout(5_000)
    initial = ge.State().serialize()
    configure(page, site_url, fake_engine(initial))
    page.route("https://ai.test/api/move", lambda route: route.fulfill(content_type="application/json", body='{"action":0,"model":"test"}'))
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
    page.route("**/gipf_engine.js", lambda route: route.fulfill(content_type="application/javascript", body=script))
    page.route("**/config.json", lambda route: route.fulfill(content_type="application/json", body='{"aiEndpoint":"https://ai.test","engineUrl":"./gipf_engine.js"}'))
    page.route("https://ai.test/api/status", lambda route: route.fulfill(content_type="application/json", body='{"model":"test"}'))
    try:
        page.goto(site_url, wait_until="domcontentloaded")
        page.locator("#mode-select").select_option("ai")
        page.locator('.ray-hit[data-ray-id="0"]').click()
        page.wait_for_function("window.aiResolvers.length === 1")
        page.locator("#undo").click()
        page.evaluate("window.aiResolvers[0]()")
        page.wait_for_timeout(100)
        assert page.locator("#move-count").text_content() == "MOVE 00"
        assert page.evaluate("window.applied || []") == [0]
    finally:
        page.close()
