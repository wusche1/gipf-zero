"""AI-mode readout, evaluation reset, capture prompt placement and game-over display."""
import json

from test_browser_mapping import GEO, block_external_fonts, browser, site_url
from test_browser_motion import state_with

WORKER = '''
  self.onmessage = ({data}) => {
    if (data.type === 'init') self.postMessage({id: data.id, ready: true, model: 'test-local', backend: 'wasm'});
    if (data.type === 'search' && data.state.current_player === -1) self.postMessage({id: data.id, action: 21, stats: {root_visits: 10, white_value: -1, evalsPerSec: 0 / 0, pv: [21], actions: [21], visits: [10]}});
  };
'''


def evaluation(page):
    return [page.locator(s).text_content() for s in ("#eval-label", "#think-readout")] + [page.locator("#eval-fill").evaluate("e => e.style.width")]


def test_readout_omits_an_undefined_rate_and_resets_with_the_game(browser, site_url):
    page = browser.new_page(viewport={"width": 1280, "height": 900}, reduced_motion="reduce")
    page.set_default_timeout(30_000)
    block_external_fonts(page)
    page.route("**/ai-worker.js", lambda route: route.fulfill(content_type="application/javascript", body=WORKER))
    try:
        page.goto(site_url, wait_until="domcontentloaded")
        page.locator("#engine-status").get_by_text("Ready to play").wait_for()
        page.locator("#mode-select").select_option("ai")
        page.locator('.ray-hit[data-ray-id="1"]').click()
        page.wait_for_function("document.querySelector('#move-count').textContent === 'MOVE 02'")
        assert evaluation(page) == ["Obsidian 100%", "10 sims · depth 1", "0%"]
        neutral = ["Even", "Waiting for the first search", "50%"]
        page.locator("#new-game").click()
        assert evaluation(page) == neutral
        page.locator('.ray-hit[data-ray-id="1"]').click()
        page.wait_for_function("document.querySelector('#move-count').textContent === 'MOVE 02'")
        page.locator("#opponent").select_option(label="Classic")
        assert evaluation(page) == neutral
        page.locator('.ray-hit[data-ray-id="0"]').click()
        page.wait_for_function("document.querySelector('#move-count').textContent === 'MOVE 04'")
        assert evaluation(page)[0] == "Obsidian 100%"
        page.locator("#ai-color").select_option("white")
        assert evaluation(page) == neutral
    finally:
        page.close()


def test_mobile_capture_prompt_sits_under_the_board_and_game_over_names_the_winner(browser, site_url):
    board = [0] * 37
    for cell in GEO["lines"][3][:4]:
        board[cell] = 1
    board[GEO["coordinates"].index([3, -3])] = 2
    board[GEO["coordinates"].index([3, 0])] = -2
    initial = state_with(board, phase="capture", current=1)
    won = initial.clone()
    won.apply(initial.legal_actions()[0])
    snapshots = [{"state": initial.serialize(), "legal": initial.legal_actions()}, {"state": {**won.serialize(), "winner": 1}, "legal": []}]
    engine = f'''
      var createGipfEngine = async function() {{
        const snapshots = {json.dumps(snapshots)};
        class State {{
          constructor() {{ this.stage = 0; Object.assign(this, snapshots[0].state); }}
          clone() {{ const copy = new State(); Object.assign(copy, JSON.parse(JSON.stringify(this))); return copy; }}
          legal_actions() {{ return snapshots[this.stage].legal; }}
          apply() {{ this.stage += 1; Object.assign(this, JSON.parse(JSON.stringify(snapshots[this.stage].state))); }}
          serialize() {{ return JSON.parse(JSON.stringify(this)); }}
        }}
        return {{ State, geometry: () => ({json.dumps(GEO)}) }};
      }};
    '''
    page = browser.new_page(viewport={"width": 390, "height": 844}, reduced_motion="reduce")
    page.set_default_timeout(30_000)
    block_external_fonts(page)
    page.route("**/gipf_engine.js", lambda route: route.fulfill(content_type="application/javascript", body=engine))
    page.route("**/ai-worker.js", lambda route: route.fulfill(content_type="application/javascript", body=WORKER))
    try:
        page.goto(site_url, wait_until="domcontentloaded")
        page.locator("#engine-status").get_by_text("Ready to play").wait_for()
        page.locator("#mode-select").select_option("ai")
        assert page.locator("#capture-copy").text_content().startswith("One row is complete")
        assert page.locator(".capture-option").count() == 1 and page.locator("#capture-options").is_hidden()
        assert page.locator(".capture-row-line.selected").count() == 1
        board_box, card, confirm = (page.locator(s).bounding_box() for s in ("#board", "#capture-card", "#confirm-capture"))
        assert board_box["y"] >= 0 and card["y"] >= board_box["y"] + board_box["height"] * .8
        assert confirm["y"] + confirm["height"] <= 844 and confirm["height"] >= 40
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.locator("#confirm-capture").click()
        assert page.locator("#phase-title").text_content() == "Ivory wins"
        assert page.locator(".turn-pip.active").count() == 0
        assert evaluation(page) == ["Ivory wins", "Game over", "100%"]
        assert page.locator("#capture-card").is_hidden()
    finally:
        page.close()
