"""Check visible intermediate positions and sequencing against native outcomes."""

import json

import gipf_engine as ge
import pytest

from test_browser_mapping import (
    GEO, block_external_fonts, browser, browser_board, expected_board, local_worker, site_url,
)


def state_with(board, phase="push", current=1):
    turn = current if phase == "capture" else -current
    return ge.State.from_dict({
        "board": board, "reserves": [8, 8], "current_player": current,
        "turn_player": turn,
        "phase": phase, "ply": 1 if turn == 1 else 2,
    })


def push_scenario(current=1):
    board = [0] * 37
    ray = GEO["rays"][0]
    for cell in ray[:2]:
        board[cell] = 1
    board[ray[2]] = -1
    free = [i for i in range(37) if i not in ray]
    board[free[0]], board[free[-1]] = 2, -2
    return state_with(board, current=current), [0]


def capture_scenario():
    board = [0] * 37
    for line in [3, 10]:
        for cell in GEO["lines"][line][:4]:
            board[cell] = -1
    board[GEO["coordinates"].index([0, 0])] = -2
    board[GEO["coordinates"].index([3, -3])] = -2
    board[GEO["coordinates"].index([3, 0])] = 2
    board[GEO["lines"][3][4]] = 1
    return state_with(board, phase="capture", current=-1), [42 + 3 * 128, 42 + 10 * 128]


def motion_page(browser, site_url, initial, actions, *, ai=False, reduced=False):
    states = []
    native = initial.clone()
    for action in actions:
        states.append({"state": native.serialize(), "legal": native.legal_actions()})
        assert action in native.legal_actions()
        native.apply(action)
    states.append({"state": native.serialize(), "legal": native.legal_actions()})
    script = f'''
      var createGipfEngine = async function() {{
        const snapshots = {json.dumps(states)};
        class State {{
          constructor() {{ this.stage = 0; Object.assign(this, snapshots[0].state); }}
          clone() {{ const copy = new State(); Object.assign(copy, JSON.parse(JSON.stringify(this))); return copy; }}
          legal_actions() {{ return snapshots[this.stage].legal; }}
          apply(action) {{
            window.applied = (window.applied || []).concat(action);
            this.stage += 1;
            Object.assign(this, JSON.parse(JSON.stringify(snapshots[this.stage].state)));
          }}
          serialize() {{ return JSON.parse(JSON.stringify(this)); }}
        }}
        return {{ State, geometry: () => ({json.dumps(GEO)}) }};
      }};
    '''
    page = browser.new_page(viewport={"width": 1100, "height": 900}, reduced_motion="reduce" if reduced else "no-preference")
    page.set_default_timeout(5_000)
    block_external_fonts(page)
    # Pause the real browser animations to inspect intermediate frames without
    # relying on wall-clock timing or slowing production motion.
    page.add_init_script('''
      const animate = Element.prototype.animate;
      Element.prototype.animate = function(...args) {
        const animation = animate.apply(this, args);
        animation.pause(); animation.currentTime = 0;
        return animation;
      };
    ''')
    page.route("**/gipf_engine.js", lambda route: route.fulfill(content_type="application/javascript", body=script))
    page.route("**/config.json", lambda route: route.fulfill(content_type="application/json", body=json.dumps({"aiMode": "browser", "engineUrl": "./gipf_engine.js"})))
    if ai:
        page.route("**/ai-worker.js", lambda route: route.fulfill(content_type="application/javascript", body=local_worker(actions)))
    page.goto(site_url + "/?designs=1", wait_until="domcontentloaded")
    page.wait_for_function("document.querySelector('#engine-status').textContent === 'Ready to play'")
    return page, native, None


def finish_motion(page):
    page.evaluate('document.querySelector("#board").getAnimations({subtree: true}).forEach(animation => animation.finish())')


def test_push_slides_the_entire_prefix_and_undo_cancels_motion(browser, site_url):
    initial, actions = push_scenario()
    page, expected, _ = motion_page(browser, site_url, initial, actions)
    try:
        page.locator('.ray-hit[data-ray-id="0"]').click()
        assert browser_board(page) == expected_board(expected)
        assert page.locator('#board').get_attribute('aria-busy') == 'true'
        assert page.locator('#designs-dialog, #designs-open').count() == 0
        frames = page.evaluate('''() => {
          const animations = document.querySelector("#board").getAnimations({subtree: true});
          return [0, 220, 440].map(time => animations.map(animation => {
            animation.currentTime = time;
            const matrix = new DOMMatrix(getComputedStyle(animation.effect.target).transform);
            return Math.hypot(matrix.e, matrix.f);
          }));
        }''')
        assert len(frames[0]) == 4  # inserted stone plus three neighbors, including an identical pair
        for start, middle, end in zip(*frames):
            assert start == pytest.approx(63, abs=.01)
            assert 0 < middle < start
            assert end == pytest.approx(0, abs=.01)
        # A second click while motion is paused cannot sneak another move in.
        page.locator('.ray-hit[data-ray-id="0"]').click()
        assert page.evaluate('window.applied') == [0]
        page.locator('#undo').click()
        assert browser_board(page) == expected_board(initial)
        assert page.evaluate('document.querySelector("#board").getAnimations({subtree: true}).length') == 0
        assert page.locator('#board').get_attribute('aria-busy') == 'false'
    finally:
        page.close()


def test_ai_capture_holds_the_row_then_fades_before_its_next_action(browser, site_url):
    initial, actions = capture_scenario()
    page, expected, calls = motion_page(browser, site_url, initial, actions, ai=True)
    try:
        page.locator('#mode-select').select_option('ai')
        page.wait_for_function('window.applied?.length === 1')
        assert page.locator('#ai-status').text_content() == 'test-local-1 ready'
        assert page.locator('.removal-row-line').count() == 1
        assert page.locator('.stone-removing').count() == 4
        assert '1 captured' in page.locator('#board-hint').inner_text()
        frames = page.locator('.stone-removing').first.evaluate('''stone => {
          const animation = stone.getAnimations()[0];
          return [300, 860].map(time => {
            animation.currentTime = time;
            const style = getComputedStyle(stone);
            return {opacity: Number(style.opacity), scale: new DOMMatrix(style.transform).a};
          });
        }''')
        assert frames[0] == {"opacity": 1, "scale": 1}
        assert 0 < frames[1]['opacity'] < 1
        assert .35 < frames[1]['scale'] < 1
        finish_motion(page)
        page.wait_for_function('window.applied?.length === 2')
        assert page.locator('#ai-status').text_content() == 'test-local-2 ready'
        assert page.locator('#board').get_attribute('aria-busy') == 'true'
        finish_motion(page)
        page.wait_for_function("document.querySelector('#board').getAttribute('aria-busy') === 'false'")
        assert page.locator('#removal-layer').locator('*').count() == 0
        assert browser_board(page) == expected_board(expected)
    finally:
        page.close()


def test_ai_push_uses_the_same_slide_as_a_human_move(browser, site_url):
    initial, actions = push_scenario(current=-1)
    page, expected, calls = motion_page(browser, site_url, initial, actions, ai=True)
    try:
        page.locator('#mode-select').select_option('ai')
        page.wait_for_function('window.applied?.length === 1')
        distances = page.evaluate('''() => document.querySelector('#board').getAnimations({subtree: true}).map(animation => {
          animation.currentTime = 220;
          const matrix = new DOMMatrix(getComputedStyle(animation.effect.target).transform);
          return Math.hypot(matrix.e, matrix.f);
        })''')
        assert len(distances) == 4
        assert all(0 < distance < 63 for distance in distances)
        assert page.locator('#board').get_attribute('aria-busy') == 'true'
        assert page.locator('#ai-status').text_content() == 'test-local-1 ready'
        finish_motion(page)
        page.wait_for_function("document.querySelector('#board').getAttribute('aria-busy') === 'false'")
        assert browser_board(page) == expected_board(expected)
    finally:
        page.close()


def test_undo_during_removal_restores_stones_without_ghosts(browser, site_url):
    initial, actions = capture_scenario()
    page, _, _ = motion_page(browser, site_url, initial, actions)
    try:
        page.get_by_role('button', name='Line 4', exact=True).click()
        page.locator('#confirm-capture').click()
        assert page.locator('.stone-removing').count() == 4
        page.locator('#undo').click()
        assert browser_board(page) == expected_board(initial)
        assert page.locator('#removal-layer').locator('*').count() == 0
        assert page.evaluate('document.querySelector("#board").getAnimations({subtree: true}).length') == 0
        assert page.locator('#board').get_attribute('aria-busy') == 'false'
    finally:
        page.close()


def test_reduced_motion_applies_moves_without_animation(browser, site_url):
    initial, actions = push_scenario()
    page, expected, _ = motion_page(browser, site_url, initial, actions, reduced=True)
    try:
        page.locator('.ray-hit[data-ray-id="0"]').click()
        assert browser_board(page) == expected_board(expected)
        assert page.evaluate('document.querySelector("#board").getAnimations({subtree: true}).length') == 0
        assert page.locator('#board').get_attribute('aria-busy') == 'false'
    finally:
        page.close()
