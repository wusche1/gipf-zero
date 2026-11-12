"""Opening variety in the real worker, with its random source stubbed."""
import json

import gipf_engine as ge
from test_browser_mapping import WEB, browser, site_url
from test_browser_persistent_search import META, RPC, rpc

STUB = """let roll = 0; Math.random = () => roll;
importScripts('/ai-worker.js');
const inner = self.onmessage;
self.onmessage = (event) => event.data.type === 'roll' ? void (roll = event.data.value) : inner(event);"""


def variety_page(browser, url, **meta):
    page = browser.new_page()
    page.set_default_timeout(120_000)
    page.route("**/stub-worker.js", lambda route: route.fulfill(body=STUB, content_type="text/javascript"))
    page.route("**/models/*.json", lambda route: route.fulfill(body=json.dumps({**META, **meta}), content_type="application/json"))
    page.goto(url + "/config.json")
    page.evaluate(RPC.replace("/ai-worker.js", "/stub-worker.js"))
    assert page.evaluate("rpc({type:'init'})")["ready"]
    return page


def roll(page, value):
    page.evaluate("(value) => worker.postMessage({type:'roll', value})", value)


def best(stats):
    return stats["actions"][stats["visits"].index(max(stats["visits"]))]


def test_metadata_ships_the_variety_defaults():
    assert (META["varietyPlies"], META["varietyRatio"]) == (6, 0.6)


def test_opening_push_is_drawn_from_the_qualifying_moves_and_rerooted(browser, site_url):
    page = variety_page(browser, site_url, varietyRatio=0.01)
    try:
        state = ge.State()
        # The lowest roll draws the first qualifying move, the highest the last.
        for value in (0, 0.999999):
            roll(page, value)
            response = rpc(page, type="search", state=state.serialize(), budgetMs=400, fresh=True)
            stats = response["stats"]
            if response["action"] != best(stats):
                break
        floor = 0.01 * max(stats["visits"])
        pool = [i for i, count in enumerate(stats["visits"]) if count > 0 and count >= floor]
        pick = pool[0] if value == 0 else pool[-1]
        assert len(pool) > 1 and response["action"] == stats["action"] == stats["actions"][pick] != best(stats)
        assert stats["pv"] == [response["action"]] and stats["value"] == stats["q"][pick]
        assert stats["white_value"] == stats["q"][pick] * stats["actor"]
        state.apply(response["action"])
        after = rpc(page, type="search", state=state.serialize(), simulations=5)["stats"]
        assert after["reusedVisits"] == stats["visits"][pick] - 1
        assert after["root_visits"] == after["reusedVisits"] + 5
    finally:
        page.close()


def test_later_and_exact_decisions_stay_most_visited(browser, site_url):
    page = variety_page(browser, site_url, varietyRatio=0.01)
    try:
        state = ge.State()
        for value in (0, 0.999999):
            roll(page, value)
            exact = rpc(page, type="search", state=state.serialize(), simulations=150, fresh=True)
            assert exact["action"] == best(exact["stats"]) == exact["stats"]["pv"][0]
        while state.ply < 6:
            state.apply(state.legal_actions()[0])
        assert state.phase != "capture"
        for value in (0, 0.999999):
            roll(page, value)
            timed = rpc(page, type="search", state=state.serialize(), budgetMs=400, fresh=True)
            assert timed["action"] == best(timed["stats"]) == timed["stats"]["pv"][0]
    finally:
        page.close()


def test_zero_variety_plies_disables_it(browser, site_url):
    page = variety_page(browser, site_url, varietyRatio=0.01, varietyPlies=0)
    try:
        for value in (0, 0.999999):
            roll(page, value)
            timed = rpc(page, type="search", state=ge.State().serialize(), budgetMs=400, fresh=True)
            assert timed["action"] == best(timed["stats"]) == timed["stats"]["pv"][0]
    finally:
        page.close()
