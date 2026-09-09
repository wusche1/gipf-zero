"""Browser checks for the selectable piece design gallery."""

from test_browser_mapping import browser, block_external_fonts, site_url


DESIGN_IDS = [
    "natural-stack",
    "tall-stack",
    "offset-stack",
    "concentric-ring",
    "engraved-ii",
    "twin-pips",
    "petite-top",
    "hex-outline",
    "cut-groove",
    "contrast-band",
]


def design_page(browser, site_url):
    page = browser.new_page(viewport={"width": 1100, "height": 900})
    page.set_default_timeout(5_000)
    block_external_fonts(page)
    page.route("**/config.json", lambda route: route.fulfill(
        content_type="application/json",
        body='{"aiEndpoint":"","engineUrl":"./gipf_engine.js"}',
    ))
    page.goto(f"{site_url}/?designs=1", wait_until="domcontentloaded")
    page.evaluate("localStorage.removeItem('gipf-piece-design')")
    page.reload(wait_until="domcontentloaded")
    page.locator("#engine-status").wait_for(state="visible")
    return page


def test_gallery_shows_ten_designs_and_applies_each_to_the_board(browser, site_url):
    page = design_page(browser, site_url)
    try:
        assert page.locator("#designs-dialog").is_visible()
        assert page.locator(".design-card").count() == 10
        for design_id in DESIGN_IDS:
            page.locator(f'.design-card[data-design="{design_id}"]').click()
            assert page.locator("body").get_attribute("data-piece-design") == design_id
            assert page.locator(f"#piece-layer .piece-design-{design_id}").count() > 0
            assert page.locator(f'.design-card[data-design="{design_id}"].selected').count() == 1
    finally:
        page.close()


def test_selected_design_persists_after_reload(browser, site_url):
    page = design_page(browser, site_url)
    try:
        page.locator('.design-card[data-design="hex-outline"]').click()
        page.get_by_role("button", name="Play with this design").click()
        page.reload(wait_until="domcontentloaded")
        page.locator("#engine-status").wait_for(state="visible")
        assert page.locator("body").get_attribute("data-piece-design") == "hex-outline"
        assert page.locator("#piece-layer .piece-design-hex-outline").count() > 0
    finally:
        page.close()
