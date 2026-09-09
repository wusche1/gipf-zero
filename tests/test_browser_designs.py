"""Browser checks for the selectable piece design gallery."""

from test_browser_mapping import browser, block_external_fonts, site_url


DESIGN_IDS = [
    "solid-open",
    "one-two-lobes",
    "full-split",
    "bowl-dome",
    "thin-tall",
    "one-two-pips",
    "plain-scallop",
    "one-two-rings",
    "one-two-bars",
    "diamond-star",
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
        first_preview = page.locator(".design-card").first.locator(".design-preview")
        assert first_preview.locator(".preview-piece").count() == 6
        assert first_preview.locator(".piece-visual").count() >= 2
        assert first_preview.locator(".design-preview-group").all_text_contents() == ["IVORY", "OBSIDIAN"]
        assert first_preview.locator(".design-preview-label").all_text_contents() == ["single", "double", "single", "double"]
        shape_tags = page.locator(".design-preview .piece-visual").evaluate_all("elements => [...new Set(elements.map(element => element.tagName))]")
        assert len(shape_tags) >= 4
        assert page.locator('.design-card[data-design="solid-open"] .design-hole').count() == 2
        assert page.locator('.design-card[data-design="one-two-lobes"] .piece-visual').count() == 4
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
        page.locator('.design-card[data-design="one-two-bars"]').click()
        page.get_by_role("button", name="Play with this design").click()
        page.reload(wait_until="domcontentloaded")
        page.locator("#engine-status").wait_for(state="visible")
        assert page.locator("body").get_attribute("data-piece-design") == "one-two-bars"
        assert page.locator("#piece-layer .piece-design-one-two-bars").count() > 0
    finally:
        page.close()
