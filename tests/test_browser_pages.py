import pytest
from playwright.sync_api import expect

from test_browser_mapping import browser, site_url

VIEWPORTS = [{"width": 1280, "height": 900}, {"width": 390, "height": 844}]


def open_page(browser, viewport):
    page = browser.new_page(viewport=viewport)
    page.set_default_timeout(60_000)
    page.route("https://fonts.googleapis.com/**", lambda route: route.fulfill(content_type="text/css", body=""))
    errors = []
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    page.on("pageerror", lambda error: errors.append(str(error)))
    return page, errors


def fills(page):
    return page.locator(".att-node").evaluate_all("nodes => nodes.map(node => node.style.fill)")


@pytest.mark.parametrize("viewport", VIEWPORTS)
@pytest.mark.parametrize("name", ["about.html", "benchmarks.html"])
def test_page_renders_cleanly(browser, site_url, name, viewport):
    page, errors = open_page(browser, viewport)
    try:
        page.goto(f"{site_url}/{name}", wait_until="load")
        expect(page.locator("h1")).to_be_visible()
        if name == "about.html":
            expect(page.locator(".att-node.selected")).to_have_count(1)
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
        assert errors == []
    finally:
        page.close()


def test_navigation_between_pages(browser, site_url):
    page, _ = open_page(browser, VIEWPORTS[0])
    try:
        page.goto(site_url, wait_until="domcontentloaded")
        page.locator(".footer").get_by_role("link", name="About the AI").click()
        expect(page.locator("h1")).to_contain_text("taught itself GIPF")
        page.locator(".page-nav").get_by_role("link", name="Benchmarks").click()
        expect(page.locator("h1")).to_contain_text("Stronger than every other GIPF program")
        page.locator(".page-nav").get_by_role("link", name="About the AI").click()
        expect(page.locator("h1")).to_contain_text("taught itself GIPF")
        page.locator(".page-nav").get_by_role("link", name="Play").click()
        expect(page.locator("#board")).to_be_visible()
        page.locator(".footer").get_by_role("link", name="Benchmarks").click()
        expect(page.locator("h1")).to_contain_text("Stronger than every other GIPF program")
        page.locator(".brand").click()
        expect(page.locator("#board")).to_be_visible()
    finally:
        page.close()


def test_attention_view_reacts_to_point_and_head(browser, site_url):
    page, errors = open_page(browser, VIEWPORTS[0])
    try:
        page.goto(f"{site_url}/about.html", wait_until="load")
        expect(page.locator('.att-node.selected[data-key="0,0"]')).to_have_count(1)
        expect(page.locator('.head[aria-pressed="true"]')).to_have_attribute("aria-label", "Layer 4, head 2")
        start = fills(page)
        assert len(start) == 37 and len(set(start)) > 3
        page.locator('.att-node[data-key="2,-1"]').click()
        expect(page.locator('.att-node.selected[data-key="2,-1"]')).to_have_count(1)
        moved = fills(page)
        assert moved != start
        page.get_by_role("button", name="Layer 2, head 1").click()
        expect(page.locator('.head[aria-pressed="true"]')).to_have_attribute("aria-label", "Layer 2, head 1")
        assert fills(page) != moved
        assert errors == []
    finally:
        page.close()
