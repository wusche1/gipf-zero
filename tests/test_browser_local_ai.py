"""Actual shipped ONNX + WASM play without an inference host or CDN."""
from urllib.parse import urlsplit
from test_browser_mapping import browser, site_url

def test_real_browser_ai_works_with_only_static_origin(browser, site_url):
    context=browser.new_context(viewport={'width':1000,'height':900},reduced_motion='reduce')
    external=[];ai_requests=[]
    def route(r):
        if '/api/' in r.request.url:ai_requests.append(r.request.url)
        if urlsplit(r.request.url).netloc==urlsplit(site_url).netloc:r.continue_()
        else:external.append(r.request.url);r.abort()
    context.route('**/*',route)
    page=context.new_page();page.set_default_timeout(30000)
    try:
        page.goto(site_url,wait_until='domcontentloaded')
        page.locator('#mode-select').select_option('ai')
        page.locator('#difficulty').select_option('casual')
        page.locator('.ray-hit[data-ray-id="1"]').click()
        page.wait_for_function("document.querySelector('#move-count').textContent === 'MOVE 02'")
        assert '43,869 games' in page.locator('#ai-status').text_content()
        assert not ai_requests
        page.locator('#undo').click()
        assert page.locator('#move-count').text_content()=='MOVE 00'
        page.locator('#mode-select').select_option('local')
        page.locator('.ray-hit[data-ray-id="1"]').click()
        assert page.locator('#move-count').text_content()=='MOVE 01'
    finally:context.close()

def test_model_download_retry_preserves_current_game(browser, site_url):
    context=browser.new_context(reduced_motion='reduce')
    page=context.new_page();page.set_default_timeout(30000)
    # This test deliberately aborts the first model fetch; keep the service
    # worker out of the path so the page route sees the worker request.
    page.add_init_script("""
      if (navigator.serviceWorker) {
        navigator.serviceWorker.register = () => Promise.reject(new Error('service worker disabled for retry fixture'));
      }
    """)
    attempts=[]
    def model_route(route):
        attempts.append(route.request.url)
        if len(attempts)==1:route.abort()
        else:route.continue_()
    page.route('**/models/champion.onnx*',model_route)
    try:
        page.goto(site_url,wait_until='domcontentloaded')
        page.locator('#mode-select').select_option('ai')
        page.locator('#difficulty').select_option('casual')
        page.locator('.ray-hit[data-ray-id="1"]').click()
        page.locator('#retry-ai').wait_for(state='visible')
        assert page.locator('#move-count').text_content()=='MOVE 01'
        page.locator('#retry-ai').click()
        page.wait_for_function("document.querySelector('#move-count').textContent === 'MOVE 02'")
        assert len(attempts)==2
        assert page.locator('#retry-ai').is_hidden()
    finally:context.close()
