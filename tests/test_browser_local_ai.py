"""Actual shipped model + WASM play without an inference host or CDN."""
import json
from urllib.parse import urlsplit
from test_browser_mapping import WEB, browser, site_url

CONFIG=json.loads((WEB/'config.json').read_text())
META=json.loads((WEB/f"models/{CONFIG['model']}.json").read_text())

def test_real_browser_ai_works_with_only_static_origin(browser, site_url):
    # Without a service worker in between, the route sees every worker request.
    context=browser.new_context(viewport={'width':1000,'height':900},reduced_motion='reduce',service_workers='block')
    external=[];ai_requests=[];same_origin=[]
    def route(r):
        if '/api/' in r.request.url:ai_requests.append(r.request.url)
        if urlsplit(r.request.url).netloc==urlsplit(site_url).netloc:same_origin.append(urlsplit(r.request.url).path);r.continue_()
        else:external.append(r.request.url);r.abort()
    context.route('**/*',route)
    page=context.new_page();page.set_default_timeout(30000)
    try:
        page.goto(site_url,wait_until='domcontentloaded')
        page.locator('#engine-status').get_by_text('Ready to play').wait_for()
        page.locator('#mode-select').select_option('ai')
        page.locator('#difficulty').select_option('fast')
        page.locator('.ray-hit[data-ray-id="1"]').click()
        page.wait_for_function("document.querySelector('#move-count').textContent === 'MOVE 02'")
        assert page.locator('#ai-status').text_content()==f"{META['name']} ready"
        assert 'pos/s' in page.locator('#think-readout').text_content()
        assert not ai_requests
        # The kernel model is the engine plus one weight file: no other runtime.
        assert META['format']=='kernel' and f"/models/{META['path']}" in same_origin
        assert not [path for path in same_origin if '/vendor/' in path or path.endswith(('.onnx','.mjs'))]
        assert page.locator('#opponent').count()==0
        page.locator('#undo').click()
        assert page.locator('#move-count').text_content()=='MOVE 00'
        page.locator('.ray-hit[data-ray-id="2"]').click()
        page.wait_for_function("document.querySelector('#move-count').textContent === 'MOVE 02'")
        assert page.locator('#ai-status').text_content()==f"{META['name']} ready"
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
    page.route(f"**/models/{META['path']}*",model_route)
    try:
        page.goto(site_url,wait_until='domcontentloaded')
        page.locator('#mode-select').select_option('ai')
        page.locator('#difficulty').select_option('fast')
        page.locator('.ray-hit[data-ray-id="1"]').click()
        page.locator('#retry-ai').wait_for(state='visible')
        assert page.locator('#move-count').text_content()=='MOVE 01'
        page.locator('#retry-ai').click()
        page.wait_for_function("document.querySelector('#move-count').textContent === 'MOVE 02'")
        assert len(attempts)==2
        assert page.locator('#retry-ai').is_hidden()
    finally:context.close()
