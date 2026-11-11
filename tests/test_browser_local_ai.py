"""Actual shipped model + WASM play without an inference host or CDN."""
import json
from urllib.parse import urlsplit
from test_browser_mapping import WEB, browser, site_url

CONFIG=json.loads((WEB/'config.json').read_text())
META=json.loads((WEB/f"models/{CONFIG['model']}.json").read_text())
CLASSIC=json.loads((WEB/f"models/{CONFIG['classic']}.json").read_text())

def test_real_browser_ai_works_with_only_static_origin(browser, site_url):
    # Without a service worker in between, the route sees every worker request.
    context=browser.new_context(viewport={'width':1000,'height':900},reduced_motion='reduce',service_workers='block')
    external=[];ai_requests=[];runtime=[]
    def route(r):
        if '/api/' in r.request.url:ai_requests.append(r.request.url)
        if '/vendor/onnxruntime/' in r.request.url or r.request.url.split('?')[0].endswith('.onnx'):runtime.append(r.request.url)
        if urlsplit(r.request.url).netloc==urlsplit(site_url).netloc:r.continue_()
        else:external.append(r.request.url);r.abort()
    context.route('**/*',route)
    page=context.new_page();page.set_default_timeout(30000)
    try:
        page.goto(site_url,wait_until='domcontentloaded')
        page.locator('#mode-select').select_option('ai')
        page.locator('#difficulty').select_option('fast')
        page.locator('.ray-hit[data-ray-id="1"]').click()
        page.wait_for_function("document.querySelector('#move-count').textContent === 'MOVE 02'")
        assert page.locator('#ai-status').text_content()==f"{META['name']} ready"
        assert 'pos/s' in page.locator('#think-readout').text_content()
        assert not ai_requests
        # A kernel model never downloads the ONNX runtime or an ONNX file.
        assert META['format']=='kernel' and not runtime
        page.locator('#undo').click()
        assert page.locator('#move-count').text_content()=='MOVE 00'
        # The earlier champion is one selection away and starts from a clean search.
        page.locator('#opponent').select_option(label='Classic')
        assert page.locator('#ai-status').text_content()=='Local model ready when needed'
        page.locator('.ray-hit[data-ray-id="1"]').click()
        page.wait_for_function("document.querySelector('#move-count').textContent === 'MOVE 02'")
        assert page.locator('#ai-status').text_content()==f"{CLASSIC['name']} ready"
        assert any(url.split('?')[0].endswith('ort.wasm.min.js') for url in runtime)
        page.locator('#opponent').select_option(label='Latest')
        page.locator('.ray-hit[data-ray-id="2"]').click()
        page.wait_for_function("document.querySelector('#move-count').textContent === 'MOVE 04'")
        assert page.locator('#ai-status').text_content()==f"{META['name']} ready"
        page.locator('#undo').click();page.locator('#undo').click()
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
