import json,time
from pathlib import Path
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright
url='https://wusche1.github.io/gipf-zero/'
result={'url':url,'external_requests':[],'errors':[],'checks':[]}
with sync_playwright() as p:
 b=p.chromium.launch(headless=True)
 c=b.new_context(service_workers='allow',reduced_motion='reduce')
 def route(r):
  if urlsplit(r.request.url).netloc=='wusche1.github.io':r.continue_()
  else:result['external_requests'].append(r.request.url);r.abort()
 c.route('**/*',route)
 page=c.new_page();page.set_default_timeout(60000)
 page.on('pageerror',lambda e:result['errors'].append(str(e)))
 page.goto(url,wait_until='domcontentloaded')
 page.wait_for_function("document.querySelector('#engine-status').textContent==='Ready to play'")
 page.locator('#mode-select').select_option('ai');page.locator('#difficulty').select_option('casual')
 page.locator('.ray-hit[data-ray-id="1"]').click()
 page.wait_for_function("document.querySelector('#move-count').textContent==='MOVE 02'")
 result['checks'].append('cold public AI reply');result['ai_status']=page.locator('#ai-status').text_content()
 page.locator('#undo').click();assert page.locator('#move-count').text_content()=='MOVE 00'
 result['checks'].append('undo')
 page.evaluate('navigator.serviceWorker.ready.then(()=>true)')
 page.goto(url+'benchmark.html',wait_until='domcontentloaded')
 assert page.locator('h1').text_content()=='How fast is your device?'
 result['checks'].append('benchmark navigation')
 c.set_offline(True)
 page.reload(wait_until='domcontentloaded');assert page.locator('h1').text_content()=='How fast is your device?'
 page.goto(url,wait_until='domcontentloaded')
 page.wait_for_function("document.querySelector('#engine-status').textContent==='Ready to play'")
 page.locator('#mode-select').select_option('ai');page.locator('#difficulty').select_option('casual')
 page.locator('.ray-hit[data-ray-id="1"]').click()
 page.wait_for_function("document.querySelector('#move-count').textContent==='MOVE 02'")
 result['checks'].append('offline reload and AI reply')
 assert not result['errors']
 c.close();b.close()
print(json.dumps(result,indent=2))
open(Path(__file__).resolve().parents[1]/'reports/browser-public-deployment.json','w').write(json.dumps(result,indent=2)+'\n')
