"""Pin application asset references to one Pages deployment revision."""
import os,re
from pathlib import Path

def version_assets(web:Path,version:str):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',version):raise ValueError('invalid asset version')
    names=('styles.css','app.js','game.js','local-ai.js','ai-worker.js','gipf_engine.js','gipf_engine.wasm','benchmark.js','offline.js','service-worker.js')
    for path in [*web.glob('*.html'),web/'config.json',*(web/name for name in names if name.endswith('.js'))]:
        if not path.exists():continue
        text=path.read_text()
        text=text.replace('__ASSET_VERSION__',version)
        for name in names:
            text=re.sub(re.escape(name)+r'(?:\?v=[A-Za-z0-9_-]+)?(?=[\'\"])',name+'?v='+version,text)
        path.write_text(text)
if __name__=='__main__':version_assets(Path('web'),os.environ['ASSET_VERSION'])
