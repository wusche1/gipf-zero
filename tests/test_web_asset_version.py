from ops.version_web_assets import version_assets

def test_versions_worker_and_engine_dependencies_together(tmp_path):
    (tmp_path/'index.html').write_text('<script src="app.js?v=old"></script>')
    (tmp_path/'app.js').write_text("import './local-ai.js'; import './game.js';")
    (tmp_path/'local-ai.js').write_text("new Worker(new URL('./ai-worker.js',import.meta.url));")
    (tmp_path/'ai-worker.js').write_text("asset('gipf_engine.js'); asset('vendor/onnxruntime/ort.wasm.min.js');")
    (tmp_path/'gipf_engine.js').write_text("locateFile('gipf_engine.wasm')")
    version_assets(tmp_path,'abc123')
    assert 'app.js?v=abc123' in (tmp_path/'index.html').read_text()
    assert "'./local-ai.js?v=abc123'" in (tmp_path/'app.js').read_text()
    assert "'./ai-worker.js?v=abc123'" in (tmp_path/'local-ai.js').read_text()
    assert "'gipf_engine.wasm?v=abc123'" in (tmp_path/'gipf_engine.js').read_text()
    assert "'vendor/onnxruntime/ort.wasm.min.js'" in (tmp_path/'ai-worker.js').read_text()
    version_assets(tmp_path,'second')
    assert 'abc123' not in (tmp_path/'local-ai.js').read_text()
