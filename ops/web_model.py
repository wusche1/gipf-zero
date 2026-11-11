import argparse
import hashlib
import json
from pathlib import Path

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('--title', required=True)
    p.add_argument('--cpuct', type=float, default=1.5)
    p.add_argument('--fpu', type=float, default=-1.0)
    p.add_argument('--budgets', type=int, nargs=3, default=[300, 1000, 3000])
    p.add_argument('--web', type=Path, default=Path('web'))
    a = p.parse_args()
    models = a.web / 'models'
    meta = {'name': a.title, 'path': f'{a.name}.onnx', 'sha256': hashlib.sha256((models / f'{a.name}.onnx').read_bytes()).hexdigest(),
            'cpuct': a.cpuct, 'fpu': a.fpu, 'budgetsMs': dict(zip(('fast', 'normal', 'strong'), a.budgets)),
            'ponder': {'maxMs': 20000, 'maxVisits': 20000}}
    (models / f'{a.name}.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')
