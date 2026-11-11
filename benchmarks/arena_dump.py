# python -m benchmarks.arena_dump a.pt b.pt  ->  /workspace/tmp/gipf-work/arena/{a,b}.{bin,json}
import json
import sys
from pathlib import Path

from ops.dump_former import dump
from training import net

out = Path('/workspace/tmp/gipf-work/arena')
out.mkdir(parents=True, exist_ok=True)
for checkpoint in sys.argv[1:]:
    model, name = net.load(checkpoint)[0], Path(checkpoint).stem
    block = model.blocks[0]
    dump(model).tofile(out / f'{name}.bin')
    (out / f'{name}.json').write_text(json.dumps({'width': model.embed.out_features, 'blocks': len(model.blocks), 'heads': block.h,
                                                  'ff': block.f1.out_features, 'relu': False, 'int16': False}))
    print(out / f'{name}.bin', (out / f'{name}.bin').stat().st_size)
