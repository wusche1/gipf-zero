"""Exercise actual checkpoint restart, including CUDA-mapped RNG tensors."""
import json
import os
import subprocess
import sys
import torch
import pytest
from training.model import PolicyValue, ModelConfig

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_training_resume_restores_counters_and_rng_on_target_device(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    source=tmp_path/'source';source.mkdir()
    model=PolicyValue(ModelConfig('mlp',16,1))
    torch.save({'model':model.state_dict(),'config':{'kind':'mlp','width':16,'blocks':1},
                'games':7,'iteration':3,'updates':9,'torch_rng':torch.get_rng_state()},source/'latest.pt')
    target=tmp_path/'resumed'
    result=subprocess.run([sys.executable,'-m','training.train','--run',str(target),
        '--resume',str(source/'latest.pt'),'--device',device,'--kind','mlp','--width','16',
        '--blocks','1','--games','2','--seconds','0'],capture_output=True,text=True,timeout=40)
    assert result.returncode==0, result.stderr
    info=torch.load(target/'latest.pt',map_location='cpu',weights_only=False)
    assert (info['games'],info['iteration'],info['updates'])==(7,3,9)
    assert info['torch_rng'].dtype==torch.uint8
    heartbeat=json.loads((target/'heartbeat.json').read_text())
    assert heartbeat['event']=='stopped' and heartbeat['reason']=='deadline'
