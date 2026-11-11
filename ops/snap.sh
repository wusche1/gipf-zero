#!/bin/bash
# usage: ops/snap.sh <tag>  -- snapshot current weights of boot + followers and push them to the eval box
set -e
tag=$1; L=/workspace/tmp/gipf-work/snap/$tag; mkdir -p $L
ops/vast.sh 60200805 pull /root/runs/boot/net/weights.pt $L/boot.pt
for f in hex64x6-r8-v2 hex96x10-r4-v2 former96x6-r4-v2; do ops/vast.sh 60203014 pull /root/runs/f-$f/net/weights.pt $L/$f.pt; done
ops/vast.sh 60211582 ssh "mkdir -p /root/snap/$tag"
ops/vast.sh 60211582 push "$L/" /root/snap/$tag/
/workspace/tmp/gipf-work/venv/bin/python -c "
import torch,glob
for f in sorted(glob.glob('$L/*.pt')):
    d=torch.load(f,map_location='cpu',weights_only=False);print(f.split('/')[-1],d['config'],'steps',d['steps'],'pos',d['positions'])"
