#!/bin/bash
# usage: evalloop.sh <run_dir> <opponent.pt> <sims> <games> <period_s> <seed>
cd /root/gipf
while true; do
  cp $1/net/weights.pt $1/eval-w$6.pt
  pos=$(python -c "import torch;print(torch.load('$1/eval-w$6.pt',weights_only=False)['positions'])")
  python -m training.arena --a $1/eval-w$6.pt --b $2 --sims-a $3 --sims-b $3 --games $4 --seed $6 | sed "s/^{/{\"positions\": $pos, /" >> $1/eval.jsonl
  sleep $5
done
