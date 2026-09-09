#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/gipf
export PYTHONPATH=/workspace/gipf
set -euo pipefail
timeout -s TERM -k 30 980 /venv/main/bin/python -u -m training.train --run runs/pilot_deeper --resume runs/pilot_mlp128/latest.pt --games 128 --games-per-iteration 64 --updates 64 --simulations 128 --batch-size 256 --seconds 900 --max-ply 240 --seed 107
cp runs/pilot_deeper/latest.pt checkpoints/pilot-deeper-final.pt
for opponent in random greedy; do
    timeout -s TERM -k 10 430 /venv/main/bin/python -u -m training.evaluate --checkpoint checkpoints/pilot-deeper-final.pt --opponent "$opponent" --games 40 --simulations 128 --seconds 400 --output "reports/pilot_deeper-vs-$opponent.json"
done
