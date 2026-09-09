#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/gipf
export PYTHONPATH=/workspace/gipf
set -o pipefail
timeout -s TERM -k 30 980 /venv/main/bin/python -u -m training.train --run runs/pilot_factorized --kind mlp --head factorized --width 128 --blocks 2 --simulations 64 --games 64 --games-per-iteration 32 --updates 32 --batch-size 256 --seconds 900 --max-ply 240 --seed 101 || exit $?
for opponent in random greedy; do
    timeout -s TERM -k 10 330 /venv/main/bin/python -u -m training.evaluate --checkpoint runs/pilot_factorized/latest.pt --opponent "$opponent" --games 40 --batch 32 --simulations 128 --seconds 300 --output "reports/pilot_factorized-vs-$opponent.json" || exit $?
done
