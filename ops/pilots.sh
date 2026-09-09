#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/gipf
export PYTHONPATH=/workspace/gipf
set -o pipefail
for config in 'mlp 128 2 64 mlp128 flat' 'mlp 256 2 64 mlp256 flat' 'resnet 32 2 32 resnet32 flat'; do
    read -r kind width blocks simulations name head <<< "$config"
    echo "PILOT START $name $(date -u +%FT%TZ)"
    timeout -s TERM -k 30 980 /venv/main/bin/python -u -m training.train --run "runs/pilot_$name" --head "$head" --kind "$kind" --width "$width" --blocks "$blocks" --simulations "$simulations" --games 64 --games-per-iteration 32 --updates 32 --batch-size 256 --seconds 900 --max-ply 240 --seed 101 || exit $?
    for opponent in random greedy; do
        timeout -s TERM -k 10 330 /venv/main/bin/python -u -m training.evaluate --checkpoint "runs/pilot_$name/latest.pt" --opponent "$opponent" --games 40 --batch 32 --simulations 128 --seconds 300 --output "reports/pilot_$name-vs-$opponent.json" || exit $?
    done
    echo "PILOT COMPLETE $name $(date -u +%FT%TZ)"
done
