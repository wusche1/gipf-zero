#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/gipf
export PYTHONPATH=/workspace/gipf
exec /venv/main/bin/python -u -m ops.final_training --source runs/pilot_mlp256/latest.pt --run runs/final_mlp --simulations 64 --lr .001 --native-forest --deadline 1788988500
