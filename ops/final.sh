#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/gipf
export PYTHONPATH=/workspace/gipf
exec /venv/main/bin/python -u -m ops.final_training --source "$GIPF_FINAL_SOURCE" --run runs/final --simulations 128 --lr .001 --deadline 1788988500
