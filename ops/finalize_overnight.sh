#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/gipf
export PYTHONPATH=/workspace/gipf
exec /venv/main/bin/python -u -m ops.finalize_overnight
