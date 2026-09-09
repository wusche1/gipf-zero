#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/gipf
pty /venv/main/bin/python -u ops/sync_endpoint.py 2>&1
