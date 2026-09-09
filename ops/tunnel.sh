#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/gipf
export PYTHONPATH=/workspace/gipf
pty /opt/instance-tools/bin/cloudflared tunnel --url http://127.0.0.1:17100 --no-autoupdate 2>&1
