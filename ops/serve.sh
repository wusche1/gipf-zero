#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/gipf
export PYTHONPATH=/workspace/gipf
pty /venv/main/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 17100 --limit-concurrency 32 --timeout-keep-alive 5 2>&1
