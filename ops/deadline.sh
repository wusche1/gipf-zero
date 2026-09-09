#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
pty /venv/main/bin/python -u /workspace/gipf/ops/deadline.py 2>&1
