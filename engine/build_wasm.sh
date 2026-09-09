#!/usr/bin/env bash
set -euo pipefail
# Requires: source /tmp/gipf-emsdk/emsdk_env.sh
repo_dir=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$repo_dir/web"
em++ "$repo_dir/engine/gipf_engine.cpp" -DGIPF_WASM -std=c++17 -O3 --bind \
  -s MODULARIZE=1 -s EXPORT_NAME=createGipfEngine \
  -s ENVIRONMENT=web,worker,node -s ALLOW_MEMORY_GROWTH=1 \
  -o "$repo_dir/web/gipf_engine.js"
