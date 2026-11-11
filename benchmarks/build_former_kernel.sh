#!/usr/bin/env bash
set -euo pipefail
# usage: build_former_kernel.sh out.mjs [extra em++ flags]
repo_dir=$(cd "$(dirname "$0")/.." && pwd)
out=$1
shift
em++ "$repo_dir/benchmarks/former_kernel.cpp" "$repo_dir/engine/former_infer.cpp" -std=c++17 -O3 -msimd128 "$@" \
  -s MODULARIZE=1 -s EXPORT_ES6=1 -s ENVIRONMENT=node -s ALLOW_MEMORY_GROWTH=1 \
  -s EXPORTED_FUNCTIONS=_malloc,_former_init,_former_forward -s EXPORTED_RUNTIME_METHODS=HEAPF32 -o "$out"
