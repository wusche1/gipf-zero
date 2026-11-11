#!/bin/bash
# usage: archstudy.sh <data> <out> <steps> <parallel> kind:width:blocks...
data=$1; out=$2; steps=$3; par=$4; shift 4
cd /root/gipf; mkdir -p $out
printf '%s\n' "$@" | xargs -P $par -I{} bash -c "IFS=: read k w b <<< {}; python -m training.learn --data $data --out $out/\$k-\$w-\$b --kind \$k --width \$w --blocks \$b --steps $steps --val 200000 --window 8000000 > $out/\$k-\$w-\$b.log 2>&1"
