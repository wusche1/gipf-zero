#!/bin/bash
# usage: ops/vast.sh <instance_id> sync | ssh <cmd...> | pull <remote_path> <local_path>
set -e
export PATH=$HOME/.local/bin:$PATH
id=$1; shift
url=$(vastai ssh-url $id)
host=${url#ssh://}; port=${host##*:}; host=${host%%:*}
S="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p $port"
case $1 in
  sync) rsync -az -e "$S" --exclude .git --exclude reports --exclude web --exclude engine --exclude '*.so' --exclude build --exclude __pycache__ ./ $host:/root/gipf/ ;;
  syncb) rsync -az -e "$S" --exclude .git --exclude reports --exclude web --exclude '*.so' --exclude build --exclude __pycache__ ./ $host:/root/gipf/
        $S $host 'cd /root/gipf && pip install -q pybind11 && python setup.py -q build_ext --inplace && ls *.so' ;;
  ssh) shift; $S $host "$@" ;;
  pull) rsync -az -e "$S" $host:$2 $3 ;;
  push) rsync -az -e "$S" $2 $host:$3 ;;
esac
