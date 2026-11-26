#!/usr/bin/env bash
# Runs the remaining experiments one after another (each ~11 min on an RTX 4060 laptop GPU).
set -e
PY=../.venv/Scripts/python.exe
run() { out=$1; shift; mkdir -p "$out"; echo "=== $out"; $PY -u train_multi.py --out_dir "$out" --epochs 8 --workers 6 "$@" > "$out/train.log" 2>&1; grep -E '^\| ' "$out/train.log"; }
run results_multi/mirror --mirror_p 0.5
run results_multi/wildA  --mirror_p 0.5 --wild_train A
run results_multi/wildB  --mirror_p 0.5 --wild_train B
run results_multi/final  --mirror_p 0.5 --wild_train all
echo "ALL DONE"
