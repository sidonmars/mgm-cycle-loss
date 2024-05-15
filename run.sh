#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=${PY:-python}
RUNS=runs
mkdir -p "$RUNS"
log() { echo "$(date '+%F %T') $*"; }

train() {
  name=$1; shift
  [ -f "$RUNS/$name/last_state.pt" ] && return
  resume=""; [ -f "$RUNS/$name/checkpoint.pt" ] && resume="--resume $RUNS/$name/checkpoint.pt"
  log "train $name"; $PY -u train.py train --output "$RUNS/$name" "$@" $resume >> "$RUNS/$name.log" 2>&1 || { log "train $name failed"; exit 1; }
}

finish() {
  name=$1
  [ -f "$RUNS/$name/dustbin/dustbin_state.pt" ] || { log "dustbin $name"
    $PY -u train.py fit-dustbin --checkpoint "$RUNS/$name/last_state.pt" --output "$RUNS/$name/dustbin" > "$RUNS/$name.dustbin.log" 2>&1; }
  [ -f "$RUNS/$name/eval/results.json" ] || { log "evaluate $name"
    $PY -u evaluate.py --checkpoint "$RUNS/$name/dustbin/dustbin_state.pt" --output "$RUNS/$name/eval" > "$RUNS/$name.eval.log" 2>&1; }
}

( train s123_3k --steps 3000 --seed 123 --matcher-seed 0
  train s123 --steps 6000 --seed 123 --matcher-seed 0 --resume "$RUNS/s123_3k/last_state.pt"
  finish s123 ) &
( train s7 --steps 6000 --seed 7 --matcher-seed 7; finish s7 ) &
( train s99 --steps 6000 --seed 99 --matcher-seed 99; finish s99 ) &
wait

[ -f "$RUNS/seed_selection.json" ] || $PY -u train.py select-seed --runs "$RUNS/s123" "$RUNS/s7" "$RUNS/s99" \
  --output "$RUNS/seed_selection.json" > "$RUNS/seed_selection.log" 2>&1

$PY - "$RUNS" <<'EOF' | tee "$RUNS/summary.txt"
import json, sys
from pathlib import Path
import numpy as np
runs = Path(sys.argv[1]); picked = json.loads((runs / 'seed_selection.json').read_text())['selected']
rows = []
for name in ('s123', 's7', 's99'):
    r = json.loads((runs / name / 'eval' / 'results.json').read_text())
    rows.append((name, 100 * r['spair_filtered']['macro_class_accuracy'], 100 * r['voc_filtered']['macro_class_accuracy'],
                 100 * r['voc_unfiltered']['macro_class_f1']))
print('%-12s %7s %7s %9s' % ('run', 'SPair', 'VOC', 'VOC F1'))
for r in rows:
    print('%-12s %7.1f %7.1f %9.1f%s' % (*r, '   <- selected by cycle violation' if r[0] == picked else ''))
a = np.array([r[1:] for r in rows])
print('%-12s %s' % ('mean +- sd', '  '.join('%.1f+-%.1f' % (m, s) for m, s in zip(a.mean(0), a.std(0, ddof=1)))))
EOF
