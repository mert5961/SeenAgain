#!/bin/sh
# Full run on the desk scene: three phone videos in data/ (A = baseline, A2 = control, B = round with changes).
set -e
PY=.venv/bin/python
for w in A A2 B; do $PY -m seenagain.frames data/$w.mp4 --scene desk --walk $w --fps 5; done
$PY -m seenagain.sfm --scene desk
for w in A A2 B; do $PY -m seenagain.segment --scene desk --walk $w; $PY -m seenagain.sheet --scene desk --walk $w; done
MPU=0.0423   # from one measurement: the two glasses were ~40 cm apart in A (approximate)
for r in A2 B; do
  $PY -m seenagain.match --scene desk --round $r --params params_frozen.json --metres-per-unit $MPU
  $PY -m seenagain.match --scene desk --round $r --params params_frozen.json --metres-per-unit $MPU --v2
done
$PY -m seenagain.score --scene desk --gt gt_desk.json
$PY -m seenagain.viz --scene desk --receipt receipt_A_B_v2.json --metres-per-unit $MPU
