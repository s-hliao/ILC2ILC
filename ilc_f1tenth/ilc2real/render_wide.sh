#!/bin/bash
# render the car videos + trajectory figures from archive_mu02_wide/trajectories as the recordings land
cd "$(dirname "$0")"
PY=~/miniconda3/envs/f1t/bin/python
R=archive_mu02_wide
until [ "$(ls $R/trajectories/eval_*.npz 2>/dev/null | wc -l)" -ge 13 ]; do sleep 30; done
$PY make_car_videos.py --root $R --only eval --jobs 4
$PY make_car_videos.py --root $R --only compare --jobs 4
MPLBACKEND=Agg $PY car_traj_figs.py --root $R --tag "original tracks, mu 0.2"
until grep -q RECORD_WIDE_DONE logs/record_wide.log; do sleep 30; done
$PY make_car_videos.py --root $R --only hwstage --jobs 4
MPLBACKEND=Agg $PY car_traj_figs.py --root $R --tag "original tracks, mu 0.2"
echo "$(date +%T) RENDER_WIDE_DONE"
