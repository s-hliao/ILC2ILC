#!/bin/bash
# car_runner.sh (2026-11-04), ON THE CAR: net_runner for one tire's session (HW_RUNBOOK_N6_1031.md section 2), with the
# runbook's environment (env_tight.sh, then mu 0.6 and plans_nom06).
#   bash ~/car_runs/car_runner.sh plastic|rubber [MMDD]     in its own terminal, after car_stack.sh; keep it up all session
# Session folder ~/car_sessions/<MMDD>_<tire> (= /home/kathy/car_sessions/..., the same path the workstation mounts).
# Each request it announces: place the car at the start, then Enter here.
TIRE=$1; DAY=${2:-$(date +%m%d)}
case "$TIRE" in plastic|rubber) ;; *) echo "usage: bash ~/car_runs/car_runner.sh plastic|rubber [MMDD]"; exit 1;; esac
S=/home/kathy/car_sessions/${DAY}_$TIRE; mkdir -p $S
pgrep -f 'drive_relay\.py' >/dev/null || { echo "!!! drive_relay is not running: start bash ~/car_runs/car_stack.sh first"; exit 1; }
if pgrep -f '^python3 \S*net_runner\.py' >/dev/null; then echo "!!! a net_runner already runs: $(pgrep -af '^python3 \S*net_runner\.py')"; exit 1; fi
source /opt/ros/foxy/setup.bash; source ~/lla_ws/install/setup.bash; source ~/ILC2ILC/install/setup.bash
unset CYCLONEDDS_URI ROS_DOMAIN_ID ROS_LOCALHOST_ONLY; export RMW_IMPLEMENTATION=rmw_fastrtps_cpp PATH=/usr/bin:$PATH
export ILC2REAL_DIR=$HOME/ILC2ILC/ilc_f1tenth/ilc2real
source $ILC2REAL_DIR/env_tight.sh
export F1T_MU=0.6 F1T_PLANS=$ILC2REAL_DIR/plans_nom06 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1   # AFTER env_tight.sh
# the track as laid out in the room (2026-11-04): the same offset file slow_lap.sh / car_pos.sh read
read DX DY _ < <(head -1 ~/car_runs/track_offset.txt 2>/dev/null); DX=${DX:-0.0}; DY=${DY:-0.0}
echo ">>> net_runner: session $S, mu $F1T_MU, plans $F1T_PLANS, envelope $F1T_SAFETY_EY m, track offset ($DX, $DY) m"
echo ">>> keys here: Enter = start the announced trial | r + Enter = re-fly the last trial (tag <tag>_rerun, not scored)"
echo "                R + Enter = re-fly the last trial IN PLACE of it (the old one becomes VOID_<tag>, the re-run is scored)"
# the node itself, not the ros2 run wrapper (a killed wrapper can orphan its node)
exec python3 $HOME/ILC2ILC/install/ilc_f1tenth_hw/lib/ilc_f1tenth_hw/net_runner.py --ros-args -p session_dir:=$S -p "track_offset:=[$DX, $DY]"
