#!/bin/bash
# Desk test of net_runner.py over ROS 2 (inside the f1t container): car_sim_node.py stands in for the car + EKF, a session
# dir carries the requests. Trial A: a trained network for 2.05 laps (must finish, small e_y); trial B: a zeroed network
# (started 0.45 m left of the path, then a constant 0.1 rad left at 3 A after the launch: away from the path) must stop on
# the 0.5 m envelope ('wall') and brake.
#   bash desk_test_runner.sh <car> <policy.npz> [vesc_msgs install setup.bash]
#   DDS_ROLE=local bash desk_test_runner.sh ...     (the same over CycloneDDS, as the remote setup runs)
set -e
CAR=${1:-real_nom}; POL=${2:?policy npz}; VESC=${3:-/tmp/f1t_scratch/install/setup.bash}
H=$(cd "$(dirname "$0")" && pwd); I=$H/../../ilc_f1tenth/ilc2real
source /opt/ros/${ROS_DISTRO:-$(ls /opt/ros | head -1)}/setup.bash; source $VESC; source $I/env_tight.sh
export ROS_DOMAIN_ID=77
if [ -n "$DDS_ROLE" ]; then source $H/dds_env.sh $DDS_ROLE; export ROS_DOMAIN_ID=77   # e.g. DDS_ROLE=local: CycloneDDS
else export ROS_LOCALHOST_ONLY=1; fi
S=/tmp/net_sess_$$; mkdir -p $S
python3 - "$POL" "$S/zero.npz" <<'PY'
import sys, numpy as np
d = np.load(sys.argv[1]); z = {k: np.zeros_like(d[k]) for k in d.files if k[0] in 'Wb'}
last = max(int(k[1:]) for k in z if k[0] == 'b')
z[f'b{last}'] = np.array([0.1 / 0.34, (3.0 - 12.5) / 37.5])   # a constant command: 0.1 rad to the left, 3 A
np.savez(sys.argv[2], **z)
PY
run_trial() {  # name plan policy laps [start offset m]
  python3 $H/car_sim_node.py --ros-args -p car:=$CAR -p plan:=$2 -p start_offset:=${5:-0.0} > $S/sim_$1.log 2>&1 & SIM=$!
  python3 $H/net_runner.py --ros-args -p session_dir:=$S -p stdin_start:=false > $S/runner_$1.log 2>&1 & RUN=$!
  # wait for the stand-in car's odometry (its multi-body model JIT-compiles first: ~2 s with a warm numba cache, ~10 s
  # cold, e.g. a fresh container) instead of a fixed sleep -- a start request before it is refused ('no fresh odometry')
  timeout 90 ros2 topic echo /odometry/filtered 2>/dev/null | head -n 1 > /dev/null || true   # (foxy has no --once)
  sleep 2
  echo "{\"plan\": \"$2\", \"policy\": \"$3\", \"laps\": $4, \"launch\": {\"v_hand\": 1.0, \"current\": 8.0}}" > $S/req_$1.json
  sleep 2
  timeout 20 ros2 service call /net_runner/start std_srvs/srv/Trigger > $S/start_$1.log 2>&1 || true
  for i in $(seq 1 120); do [ -f $S/res_$1.json ] && break; sleep 0.5; done
  sleep 1.5
  kill $RUN $SIM 2>/dev/null || true; wait $RUN $SIM 2>/dev/null || true
  echo "== trial $1 ($2)"; grep -h "success\|message" $S/start_$1.log | head -2
  cat $S/res_$1.json 2>/dev/null || { echo "NO RESULT"; tail -5 $S/runner_$1.log; }
  grep -o "trial $1 [^\"]*" $S/runner_$1.log | tail -1 || true
}
run_trial a mocap_square2fast_b25 $POL 2.05
run_trial b mocap_square2fast_b25 $S/zero.npz 2.05 0.45
python3 - $S <<'PY'
import sys, numpy as np
for k in ('a', 'b'):
    d = np.load(f'{sys.argv[1]}/res_{k}.npz', allow_pickle=True)
    h = int(d['handover']); e = d['err'][:, 0]
    print(k, 'steps', int(d['steps']), 'handover', h, 'laps %.2f' % float(d['laps']), 'rms e_y after launch %.3f m' % float(np.sqrt(np.mean(e[h:] ** 2))),
          'peak |e_y| %.3f' % float(np.max(np.abs(e))), 'crash', str(d['crash']), 'max cmd current %.1f' % float(np.max(d['applied'][:, 1])))
PY
echo "brake after B (last drive msgs: jerk 1 = brake):"; grep -c "STOPPED: wall" $S/runner_b.log || true
