#!/bin/bash
# car_stack.sh (2026-11-04), ON THE CAR: the car's hardware stack for the net_runner sessions (HW_RUNBOOK_N6_1031.md):
# f1tenth_system (VESC), the NatNet client, ZUPT, optitrack_node, the EKF -> /odometry/filtered, AND drive_relay.py.
#   bash ~/car_runs/car_stack.sh [extra launch args]      in its own terminal; Ctrl-C stops the stack
# Differs from ~/ILC2ILC/run_opti.sh in two things only:
#  - client_ip is this car's CURRENT address (the one it uses to reach Motive), found at start: the WiFi address changes
#    between sessions; run_opti.sh hardcodes 172.26.50.69.
#  - drive_relay:=true: net_runner publishes /mpc/drive and only drive_relay.py forwards it to /drive; its watchdog is
#    also what keeps braking after a trial stops (net_runner header). run_opti.sh's drive_relay:=false is for
#    manual_policy.py, which publishes /drive itself (do not run both).
MOTIVE=172.26.30.22
if pgrep -f 'ros2 launch ilc_f1tenth_hw opti_launch.py' >/dev/null; then
  echo "!!! an opti_launch stack already runs: $(pgrep -af 'ros2 launch ilc_f1tenth_hw opti_launch.py' | cut -c1-120)"; exit 1; fi
if pgrep -f '^python3 \S*manual_policy\.py' >/dev/null; then echo "!!! manual_policy.py runs (it publishes /drive itself): stop it first"; exit 1; fi
ME=$(ip -4 route get $MOTIVE 2>/dev/null | grep -oP 'src \K\S+')
[ -n "$ME" ] || { echo "!!! no route to Motive ($MOTIVE): WiFi up?"; exit 1; }
ping -c 2 -W 1 $MOTIVE >/dev/null || echo "  (warning: Motive $MOTIVE does not answer ping; NatNet may still work)"
source /opt/ros/foxy/setup.bash; source ~/lla_ws/install/setup.bash; source ~/ILC2ILC/install/setup.bash
unset CYCLONEDDS_URI ROS_DOMAIN_ID ROS_LOCALHOST_ONLY; export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
echo ">>> car stack: Motive $MOTIVE, this car $ME, drive_relay on"
# NatNet watchdog (2026-11-04): the NatNet SDK can abort inside Connect() (a race in its own threads); launch respawns the
# node but never configures it. natnet_watchdog.py configures + activates it whenever mocap is missing. It is started here,
# becomes a child of the launch below (exec keeps this PID), and exits with it.
python3 ~/car_runs/natnet_watchdog.py &
exec ros2 launch ilc_f1tenth_hw opti_launch.py server_ip:=$MOTIVE client_ip:=$ME drive_relay:=true "$@"
