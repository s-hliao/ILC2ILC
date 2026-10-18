#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Refuse duplicate hardware stacks; do not stop the existing processes.
if pgrep -af '[r]os2 launch ilc_f1tenth_hw opti_launch.py'; then
  echo 'ERROR: an opti_launch hardware stack is already running. Stop its terminal with Ctrl+C before launching another.' >&2
  exit 1
fi
source /opt/ros/foxy/setup.bash
source "$PWD/install/setup.bash"
unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
echo "Onboard ROS 2: $RMW_IMPLEMENTATION; CycloneDDS configuration cleared."
exec ros2 launch ilc_f1tenth_hw opti_launch.py server_ip:=172.26.30.22 client_ip:=172.26.50.69 drive_relay:=false "$@"
