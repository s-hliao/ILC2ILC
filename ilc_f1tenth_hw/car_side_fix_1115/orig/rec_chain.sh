#!/bin/bash
# rec_chain.sh [seconds = 120] (2026-11-05): read-only recording of the car state chain (mocap, optitrack_node, EKF, IMU, VESC)
source /opt/ros/foxy/setup.bash; source ~/lla_ws/install/setup.bash
unset ROS_DOMAIN_ID ROS_LOCALHOST_ONLY CYCLONEDDS_URI; export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
exec python3 ~/car_runs/rec_chain.py "${1:-120}" 2>&1 | grep -v Deprecat
