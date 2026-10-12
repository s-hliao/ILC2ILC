#!/bin/bash
# The ROS 1 (Go1) <-> ROS 2 (jump controller) bridge, on the WORKSTATION wired to the robot (192.168.123.x).
#   bash ros1_bridge.sh <go1 ip, e.g. 192.168.123.161> <this machine's ip on that network>
# ros:foxy-ros1-bridge carries noetic + foxy; foxy's DDS talks to the controller's humble for these message types
# (std_msgs, sensor_msgs, geometry_msgs, nav_msgs, std_srvs). Bridged: joint_states, foot_contacts, imu, contact_state,
# mocap/pose, trunk/odometry (robot -> controller); joint_cmd_r (controller -> robot, via joint_cmd_relay.py); clear_fault.
# Every hop adds latency: measure the END-TO-END mocap latency through this bridge (README, "Latency") and give that to the
# controller as pose_latency.
set -e
GO1=${1:?go1 ip}; ME=${2:?this machine ip}
exec docker run --rm -it --net=host -e ROS_MASTER_URI=http://$GO1:11311 -e ROS_IP=$ME \
  -e ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0} ros:foxy-ros1-bridge \
  bash -lc "source /opt/ros/noetic/setup.bash && source /opt/ros/foxy/setup.bash && ros2 run ros1_bridge dynamic_bridge --bridge-all-topics"
