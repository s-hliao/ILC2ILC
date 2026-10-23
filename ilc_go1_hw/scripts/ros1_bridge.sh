#!/bin/bash
# The ROS 1 (Go1) <-> ROS 2 (jump controller) bridge, on the WORKSTATION wired to the robot (192.168.123.x).
#   bash ros1_bridge.sh <go1 ip, e.g. 192.168.123.161> <this machine's ip on that network>
# ros:foxy-ros1-bridge carries noetic + foxy; foxy's DDS talks to the controller's humble for these message types
# (std_msgs, sensor_msgs, geometry_msgs, nav_msgs, std_srvs). The dynamic bridge carries the Go1's joint_states, IMU,
# leg_odom, ZUPT and contact topics to ROS 2, and joint_cmd_r / clear_fault toward ROS 1. NatNet stays entirely on the
# workstation; because ROS 1 has no subscriber for it by default, the dynamic bridge does not send it over Wi-Fi.
# --bridge-all-1to2-topics: every ROS 1 topic is bridged whether or not a ROS 2 subscriber is detected. Detection of
# the controller's subscriptions proved unreliable (joint_states silently not bridged), and `ros2 topic hz` never subscribes
# to a topic that has no publisher yet. Do not run Unitree's autostart nodes alongside: their topics would be bridged too.
set -e
GO1=${1:?go1 ip}; ME=${2:?this machine ip}
exec docker run --rm -it --net=host -e ROS_MASTER_URI=http://$GO1:11311 -e ROS_IP=$ME \
  -e ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0} osrf/ros:foxy-ros1-bridge \
  bash -lc "source /opt/ros/noetic/setup.bash && source /opt/ros/foxy/setup.bash && ros2 run ros1_bridge dynamic_bridge --bridge-all-1to2-topics"
