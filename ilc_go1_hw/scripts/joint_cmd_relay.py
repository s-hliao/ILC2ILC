#!/usr/bin/env python3
"""ROS 2 side (the workstation / ilc_quad container): joint_cmd (BEST_EFFORT, as the jump controllers publish it) ->
joint_cmd_r (RELIABLE). ros1_bridge's ROS 2 subscriptions are RELIABLE, and DDS never matches a RELIABLE reader with a
BEST_EFFORT writer, so without this relay no command would reach the robot. The other direction needs nothing (a
RELIABLE writer reaches the controllers' BEST_EFFORT readers). go1_hw.launch remaps the robot's joint_cmd to joint_cmd_r.
    python3 joint_cmd_relay.py"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float64MultiArray


class Relay(Node):
    def __init__(self):
        super().__init__("joint_cmd_relay")
        be = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        rel = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(Float64MultiArray, "joint_cmd_r", rel)
        self.create_subscription(Float64MultiArray, "joint_cmd", self.pub.publish, be)
        self.get_logger().info("joint_cmd (best effort) -> joint_cmd_r (reliable) for ros1_bridge")


def main():
    rclpy.init()
    rclpy.spin(Relay())


if __name__ == "__main__":
    main()
