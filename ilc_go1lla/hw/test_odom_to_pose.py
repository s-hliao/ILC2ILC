#!/usr/bin/env python3
"""ROS test helper (2026-11-21): ilc_quad sim_node's base_odom -> a mocap-style PoseStamped on /dl_go1/pose, so walker_node.py
can be tested against sim_node on the workstation (run walker_node with --mocap-offset 0 0 0 --mocap-body-yaw-deg 0)."""
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

Q = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1,
               durability=QoSDurabilityPolicy.VOLATILE)


class Relay(Node):
    def __init__(self):
        super().__init__('odom_to_pose')
        self.pub = self.create_publisher(PoseStamped, '/dl_go1/pose', Q)
        self.create_subscription(Odometry, 'base_odom', self.cb, Q)

    def cb(self, m):
        p = PoseStamped()
        p.header = m.header
        p.pose = m.pose.pose
        self.pub.publish(p)


rclpy.init(); n = Relay(); rclpy.spin(n)
