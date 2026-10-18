#!/usr/bin/python3
"""NatNet PoseStamped -> tracked pose and trunk Odometry for the workstation EKF."""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time

from ilc_quad.mocap import MocapProcessor


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class MocapOdomNode(Node):
    def __init__(self):
        super().__init__("mocap_odom")
        defaults = {
            "mocap_topic": "/dl_go1/pose", "up_axis": "z", "yaw_offset": 0.0,
            "mocap_offset": [0.0, 0.0, 0.0], "pose_latency": 0.006,
            "vel_ema_tau": 0.02, "vel_spike_pct": 15.0, "vel_spike_window": 0.3,
            "max_gap": 0.1, "pos_var": 1e-6, "rot_var": 1e-4,
            "lin_vel_var": 0.04, "ang_vel_var": 0.09,
            "world_frame": "world", "trunk_frame": "trunk",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        get = lambda name: self.get_parameter(name).value
        self.world, self.trunk = get("world_frame"), get("trunk_frame")
        self.latency = float(get("pose_latency"))
        self.proc = MocapProcessor(get("up_axis"), float(get("yaw_offset")), get("mocap_offset"),
                                   float(get("vel_ema_tau")), float(get("vel_spike_pct")),
                                   float(get("vel_spike_window")), float(get("max_gap")))
        self.pos_var, self.rot_var = float(get("pos_var")), float(get("rot_var"))
        self.lin_var, self.ang_var = float(get("lin_vel_var")), float(get("ang_vel_var"))
        self.pose_pub = self.create_publisher(PoseStamped, "/mocap/pose", 1)
        self.odom_pub = self.create_publisher(Odometry, "/mocap/odom", 10)
        topic = get("mocap_topic")
        self.create_subscription(PoseStamped, topic, self.on_pose, 10)
        self.get_logger().info(f"NatNet {topic} -> /mocap/pose, /mocap/odom")

    def on_pose(self, msg):
        source_time = Time.from_msg(msg.header.stamp)
        t = stamp_seconds(msg.header.stamp)
        if t <= 0:
            source_time = self.get_clock().now()
            t = source_time.nanoseconds * 1e-9
        o = msg.pose.orientation
        p, origin, q, twist = self.proc.update(
            t, [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], [o.x, o.y, o.z, o.w])
        pose = PoseStamped()
        pose.header.stamp, pose.header.frame_id = source_time.to_msg(), self.world
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = p
        pose.pose.orientation.x, pose.pose.orientation.y = q[0], q[1]
        pose.pose.orientation.z, pose.pose.orientation.w = q[2], q[3]
        self.pose_pub.publish(pose)

        odom = Odometry()
        odom.header.stamp = (source_time - Duration(seconds=self.latency)).to_msg()
        odom.header.frame_id, odom.child_frame_id = self.world, self.trunk
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = origin
        odom.pose.pose.orientation = pose.pose.orientation
        for i in range(3):
            odom.pose.covariance[7*i], odom.pose.covariance[7*(i+3)] = self.pos_var, self.rot_var
        if twist is not None:
            values = list(np.asarray(twist, float))
            (odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z,
             odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z) = values
            for i in range(3):
                odom.twist.covariance[7*i], odom.twist.covariance[7*(i+3)] = self.lin_var, self.ang_var
        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = MocapOdomNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
