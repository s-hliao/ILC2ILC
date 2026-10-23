#!/usr/bin/python3
"""Put bridged Go1 sensor messages in the workstation clock domain for robot_localization."""

import rclpy
from geometry_msgs.msg import TwistWithCovarianceStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu


class SensorRestamp(Node):
    def __init__(self):
        super().__init__("go1_sensor_restamp")
        qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
        self.imu_pub = self.create_publisher(Imu, "/ekf/imu", qos)
        self.leg_pub = self.create_publisher(TwistWithCovarianceStamped, "/ekf/leg_odom", qos)
        self.zupt_pub = self.create_publisher(TwistWithCovarianceStamped, "/ekf/zupt", qos)
        self.create_subscription(Imu, "/imu", lambda msg: self.republish(msg, self.imu_pub), qos)
        self.create_subscription(TwistWithCovarianceStamped, "/leg_odom",
                                 lambda msg: self.republish(msg, self.leg_pub), qos)
        self.create_subscription(TwistWithCovarianceStamped, "/zupt",
                                 lambda msg: self.republish(msg, self.zupt_pub), qos)

    def republish(self, msg, publisher):
        msg.header.stamp = self.get_clock().now().to_msg()
        publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SensorRestamp()
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
