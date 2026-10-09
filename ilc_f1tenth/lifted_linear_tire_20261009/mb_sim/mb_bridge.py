"""
ROS 2 bridge for the multi-body plant (mb_plant.MBPlant), a drop-in for the stock
f1tenth_gym_ros gym_bridge on the topics the ILC uses: /drive (AckermannDriveStamped: speed,
steering_angle), /initialpose (PoseWithCovarianceStamped: reset), /ego_racecar/odom
(Odometry). Physics every 0.01 s once a drive command has arrived, odometry every 0.004 s,
as the stock bridge; no map, lidar or collision (an open arena).

    python3 mb_bridge.py [--log truth.npz]

--log records the hidden MB state at each physics tick (true sideslip, steering angle, roll,
pitch, wheel speeds) for diagnostics; the controller never sees it.
"""
import argparse
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

from mb_plant import MBPlant


def yaw_of(q):
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class MBBridge(Node):
    def __init__(self, log_path=None):
        super().__init__('bridge')
        self.plant = MBPlant()
        self.lock = threading.Lock()
        self.steer, self.speed = 0.0, 0.0
        self.drive_published = False
        self.log_path = log_path
        self.log = []
        self.odom_pub = self.create_publisher(Odometry, '/ego_racecar/odom', 10)
        self.create_subscription(AckermannDriveStamped, '/drive', self.on_drive, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.on_reset, 10)
        self.create_timer(0.01, self.on_physics)
        self.create_timer(0.004, self.on_odom)
        self.get_logger().info('multi-body bridge up (F1TENTH-scale MB parameters)')

    def on_drive(self, msg):
        with self.lock:
            self.speed = msg.drive.speed
            self.steer = msg.drive.steering_angle
            self.drive_published = True

    def on_reset(self, msg):
        p = msg.pose.pose
        with self.lock:
            self.plant.reset((p.position.x, p.position.y, yaw_of(p.orientation)))

    def on_physics(self):
        with self.lock:
            if not self.drive_published:
                return
            self.plant.step(self.steer, self.speed)
            if self.log_path:
                self.log.append(np.r_[time.monotonic(), self.steer, self.speed, self.plant.truth()])

    def on_odom(self):
        with self.lock:
            x, y, yaw, vx, vy, wz = self.plant.odom()
        m = Odometry()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.child_frame_id = 'ego_racecar/base_link'
        m.pose.pose.position.x = float(x)
        m.pose.pose.position.y = float(y)
        m.pose.pose.orientation.z = float(np.sin(yaw / 2))
        m.pose.pose.orientation.w = float(np.cos(yaw / 2))
        m.twist.twist.linear.x = float(vx)
        m.twist.twist.linear.y = float(vy)
        m.twist.twist.angular.z = float(wz)
        self.odom_pub.publish(m)

    def save(self):
        if self.log_path and self.log:
            np.savez(self.log_path, log=np.asarray(self.log),
                     columns=['t', 'steer_cmd', 'speed_cmd', 'vx', 'vy', 'steer', 'roll', 'pitch',
                              'w_lf', 'w_rf', 'w_lr', 'w_rr'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=None)
    a, _ = ap.parse_known_args()
    rclpy.init()
    node = MBBridge(a.log)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
