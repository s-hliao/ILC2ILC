#!/usr/bin/env python3
"""go1_bridge (ilc_quad/scripts/go1_bridge.py, unchanged) + the Go1's IMU on `imu` (sensor_msgs/Imu), 2026-11-21.

The walker's observation needs the trunk's angular velocity (the IMU gyro); the jump bridge reads LowState every cycle but does
not publish its IMU. This node IS that bridge (subclassed: same UDP loop, safety, watchdog, joint_states, foot_contacts,
clear_fault) and publishes, from the same LowState, at the same rate:
    imu   sensor_msgs/Imu, frame_id trunk: orientation (x, y, z, w) from LowState.imu.quaternion (w, x, y, z),
          angular_velocity = imu.gyroscope (rad/s), linear_acceleration = imu.accelerometer (m/s^2)
Usage (workstation, wired to the robot, robot in low-level mode -- see go1_bridge.py's notes):
    source /opt/ros/humble/setup.bash; source ~/ros2_ws/install/setup.bash
    python3 go1_bridge_imu.py --ros-args -p sdk_path:=$HOME/unitree_legged_sdk/lib/python/amd64 -p power_level:=10 \
        -p menagerie_root:=$HOME/mujoco_menagerie [-p bridge_file:=~/ros2_ws/src/ilc_quad/scripts/go1_bridge.py]
"""
import importlib.util
import os
import sys

import rclpy
from sensor_msgs.msg import Imu

BRIDGE = os.environ.get('GO1_BRIDGE_FILE', os.path.expanduser('~/ros2_ws/src/ilc_quad/scripts/go1_bridge.py'))
spec = importlib.util.spec_from_file_location('go1_bridge', BRIDGE)
gb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gb)


class Go1BridgeImu(gb.Go1Bridge):
    def __init__(self):
        super().__init__()
        self.imu_pub = self.create_publisher(Imu, 'imu', gb.SENSOR_QOS)
        self.get_logger().info(f'+ imu on {self.resolve_topic_name("imu")} (bridge code: {BRIDGE})')

    def _cycle(self):
        super()._cycle()
        imu = self.lowstate.imu
        qw, qx, qy, qz = (float(x) for x in imu.quaternion)
        if qw == 0.0 and qx == 0.0 and qy == 0.0 and qz == 0.0:      # nothing from the robot yet
            return
        m = Imu()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'trunk'
        m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = qx, qy, qz, qw
        g = [float(x) for x in imu.gyroscope]
        acc = [float(x) for x in imu.accelerometer]
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = g
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = acc
        self.imu_pub.publish(m)


def main(argv=None):
    rclpy.init(args=argv)
    node = None
    try:
        node = Go1BridgeImu()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv)
