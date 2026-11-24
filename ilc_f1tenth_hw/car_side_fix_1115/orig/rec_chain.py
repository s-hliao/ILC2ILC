#!/usr/bin/env python3
"""rec_chain.py (2026-11-05), ON THE CAR: records the car's whole state-estimation chain, read-only, to find where the
position jumps seen in fast trials come from (raw mocap vs optitrack_node vs the EKF).
   python3 ~/car_runs/rec_chain.py [seconds = 120]      -> ~/car_runs/chain_<HHMMSS>.npz (also on Ctrl-C)
Each row: receipt time (monotonic), header stamp, then the message values."""
import sys, os, time, math, signal, numpy as np, rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from vesc_msgs.msg import VescStateStamped
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
rclpy.init(); n = rclpy.create_node('rec_chain'); R = {k: [] for k in ('mocap', 'opti', 'ekf', 'imu', 'vesc')}
be = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT)
st = lambda h: h.stamp.sec + h.stamp.nanosec * 1e-9
yaw = lambda q: math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
n.create_subscription(PoseStamped, '/dl_f1tenth/pose', lambda m: R['mocap'].append([time.monotonic(), time.time(), st(m.header), m.pose.position.x, m.pose.position.y, m.pose.position.z, yaw(m.pose.orientation)]), be)
n.create_subscription(Odometry, '/optitrack/odom', lambda m: R['opti'].append([time.monotonic(), time.time(), st(m.header), m.pose.pose.position.x, m.pose.pose.position.y, yaw(m.pose.pose.orientation), m.twist.twist.linear.x, m.twist.twist.linear.y]), be)
n.create_subscription(Odometry, '/odometry/filtered', lambda m: R['ekf'].append([time.monotonic(), time.time(), st(m.header), m.pose.pose.position.x, m.pose.pose.position.y, yaw(m.pose.pose.orientation), m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.angular.z]), be)
n.create_subscription(Imu, '/sensors/imu/data', lambda m: R['imu'].append([time.monotonic(), time.time(), st(m.header), m.linear_acceleration.x, m.linear_acceleration.y, m.angular_velocity.z]), be)
n.create_subscription(VescStateStamped, '/sensors/core', lambda m: R['vesc'].append([time.monotonic(), time.time(), st(m.header), m.state.speed, m.state.avg_iq]), be)
out = os.path.expanduser(f'~/car_runs/chain_{time.strftime("%H%M%S")}.npz'); stop = [False]
signal.signal(signal.SIGINT, lambda *a: stop.__setitem__(0, True)); signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__(0, True))
print(f'rec_chain: recording {secs:.0f} s to {out} (Ctrl-C to stop early)', flush=True); t0 = time.monotonic()
while not stop[0] and time.monotonic() - t0 < secs: rclpy.spin_once(n, timeout_sec=0.01)
np.savez(out, **{k: np.array(v) for k, v in R.items()}); print(f'rec_chain: saved {out}: ' + ', '.join(f'{k} {len(v)}' for k, v in R.items()), flush=True)
