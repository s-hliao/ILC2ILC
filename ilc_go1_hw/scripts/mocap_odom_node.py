#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""mocap_odom_node: the trunk's motion-capture pose for the Go1, in ROS 1 (Python 2.7 / 3: melodic or noetic).

Ported from the car's hardware layer (ilc_f1tenth_hw/scripts/optitrack_node.py, itself from LLA-MPC-onboard): the same
axis handling and the same causal velocity filter (a windowed spike clamp, then a dt-normalized EMA), for a 3-D jumping
trunk instead of a planar car.

    subscribes  <mocap_topic>   geometry_msgs/PoseStamped of the trunk's rigid body (normally the workstation's
                                natnet_ros2 /<body>/pose, carried into ROS 1 by ros1_bridge)
    publishes   mocap/pose      geometry_msgs/PoseStamped, the TRACKED point in a z-up world (frame `world_frame`) --
                                what the jump controller reads (pose_type:=pose); it applies its own mocap_offset and
                                pose_latency, exactly as in simulation
                mocap/odom      nav_msgs/Odometry of the trunk ORIGIN (tracked point less mocap_offset in the body
                                frame), stamped at receipt - pose_latency; twist = filtered finite differences in the
                                body frame: the EKF's absolute input
    tf          world_frame -> trunk_frame (publish_tf, off when the EKF publishes it)

Axes: up_axis z (Motive / VRPN streaming Z-up: used as is) or y (Motive's default Y-up: position (-x, z, y) and the
orientation B(yaw_offset) P R, the car's conversion). Check by moving the robot forward by hand: x must grow.
"""
from __future__ import division, print_function

import math
from collections import deque

import numpy as np


def quat_to_rot(q):
    """(x, y, z, w) -> R (body -> world)."""
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_to_quat(R):
    """R -> (x, y, z, w), w >= 0."""
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        q = np.array([(R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s, 0.25 / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s])
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s])
    return q if q[3] >= 0 else -q


def rot_log(R):
    """The rotation vector of R (axis * angle)."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    th = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-9:
        return 0.5 * v
    return th / (2.0 * math.sin(th)) * v


class CausalSpikeEMA(object):
    """The car's live filter for one channel: clamp to the trailing window's [pct, 100 - pct] percentile band, then a
    dt-normalized EMA with time constant tau (keep it small: its lag adds to the EKF's)."""

    def __init__(self, tau, spike_pct, window_sec):
        self.tau = tau
        self.spike_pct = max(0.0, min(49.0, spike_pct))
        self.window_sec = window_sec
        self._t, self._v = deque(), deque()
        self._ema = None
        self._last = None

    def update(self, t, v):
        self._t.append(t)
        self._v.append(v)
        while self._t and t - self._t[0] > self.window_sec:
            self._t.popleft()
            self._v.popleft()
        if self.spike_pct > 0 and len(self._v) >= 3:
            arr = np.asarray(self._v, float)
            v = min(max(v, np.percentile(arr, self.spike_pct)), np.percentile(arr, 100.0 - self.spike_pct))
        if self._ema is None or self._last is None:
            self._ema = v
        else:
            dt = t - self._last
            if dt > 0 and self.tau > 0:
                self._ema += dt / (self.tau + dt) * (v - self._ema)
            elif dt > 0:
                self._ema = v
        self._last = t
        return self._ema


class MocapProcessor(object):
    """Everything but ROS: axes, the trunk origin, the filtered body-frame twist."""

    P_YUP = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    def __init__(self, up_axis="z", yaw_offset=0.0, mocap_offset=(0.0, 0.0, 0.0), tau=0.02, spike_pct=15.0,
                 spike_window=0.3, max_gap=0.1):
        if up_axis not in ("z", "y"):
            raise ValueError("up_axis must be z or y")
        self.up_axis = up_axis
        c, s = math.cos(yaw_offset), math.sin(yaw_offset)
        self.B = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.offset = np.asarray(mocap_offset, float)
        self.filt = [CausalSpikeEMA(tau, spike_pct, spike_window) for _ in range(6)]
        self.max_gap = max_gap
        self.prev = None                       # (t, origin, R)

    def to_world(self, p, q):
        """Raw mocap position / quaternion (x, y, z, w) -> z-up world (position, R)."""
        p, R = np.asarray(p, float), quat_to_rot(q)
        if self.up_axis == "y":
            p = np.array([-p[0], p[2], p[1]])
            R = self.B.dot(self.P_YUP).dot(R)
        else:
            p = self.B.dot(p)
            R = self.B.dot(R)
        return p, R

    def update(self, t, p_raw, q_raw):
        """-> (tracked point, origin, quaternion (x, y, z, w), body-frame [vx vy vz wx wy wz] or None)."""
        p, R = self.to_world(p_raw, q_raw)
        origin = p - R.dot(self.offset)          # tracked point -> trunk origin (the controller's convention)
        twist = None
        if self.prev is not None:
            t0, o0, R0 = self.prev
            dt = t - t0
            if 0 < dt <= self.max_gap:
                v_body = R.T.dot((origin - o0) / dt)
                w_body = rot_log(R0.T.dot(R)) / dt          # body-frame angular velocity over the interval
                raw = np.concatenate([v_body, w_body])
                twist = np.array([f.update(t, x) for f, x in zip(self.filt, raw)])
            elif dt > self.max_gap:                          # a dropout: restart the filters
                self.filt = [CausalSpikeEMA(f.tau, f.spike_pct, f.window_sec) for f in self.filt]
        self.prev = (t, origin, R)
        return p, origin, rot_to_quat(R), twist


def main():
    import rospy
    import tf2_ros
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from nav_msgs.msg import Odometry

    rospy.init_node("mocap_odom")
    gp = lambda n, d: rospy.get_param("~" + n, d)
    world, trunk = gp("world_frame", "world"), gp("trunk_frame", "trunk")
    latency = float(gp("pose_latency", 0.0))
    use_source_stamp = bool(gp("use_source_stamp", False))
    proc = MocapProcessor(gp("up_axis", "z"), float(gp("yaw_offset", 0.0)), gp("mocap_offset", [0.0, 0.0, 0.0]),
                          float(gp("vel_ema_tau", 0.02)), float(gp("vel_spike_pct", 15.0)),
                          float(gp("vel_spike_window", 0.3)), float(gp("max_gap", 0.1)))
    pos_var, rot_var = float(gp("pos_var", 1e-6)), float(gp("rot_var", 1e-4))
    lin_var, ang_var = float(gp("lin_vel_var", 0.04)), float(gp("ang_vel_var", 0.09))
    pub_pose = rospy.Publisher("mocap/pose", PoseStamped, queue_size=1, tcp_nodelay=True)
    pub_odom = rospy.Publisher("mocap/odom", Odometry, queue_size=1, tcp_nodelay=True)
    br = tf2_ros.TransformBroadcaster() if gp("publish_tf", False) else None

    def on_pose(msg):
        receipt_stamp = rospy.Time.now()
        source_stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0 else receipt_stamp
        # NatNet runs on the workstation, so its ROS stamp may be in a different clock domain. Keep its frame interval
        # for finite differences, but date the EKF measurement in the robot's clock unless clocks are explicitly synced.
        stamp = source_stamp if use_source_stamp else receipt_stamp
        o = msg.pose.orientation
        p, origin, q, twist = proc.update(source_stamp.to_sec(), [msg.pose.position.x, msg.pose.position.y,
                                                                  msg.pose.position.z], [o.x, o.y, o.z, o.w])
        ps = PoseStamped()
        ps.header.stamp, ps.header.frame_id = stamp, world
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = p
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = q
        pub_pose.publish(ps)
        od = Odometry()
        od.header.stamp = stamp - rospy.Duration(latency)
        od.header.frame_id, od.child_frame_id = world, trunk
        od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = origin
        od.pose.pose.orientation = ps.pose.orientation
        cov = [0.0] * 36
        for i in range(3):
            cov[7 * i], cov[7 * (i + 3)] = pos_var, rot_var
        od.pose.covariance = cov
        if twist is not None:
            t = od.twist.twist
            t.linear.x, t.linear.y, t.linear.z, t.angular.x, t.angular.y, t.angular.z = twist
            tc = [0.0] * 36
            for i in range(3):
                tc[7 * i], tc[7 * (i + 3)] = lin_var, ang_var
            od.twist.covariance = tc
        pub_odom.publish(od)
        if br is not None:
            tf = TransformStamped()
            tf.header.stamp, tf.header.frame_id, tf.child_frame_id = od.header.stamp, world, trunk
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = origin
            tf.transform.rotation = ps.pose.orientation
            br.sendTransform(tf)

    topic = gp("mocap_topic", "/dl_go1/pose")
    rospy.Subscriber(topic, PoseStamped, on_pose, queue_size=1, tcp_nodelay=True)
    rospy.loginfo("mocap_odom: %s (up %s, yaw offset %.3f, offset %s, latency %.4f s, %s stamps) -> mocap/pose, mocap/odom",
                  topic, proc.up_axis, float(gp("yaw_offset", 0.0)), list(proc.offset), latency,
                  "source" if use_source_stamp else "robot receipt")
    rospy.spin()


if __name__ == "__main__":
    main()
