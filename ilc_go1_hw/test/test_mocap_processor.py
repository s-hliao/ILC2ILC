#!/usr/bin/env python
"""Off-robot test of mocap_odom_node.MocapProcessor (no ROS): python test/test_mocap_processor.py"""
from __future__ import division, print_function

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from mocap_odom_node import MocapProcessor, quat_to_rot, rot_to_quat  # noqa: E402


def rot(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K.dot(K)


def main():
    fails = 0
    offset = np.array([0.05, -0.02, 0.08])              # the marker body's origin in the trunk frame
    v_body, w_body = np.array([0.6, 0.05, -0.2]), np.array([0.3, -1.2, 0.4])
    rate = 240.0
    for up in ("z", "y"):
        proc = MocapProcessor(up_axis=up, yaw_offset=0.0 if up == "z" else 0.0, mocap_offset=offset, tau=0.0,
                              spike_pct=0.0)
        R, o = rot([0, 0, 1], 0.7), np.array([0.2, -0.1, 0.3])
        errs_o, errs_v, errs_w = [], [], []
        for k in range(200):
            t = k / rate
            p_track = o + R.dot(offset)
            q = rot_to_quat(R)
            if up == "y":                               # what Motive streams in Y-up: invert the node's mapping
                P = MocapProcessor.P_YUP
                p_raw = np.array([-p_track[0], p_track[2], p_track[1]])
                q_raw = rot_to_quat(P.T.dot(R))         # B = I (yaw_offset 0): R_world = P R_raw
            else:
                p_raw, q_raw = p_track, q
            p, origin, qo, twist = proc.update(t, p_raw, q_raw)
            errs_o.append(np.abs(origin - o).max())
            if twist is not None:
                errs_v.append(np.abs(twist[:3] - v_body).max())
                errs_w.append(np.abs(twist[3:] - w_body).max())
            # integrate the trunk with constant body-frame velocity / rate (exact over the interval for the rotation)
            dt = 1.0 / rate
            R_next = R.dot(rot(w_body, np.linalg.norm(w_body) * dt))
            # constant body velocity under a rotating frame: integrate finely
            n = 50
            for i in range(n):
                Ri = R.dot(rot(w_body, np.linalg.norm(w_body) * dt * (i + 0.5) / n))
                o = o + Ri.dot(v_body) * dt / n
            R = R_next
        eo, ev, ew = max(errs_o), max(errs_v), max(errs_w)
        ok = eo < 1e-9 and ev < 2e-3 and ew < 1e-6
        fails += not ok
        print("up=%s: origin err %.1e m, body velocity err %.1e m/s, body rate err %.1e rad/s  %s"
              % (up, eo, ev, ew, "ok" if ok else "FAIL"))
    # quaternion round trip
    for k in range(50):
        R = rot(np.random.RandomState(k).randn(3), 0.1 + 0.06 * k)
        e = np.abs(quat_to_rot(rot_to_quat(R)) - R).max()
        if e > 1e-12:
            fails += 1
            print("quaternion round trip FAIL", e)
    print("all tests passed" if not fails else "%d FAILED" % fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
