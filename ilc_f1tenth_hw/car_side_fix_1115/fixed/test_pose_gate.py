"""Unit tests of optitrack_node.PoseGate on synthetic 240 Hz streams shaped like the 11-08 / 11-09 glitches.

    python3 -m pytest test_pose_gate.py -q        (no ROS needed: rclpy etc. are stubbed)
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

# optitrack_node imports ROS at module level; stub the modules so PoseGate imports on a workstation without ROS.
for name in ('rclpy', 'rclpy.node', 'rclpy.qos', 'geometry_msgs', 'geometry_msgs.msg', 'nav_msgs', 'nav_msgs.msg',
             'tf2_ros'):
    module = types.ModuleType(name)
    module.Node = object
    module.QoSProfile = module.QoSReliabilityPolicy = module.PoseStamped = module.TransformStamped = object
    module.Odometry = module.TransformBroadcaster = object
    sys.modules.setdefault(name, module)
spec = importlib.util.spec_from_file_location('optitrack_node', Path(__file__).with_name('optitrack_node.py'))
node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)

DT = 1.0 / 240.0


def gate():
    """The node's defaults: 30 deg + 8 rad/s yaw, 35 deg tilt, 0.3 s reset, 0.05 m + 7 m/s position."""
    return node.PoseGate(np.radians(30.0), 8.0, 35.0, 0.3, 0.05, 7.0)


def rot(yaw, tilt=0.0):
    """Body->world rotation: yaw about z, then a tilt about x."""
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    ct, st = np.cos(tilt), np.sin(tilt)
    rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    return rz @ rx


def drift_stream(n, speed=2.5, yaw_rate=2.5, t0=0.0):
    """A power slide: 2.5 m/s on a circle at 2.5 rad/s (the hardest real motion logged)."""
    for k in range(n):
        t = t0 + k * DT
        yaw = yaw_rate * t
        pos = np.array([speed / yaw_rate * np.sin(yaw), speed / yaw_rate * (1 - np.cos(yaw)), 0.1])
        yield t, pos, rot(yaw)


def test_real_drift_is_never_dropped():
    g = gate()
    assert all(g.check(t, p, r) == '' for t, p, r in drift_stream(2400))


def test_heading_flip_dropped_then_recovers():
    g = gate()
    frames = list(drift_stream(200))
    reasons = []
    for k, (t, p, r) in enumerate(frames):
        if k == 100:                                   # 00064: +140 deg for one frame
            r = rot(np.arctan2(r[1, 0], r[0, 0]) + np.radians(140))
        reasons.append(g.check(t, p, r))
    assert reasons[100].startswith('yaw jump')
    assert all(x == '' for k, x in enumerate(reasons) if k != 100)


def test_upside_down_solve_always_dropped():
    g = gate()
    for t, p, r in drift_stream(50):
        g.check(t, p, r)
    flipped = [g.check(t + 50 * DT, p, rot(0.0, np.radians(170))) for t, p, _ in drift_stream(240)]
    assert all(x.startswith('tilt') for x in flipped)   # 00081: persistent 169-172 deg tilt, never accepted


def test_position_jumps_dropped():
    g = gate()
    frames = list(drift_stream(300))
    out = []
    for k, (t, p, r) in enumerate(frames):
        if k == 120:
            p = p + np.array([1.9, 0.0, 0.0])           # 00088: 1.9 m in one tick
        if k == 200:
            p = p + np.array([0.0, 0.49, 0.0])          # 00043: 0.49 m
        out.append(g.check(t, p, r))
    assert out[120].startswith('position jump') and out[200].startswith('position jump')
    assert all(x == '' for k, x in enumerate(out) if k not in (120, 200))


def test_alternating_missolves_on_a_straight():
    g = gate()                                          # 00092: pose jumping 0.9-1.4 m back and forth
    out = []
    for k in range(480):
        t = k * DT
        p = np.array([2.0 * t, 0.0, 0.1])
        if k % 7 == 3:
            p = p + np.array([1.2 * (-1) ** k, 0.0, 0.0])
        out.append((k % 7 == 3, g.check(t, p, rot(0.0))))
    assert all(bool(reason) == bad for bad, reason in out)


def test_reacquire_after_dropout_is_accepted():
    g = gate()
    for t, p, r in drift_stream(100):
        g.check(t, p, r)
    t_back = 100 * DT + 0.5                             # 0.5 s without frames (WiFi / Motive stall); the car moved 1.25 m
    assert g.check(t_back, np.array([1.25, 0.6, 0.1]), rot(1.3)) == ''
