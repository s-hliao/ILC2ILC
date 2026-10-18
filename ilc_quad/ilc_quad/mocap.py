"""ROS-independent 3-D OptiTrack pose processing shared by the Go1 workstation nodes."""

import math
from collections import deque

import numpy as np


def quat_to_rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                     [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                     [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])


def rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        q = np.array([(R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s,
                      (R[1, 0] - R[0, 1]) * s, 0.25 / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s,
                      (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s,
                      (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s])
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s,
                      0.25 * s, (R[1, 0] - R[0, 1]) / s])
    return q if q[3] >= 0 else -q


def rot_log(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    th = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return 0.5 * v if th < 1e-9 else th / (2.0 * math.sin(th)) * v


class CausalSpikeEMA:
    def __init__(self, tau, spike_pct, window_sec):
        self.tau = tau
        self.spike_pct = max(0.0, min(49.0, spike_pct))
        self.window_sec = window_sec
        self._t, self._v = deque(), deque()
        self._ema = self._last = None

    def update(self, t, v):
        self._t.append(t)
        self._v.append(v)
        while self._t and t - self._t[0] > self.window_sec:
            self._t.popleft()
            self._v.popleft()
        if self.spike_pct > 0 and len(self._v) >= 3:
            values = np.asarray(self._v, float)
            v = min(max(v, np.percentile(values, self.spike_pct)),
                    np.percentile(values, 100.0 - self.spike_pct))
        if self._ema is None or self._last is None:
            self._ema = v
        else:
            dt = t - self._last
            if dt > 0:
                self._ema = v if self.tau <= 0 else self._ema + dt / (self.tau + dt) * (v - self._ema)
        self._last = t
        return self._ema


class MocapProcessor:
    """Convert axes/offset and produce filtered body-frame linear and angular velocity."""

    P_YUP = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    def __init__(self, up_axis="z", yaw_offset=0.0, mocap_offset=(0.0, 0.0, 0.0),
                 tau=0.02, spike_pct=15.0, spike_window=0.3, max_gap=0.1):
        if up_axis not in ("z", "y"):
            raise ValueError("up_axis must be z or y")
        self.up_axis = up_axis
        c, s = math.cos(yaw_offset), math.sin(yaw_offset)
        self.B = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.offset = np.asarray(mocap_offset, float)
        self.filt = [CausalSpikeEMA(tau, spike_pct, spike_window) for _ in range(6)]
        self.max_gap = max_gap
        self.prev = None

    def to_world(self, p, q):
        p, R = np.asarray(p, float), quat_to_rot(q)
        if self.up_axis == "y":
            return np.array([-p[0], p[2], p[1]]), self.B.dot(self.P_YUP).dot(R)
        return self.B.dot(p), self.B.dot(R)

    def update(self, t, p_raw, q_raw):
        p, R = self.to_world(p_raw, q_raw)
        origin = p - R.dot(self.offset)
        twist = None
        if self.prev is not None:
            t0, o0, R0 = self.prev
            dt = t - t0
            if 0 < dt <= self.max_gap:
                raw = np.concatenate([R.T.dot((origin - o0) / dt), rot_log(R0.T.dot(R)) / dt])
                twist = np.array([f.update(t, x) for f, x in zip(self.filt, raw)])
            elif dt > self.max_gap:
                self.filt = [CausalSpikeEMA(f.tau, f.spike_pct, f.window_sec) for f in self.filt]
        self.prev = (t, origin, R)
        return p, origin, rot_to_quat(R), twist
