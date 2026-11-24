#!/bin/bash
# car_pos.sh (2026-11-04), ON THE CAR: where is the car, and is it at the start of the next trial's track?
#   bash ~/car_runs/car_pos.sh                 position only
#   bash ~/car_runs/car_pos.sh plastic|rubber [MMDD]  + the start check against the PENDING request of that session (default today)
#   bash ~/car_runs/car_pos.sh <plan name> [pq|0|<m>]  + the start check against that plan's start: pq = the slow laps'
#        positive-quadrant start (slow_lap.sh's default), 0 (default here) = the plan start (the trials' start line)
# POSITION = the EKF (/odometry/filtered): the frame the plans and net_runner / slow_lap use (user 11-04: "the full position
# of the track is the ekf frame"); the start check uses only it (no EKF -> no check). /dl_f1tenth/pose (NatNet, 180 Hz) is
# shown as health: its rate, worst gap, and the EKF - mocap offset (mocap converted as optitrack_node.py does).
# (11-07, user: in x and y, not relative to the track) the start line says how far to MOVE the car in EKF x / y and
# TURN it. The start check uses net_runner's limits: <= 0.25 m from the start, |lateral| <= 0.10 m, |heading| <= 20 deg, at rest.
# Read-only (subscribes only). Ctrl-C to quit. Runs alongside car_stack.sh / car_runner.sh.
A=$1; WHERE=${2:-0}; PLAN=; SESS=
case "$A" in
  plastic|rubber)   # optional 2nd argument = the session day, as car_runner.sh plastic|rubber [MMDD] (e.g. 1105b)
    DAY=$(date +%m%d); case "$2" in ""|pq|auto|0|[0-9]*.*) ;; *) DAY=$2; WHERE=0;; esac
    SESS=/home/kathy/car_sessions/${DAY}_$A;;
  "") ;;
  *) PLAN=$A;;
esac
source /opt/ros/foxy/setup.bash; source ~/lla_ws/install/setup.bash; source ~/ILC2ILC/install/setup.bash
unset CYCLONEDDS_URI; export RMW_IMPLEMENTATION=rmw_fastrtps_cpp PATH=/usr/bin:$PATH ILC2REAL_DIR=$HOME/ILC2ILC/ilc_f1tenth/ilc2real
source $ILC2REAL_DIR/env_tight.sh; export F1T_MU=0.6 F1T_PLANS=$ILC2REAL_DIR/plans_nom06
exec python3 /dev/fd/3 "$PLAN" "$SESS" "$WHERE" 3<<'PY'
import sys, os, glob, json, math, time, collections
import numpy as np, rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
sys.path.insert(0, os.environ['ILC2REAL_DIR'])
import bank, sim_jax_np as sn
plan_name, sess, WHERE = sys.argv[1], sys.argv[2], sys.argv[3]
def pick_start(xy, ds, how):
    """the lap's start index: 'pq' = the straightest point with x, y > 0.05 m (user 11-04: start in the positive quadrant;
    straightness = max |curvature| from 0.12 m behind to 0.5 m ahead), '0' = the plan's own start, a number = that arc length (m)."""
    n = len(xy)
    if how not in ('pq', 'auto'):
        return int(round(float(how) / ds)) % n
    k = max(1, int(round(0.15 / ds))); a, b, c = np.roll(xy, k, 0), xy, np.roll(xy, -k, 0)
    cr = (b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]
    kap = 2 * cr / (np.linalg.norm(b - a, axis=1) * np.linalg.norm(c - b, axis=1) * np.linalg.norm(c - a, axis=1))
    w = int(round(0.5 / ds)); kmax = np.array([np.abs(kap[np.arange(i - w // 4, i + w) % n]).max() for i in range(n)])
    q = np.where((xy[:, 0] > 0.05) & (xy[:, 1] > 0.05))[0]
    return int(q[np.argmin(kmax[q])]) if len(q) else 0
LIM = (0.25, 0.10, math.radians(20.0), 0.10)          # = net_runner's start_max_dist / _ey / _dpsi_deg / _speed defaults
def quat_to_rot(x, y, z, w):
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
B = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]]); P = np.diag([-1., -1, 1])     # optitrack_node: theta = pi/2, P
def track_offset():
    """(dx, dy) m in the EKF frame from ~/car_runs/track_offset.txt (first line), added to the plan path (user 11-04)."""
    try:
        v = open(os.path.expanduser('~/car_runs/track_offset.txt')).readline().split()
        return np.array([float(v[0]), float(v[1])])
    except (OSError, ValueError, IndexError):
        return np.zeros(2)
OFFSET = track_offset()
plans = {}
def plan_for():
    """(name, PlanNP) of the plan to check: the given one, or the session's first request without a result."""
    name, note = plan_name, ''
    if sess:
        name = None
        try:                                            # the trial net_runner has announced (a re-run jumps the queue)
            pj = json.load(open(os.path.join(sess, '.pending.json')))
            name, note = pj['plan'], f" (req {pj['k']}, announced by net_runner" + (f", re-run of {pj['rerun_of']}" if pj.get('rerun_of') else '') + ')'
        except (OSError, ValueError, KeyError):
            pass
        for f in ([] if name else sorted(glob.glob(os.path.join(sess, 'req_*.json')))):
            k = os.path.basename(f)[4:-5]
            if not os.path.exists(os.path.join(sess, f'res_{k}.json')):
                try: name, note = json.load(open(f))['plan'], f' (req {k}: the next in the queue; no runner announcement)'
                except (OSError, ValueError): pass
                break
        if name is None: return f'no pending request in {sess} (queue the trials, or name a plan)', None   # (message, no plan)
    if not name: return None, None                  # position only: no plan to check (11-05 fix: was '' -> unpack error)
    if name not in plans:
        P_ = sn.PlanNP(bank.load_plan(name)); P_.xy = P_.xy + OFFSET; plans[name] = (P_, pick_start(P_.xy, P_.ds, WHERE))
    return name + note, plans[name]
rclpy.init(); n = rclpy.create_node('car_pos')
st = dict(raw=None, X=None, ekf=None, ekf_t=0.0); ts = collections.deque()
def on_pose(m):
    p, q = m.pose.position, m.pose.orientation
    Rn = B @ P @ quat_to_rot(q.x, q.y, q.z, q.w)
    st['raw'] = (p.x, p.y, p.z); st['X'] = np.array([-p.x, -p.y, math.atan2(Rn[1, 0], Rn[0, 0])])
    ts.append(time.monotonic())
def on_odom(m):
    q, t = m.pose.pose.orientation, m.twist.twist
    st['ekf'] = np.array([m.pose.pose.position.x, m.pose.pose.position.y,
                          math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z)), math.hypot(t.linear.x, t.linear.y)])
    st['ekf_t'] = time.monotonic()
n.create_subscription(PoseStamped, '/dl_f1tenth/pose', on_pose, QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT))
n.create_subscription(Odometry, '/odometry/filtered', on_odom, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
print(f'car_pos: position = EKF; track offset {OFFSET[0]:+.2f}, {OFFSET[1]:+.2f} m (~/car_runs/track_offset.txt); Ctrl-C to quit', flush=True)
last = t0s = time.monotonic()
try:
    while rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.01)
        now = time.monotonic()
        if now - last < 0.5: continue
        last = now
        while ts and now - ts[0] > 2.0: ts.popleft()
        if now - t0s < 1.0: continue                    # the first second: frames are still arriving
        if not ts or now - ts[-1] > 0.5:
            print(f'{time.strftime("%H:%M:%S")}  NO MOCAP on /dl_f1tenth/pose (Motive tracking the car? NatNet up?)', flush=True); continue
        t = np.array(ts); hz = (len(t) - 1) / max(t[-1] - t[0], 1e-6); gap = np.diff(t).max() * 1e3 if len(t) > 1 else 0
        X = st['X']
        mocap = f'mocap {hz:3.0f} Hz, worst gap {gap:3.0f} ms' + ('  LOW RATE' if hz < 150 or gap > 50 else '')
        ekf_ok = st['ekf'] is not None and now - st['ekf_t'] < 0.2
        if not ekf_ok:
            print(f'{time.strftime("%H:%M:%S")}  NO EKF (/odometry/filtered): car_stack.sh up? -- no position / start check without it | {mocap}', flush=True)
            continue
        e = st['ekf']; off = e[:2] - X[:2]
        print(f'{time.strftime("%H:%M:%S")}  EKF X {e[0]:+.3f} Y {e[1]:+.3f} yaw {math.degrees(e[2]):+6.1f} deg v {e[3]:.2f} | {mocap}, '
              f'EKF - mocap ({off[0]:+.3f}, {off[1]:+.3f}) m, yaw {math.degrees(math.atan2(math.sin(e[2] - X[2]), math.cos(e[2] - X[2]))):+.1f} deg', flush=True)
        name, pl = plan_for()
        if pl is not None: pl, I0 = pl
        if pl is None:
            if name: print(f'          {name}', flush=True)
            continue
        x = st['ekf']                                       # the EKF: what net_runner / slow_lap check
        s_off, ey, _ = pl.path_coords(I0, np.r_[x[:3], 0, 0, 0])
        dpsi = math.atan2(math.sin(x[2] - pl.theta[I0]), math.cos(x[2] - pl.theta[I0]))
        dist = float(np.hypot(*(x[:2] - pl.xy[I0])))
        bad = [w for w, b in (('far', dist > LIM[0]), ('lateral', abs(ey) > LIM[1]), ('heading', abs(dpsi) > LIM[2]),
                              ('moving', x[3] > LIM[3])) if b]
        print(f'          start of {name}{" (positive quadrant)" if I0 else ""}: ({pl.xy[I0][0]:+.2f}, {pl.xy[I0][1]:+.2f}) heading {math.degrees(pl.theta[I0]):+.0f} deg | car '
              f'at ({x[0]:+.2f}, {x[1]:+.2f}): MOVE x {pl.xy[I0][0] - x[0]:+.2f} m, y {pl.xy[I0][1] - x[1]:+.2f} m ({dist:.2f} m), TURN {math.degrees(-dpsi):+.0f} deg (+ = counter-clockwise) '
              f'[EKF]  ->  {"OK to start" if not bad else "NOT OK: " + ", ".join(bad)}', flush=True)
except KeyboardInterrupt:
    pass
PY
