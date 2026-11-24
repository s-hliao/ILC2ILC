#!/bin/bash
# slow_lap.sh (2026-11-04), ON THE CAR: drive a plan's path SLOWLY (no network) to check the car's position and the path's
# size in the room before any trial.
#   bash ~/car_runs/slow_lap.sh <plan> [speed m/s = 0.6] [laps = 1] [start = pq | 0 | <arc length m>]
#   start pq (default, user 11-04): the straightest point of the path in the positive quadrant (x, y > 0); 0: the plan's
#   own start; near: wherever the car is (the nearest path point; looser check: lateral <= 0.4 m, heading <= 45 deg)
#   DITHER=<rad> adds a steering wiggle (that amplitude, 2.5 s period) on top of pure pursuit: the steering calibration
#   (steer_cal.sh) uses it so the actual-vs-commanded fit is identifiable in closed loop
#   a number: that arc length along the path
#   e.g. bash ~/car_runs/slow_lap.sh mocap_peanutM_b0_free_a33
# Needs car_stack.sh running (drive_relay: its watchdog brakes if this stops); net_runner must NOT be running (one
# controller on /mpc/drive). Walk next to the car. Enter (or Ctrl-C) here stops it at once: brake, wheels straight.
# Steering: pure pursuit (user: "just run a slow pure pursuit"), 0.5 m lookahead along the path, clipped to +-0.32 rad (the
# servo limits). Pure pursuit cuts corners slightly (a few cm inside on these radii). Speed: the VESC's closed-loop speed
# mode (drive.speed), ramped up over 1.5 s. Start: the same check as net_runner (<= 0.25 m from the plan's start, lateral
# <= 0.10 m, heading <= 20 deg, at rest), then wheels straight 0.5 s. Stops: |lateral| > 0.5 m (the envelope), |heading
# error| > 1.2 rad, odometry older than 0.1 s, MOCAP (/dl_f1tenth/pose) older than 0.5 s, the laps done.
# (21:20: 0.1 s stopped a lap on a 120 ms WiFi gap; parked, 30 s showed gaps to 88 ms and none > 100 ms. The EKF carries
# short gaps; 0.5 s = 0.3 m at 0.6 m/s still ends a real drop.)
# (20:45: the car's WiFi dropped; Motive reaches the car over WiFi, but the on-car EKF keeps publishing /odometry/filtered
# from its own prediction -- an odometry age check alone never sees a lost mocap.) Log: ~/car_runs/slow_laps/<time>_<plan>.npz + a summary.
PLAN=$1; V=${2:-0.6}; LAPS=${3:-1}; START=${4:-pq}
[ -n "$PLAN" ] || { echo "usage: bash ~/car_runs/slow_lap.sh <plan> [speed m/s] [laps]"; exit 1; }
python3 -c "import sys; v=float('$V'); sys.exit(0 if 0.2 <= v <= 1.5 else 1)" || { echo "!!! speed must be 0.2..1.5 m/s"; exit 1; }
if [ -z "$SLOWLAP_TEST" ]; then
  pgrep -f 'drive_relay\.py' >/dev/null || { echo "!!! drive_relay is not running: start bash ~/car_runs/car_stack.sh first"; exit 1; }
fi
if [ -z "$SLOWLAP_TEST" ] && pgrep -f '^python3 \S*net_runner(_patched)?\.py' >/dev/null; then echo "!!! net_runner is running: stop it (Ctrl-C) first, one controller at a time"; exit 1; fi
source /opt/ros/foxy/setup.bash; source ~/lla_ws/install/setup.bash; source ~/ILC2ILC/install/setup.bash
if [ -z "$SLOWLAP_TEST" ]; then unset CYCLONEDDS_URI ROS_DOMAIN_ID ROS_LOCALHOST_ONLY; fi
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp PATH=/usr/bin:$PATH ILC2REAL_DIR=$HOME/ILC2ILC/ilc_f1tenth/ilc2real
source $ILC2REAL_DIR/env_tight.sh; export F1T_MU=0.6 F1T_PLANS=$ILC2REAL_DIR/plans_nom06 PYTHONUNBUFFERED=1
mkdir -p ~/car_runs/slow_laps
export DITHER=${DITHER:-0}
exec python3 /dev/fd/3 "$PLAN" "$V" "$LAPS" "$START" 3<<'PY'
import sys, os, math, time, threading, numpy as np, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy
sys.path.insert(0, os.environ['ILC2REAL_DIR'])
import bank, car_model as cm, sim_jax_np as sn
plan_name, V, LAPS, START = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
L = cm.NOMINAL['lf'] + cm.NOMINAL['lr']; SMAX = 0.32
LD = 0.5                                              # m: pure-pursuit lookahead along the path
LIM = (0.25, 0.10, math.radians(20.0), 0.10)          # net_runner's start limits
ENV, SPIN, STALE, MOCAP_LOST = 0.5, 1.2, 0.1, 0.5
def track_offset():
    """(dx, dy) m in the EKF frame from ~/car_runs/track_offset.txt (first line), added to the plan path (user 11-04)."""
    try:
        v = open(os.path.expanduser('~/car_runs/track_offset.txt')).readline().split()
        return np.array([float(v[0]), float(v[1])])
    except (OSError, ValueError, IndexError):
        return np.zeros(2)
OFFSET = track_offset()
pl = sn.PlanNP(bank.load_plan(plan_name)); pl.xy = pl.xy + OFFSET; xy, th, M, ds = pl.xy, pl.theta, pl.M, pl.ds
k = max(1, int(round(0.15 / ds)))                     # signed geometric curvature over a 0.3 m chord (the plan's kappa array spikes)
a, b, c = np.roll(xy, k, 0), xy, np.roll(xy, -k, 0)
cr = (b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]
kap = 2 * cr / (np.linalg.norm(b - a, axis=1) * np.linalg.norm(c - b, axis=1) * np.linalg.norm(c - a, axis=1))
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
wrap = lambda x: math.atan2(math.sin(x), math.cos(x))
I0 = 0 if START == 'near' else pick_start(xy, ds, START)
DITHER = float(os.environ.get('DITHER', 0)); DPER = 2.5
if START == 'near': LIM = (0.5, 0.40, math.radians(45.0), 0.10)
def coords(i, x):
    d = x[:2] - xy[i]; t = th[i]
    return d[0] * math.cos(t) + d[1] * math.sin(t), -d[0] * math.sin(t) + d[1] * math.cos(t), wrap(x[2] - t)
class Slow(Node):
    def __init__(s):
        super().__init__('slow_lap')
        s.odom, s.odom_t, s.odom_age = None, 0.0, 0.0
        s.create_subscription(Odometry, '/odometry/filtered', s.on_odom, 10)
        s.mocap_t = 0.0
        s.create_subscription(PoseStamped, '/dl_f1tenth/pose', lambda m: setattr(s, 'mocap_t', time.monotonic()),
                              QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        s.pub = s.create_publisher(AckermannDriveStamped, '/mpc/drive', 10)
    def on_odom(s, m):
        q, t = m.pose.pose.orientation, m.twist.twist
        s.odom = np.array([m.pose.pose.position.x, m.pose.pose.position.y,
                           math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)), t.linear.x, t.linear.y, t.angular.z])
        s.odom_t = time.monotonic()
        st = m.header.stamp; s.odom_age = (s.get_clock().now().nanoseconds - (st.sec * 10**9 + st.nanosec)) * 1e-9
    def send(s, steer, speed, brake=False):
        m = AckermannDriveStamped(); m.header.stamp = s.get_clock().now().to_msg(); m.header.frame_id = 'base_link'
        m.drive.steering_angle = float(steer)
        if brake: m.drive.jerk, m.drive.acceleration, m.drive.speed = 1.0, 0.0, 0.0     # duty 0: the VESC brakes
        else: m.drive.jerk, m.drive.acceleration, m.drive.speed = 0.0, 0.0, float(speed)  # closed-loop speed mode
        s.pub.publish(m)
    def spin_for(s, secs, fn=None):             # ROS spins in its own thread: here only wait (and send)
        t = time.monotonic()
        while time.monotonic() - t < secs:
            if fn: fn()
            time.sleep(0.025)
rclpy.init(); n = Slow()
# (desk test 20:33) spinning once per control step let odometry queue up behind the loop: the steering then acted on old
# positions and oscillated. A spin thread keeps n.odom the newest message.
from rclpy.executors import SingleThreadedExecutor
exe = SingleThreadedExecutor(); exe.add_node(n); threading.Thread(target=exe.spin, daemon=True).start()
n.spin_for(2.0)
others = n.count_publishers('/mpc/drive') - 1
if others > 0: print(f'!!! {others} other publisher(s) on /mpc/drive: stop them first'); sys.stdout.flush(); os._exit(1)
if n.odom is None or time.monotonic() - n.odom_t > STALE: print('!!! no fresh /odometry/filtered: is car_stack.sh up and Motive tracking the car?'); sys.stdout.flush(); os._exit(1)
if time.monotonic() - n.mocap_t > MOCAP_LOST: print('!!! no fresh /dl_f1tenth/pose (mocap): Motive tracking the car? WiFi up?'); sys.stdout.flush(); os._exit(1)
print(f'slow lap: {plan_name}, {V:.2f} m/s, {LAPS:g} lap(s) = {LAPS * M * ds:.1f} m; START ({xy[I0][0]:+.2f}, {xy[I0][1]:+.2f}) heading {math.degrees(th[I0]):+.0f} deg '
      f'(track offset {OFFSET[0]:+.2f}, {OFFSET[1]:+.2f} m; path s = {I0 * ds:.2f} m{", nearest point to the car (re-picked at each check)" if START == "near" else ", the plan start" if I0 == 0 else ", positive quadrant" if START in ("pq", "auto") else ""})')
while True:
    x = n.odom
    if START == 'near': I0 = int(np.argmin(np.sum((xy - x[:2]) ** 2, 1)))
    s_off, ey, dpsi = coords(I0, x); dist = float(np.hypot(*(x[:2] - xy[I0]))); v = float(np.hypot(x[3], x[4]))
    bad = [w for w, bb in (('too far from the start', dist > LIM[0]), ('lateral', abs(ey) > LIM[1]), ('heading', abs(dpsi) > LIM[2]), ('moving', v > LIM[3])) if bb]
    print(f'  car {dist:.2f} m from the start (along {s_off:+.2f}, lateral {ey:+.2f}), heading {math.degrees(dpsi):+.0f} deg, speed {v:.2f}'
          + (f'  -> NOT OK: {", ".join(bad)}' if bad else '  -> OK'))
    ans = input('>>> Enter to DRIVE (wheels straight 0.5 s first), r + Enter to re-check, q + Enter to quit: ' if not bad else
                '>>> re-place the car, then Enter to re-check (q + Enter to quit): ').strip().lower()
    n.spin_for(0.1)
    if ans == 'q': os._exit(0)
    if not bad and ans == '': break
stop = threading.Event(); threading.Thread(target=lambda: (sys.stdin.readline(), stop.set()), daemon=True).start()
n.spin_for(0.5, lambda: n.send(0.0, 0.0, brake=True))
x = n.odom; i = pl.project(I0, x[:2]); prog = 0.0; t0 = time.monotonic(); last_print = t0
rec = []; reason = ''
print('>>> DRIVING -- Enter here stops it', flush=True)
try:
    while True:
        now = time.monotonic()
        if stop.is_set(): reason = 'stopped (Enter)'; break
        if now - n.odom_t > STALE: reason = f'stale odometry ({(now - n.odom_t) * 1e3:.0f} ms)'; break
        if now - n.mocap_t > MOCAP_LOST: reason = f'mocap lost ({(now - n.mocap_t) * 1e3:.0f} ms without /dl_f1tenth/pose)'; break
        x = n.odom
        j = pl.project(i, x[:2]); prog += (((j - i) + M // 2) % M - M // 2) * ds; i = j
        _, ey, epsi = coords(i, x)
        if abs(ey) > ENV: reason = f'lateral {ey:+.2f} m beyond the {ENV} m envelope'; break
        if abs(epsi) > SPIN: reason = f'heading error {math.degrees(epsi):+.0f} deg'; break
        if prog >= LAPS * M * ds: reason = 'done'; break
        g = xy[(i + int(round(LD / ds))) % M] - x[:2]          # the lookahead point, in the car's frame
        gx, gy = math.cos(x[2]) * g[0] + math.sin(x[2]) * g[1], -math.sin(x[2]) * g[0] + math.cos(x[2]) * g[1]
        steer = math.atan2(2 * L * gy, gx * gx + gy * gy)                                 # delta = atan(2 L sin(a) / ld)
        steer = float(np.clip(steer + DITHER * math.sin(2 * math.pi * (now - t0) / DPER), -SMAX, SMAX))
        vc = V * min(1.0, (now - t0) / 1.5)
        n.send(steer, vc)
        rec.append([now - t0, *x, i, ey, epsi, steer, vc, n.odom_age])
        if now - last_print >= 1.0:
            last_print = now
            print(f'  {now - t0:5.1f} s  lap {prog / (M * ds):.2f}  speed {math.hypot(x[3], x[4]):.2f}  lateral {ey:+.3f} m  heading err {math.degrees(epsi):+4.0f} deg  steer {steer:+.3f}', flush=True)
        time.sleep(max(0.0, 0.025 - (time.monotonic() - now)))
except KeyboardInterrupt:
    reason = 'stopped (Ctrl-C)'
finally:
    n.spin_for(1.0, lambda: n.send(0.0, 0.0, brake=True))
R = np.array(rec)
out = os.path.expanduser(f'~/car_runs/slow_laps/{time.strftime("%H%M%S")}_{plan_name}.npz')
np.savez(out, dither=DITHER, track_offset=OFFSET, start_idx=I0, rec=R, cols='t x y psi vx vy r idx ey epsi steer vcmd odom_age', plan=plan_name, speed=V, reason=reason, path_xy=xy)
print(f'\n=== {reason}: {prog / (M * ds):.2f} laps in {R[-1, 0] if len(R) else 0:.1f} s')
if len(R):
    ey = R[:, 8]; m = R[:, 0] > 1.5                  # cols: t x y psi vx vy r idx ey epsi steer vcmd odom_age
    print(f'  lateral error after the ramp: rms {np.sqrt(np.mean(ey[m] ** 2)) if m.any() else float("nan"):.3f} m, worst {np.max(np.abs(ey)):.3f} m '
          f'at ({R[np.argmax(np.abs(ey)), 1]:+.2f}, {R[np.argmax(np.abs(ey)), 2]:+.2f}); steering at the limit {100 * np.mean(np.abs(R[:, 10]) >= SMAX - 1e-6):.0f} % of the time')
    print(f'  car covered x {R[:, 1].min():+.2f}..{R[:, 1].max():+.2f}, y {R[:, 2].min():+.2f}..{R[:, 2].max():+.2f}  '
          f'(path: x {xy[:, 0].min():+.2f}..{xy[:, 0].max():+.2f}, y {xy[:, 1].min():+.2f}..{xy[:, 1].max():+.2f})')
    print(f'  odometry age (stamp to use): median {np.median(R[:, 12]) * 1e3:.0f} ms, worst {np.max(R[:, 12]) * 1e3:.0f} ms')
print(f'  log: {out}')
sys.stdout.flush(); os._exit(0)                      # (desk test: shutting the executor down from here could hang)
PY
