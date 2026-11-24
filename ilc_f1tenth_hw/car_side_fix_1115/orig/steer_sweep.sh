#!/bin/bash
# steer_sweep.sh (2026-11-04), ON THE CAR: can the steering reach the model's +-0.34 rad without binding?
# (HW_RUNBOOK_N6_1031.md section 0.3: with vesc.yaml's servo_min/max 0.15/0.85 the driver clips at +0.313 / -0.263 rad.)
#   bash ~/car_runs/steer_sweep.sh          car ON A STAND (wheels off the ground), battery in, stack NOT running
#   DRYRUN=1 bash ~/car_runs/steer_sweep.sh  the prompts and logging only: no VESC driver, nothing moves
# What it does: starts ONLY vesc_driver_node (no ackermann_to_vesc, mux, teleop: nothing commands the motor) on an isolated
# ROS domain, with its servo limits widened to 0.05..0.95 for this test only (vesc.yaml is not touched), then steps the steering
# LEFT out to +0.34 rad and RIGHT out to -0.34 rad (vesc.yaml's map servo = -1.2135 * delta + 0.5304; never beyond 0.34).
# Every step: a slow ramp from centre, a hold (at most 8 s) while you watch, back to centre. Answer at each step:
#   Enter   = moved smoothly, no buzz        b = binding (stalled / buzzing / wheel at its stop): that side ends here
#   a number = the wheel angle you measured, in degrees (optional), then Enter
# Ctrl-C at any time: back to centre, driver stopped. Log: ~/car_runs/steer_sweep_<time>.csv
set -e
if pgrep -f 'opti_launch.py|bringup_launch.py|vesc_driver_node|ackermann_to_vesc_node' >/dev/null; then
  echo "!!! the car stack (or a VESC driver) is running: stop it first (the VESC serial port is exclusive)"
  pgrep -af 'opti_launch.py|bringup_launch.py|vesc_driver_node|ackermann_to_vesc_node' | cut -c1-120; exit 1; fi
source /opt/ros/foxy/setup.bash; source ~/lla_ws/install/setup.bash
export ROS_DOMAIN_ID=78 ROS_LOCALHOST_ONLY=1 PATH=/usr/bin:$PATH; unset CYCLONEDDS_URI
LOG=~/car_runs/steer_sweep_$(date +%H%M%S).csv
DRV=
cleanup() { [ -n "$DRV" ] && { kill -INT $DRV 2>/dev/null; sleep 1; kill -9 $DRV 2>/dev/null; }; true; }
trap cleanup EXIT
read -p ">>> car on a stand, wheels free, hands clear of the steering? Enter to start, Ctrl-C to abort " _
if [ -z "$DRYRUN" ]; then
  [ -e /dev/sensors/vesc ] || { echo "!!! /dev/sensors/vesc missing: VESC powered and plugged in?"; exit 1; }
  CFG=$(ros2 pkg prefix f1tenth_stack)/share/f1tenth_stack/config/vesc.yaml
  $(ros2 pkg prefix vesc_driver)/lib/vesc_driver/vesc_driver_node --ros-args -r __node:=vesc_driver_node \
    --params-file $CFG -p servo_min:=0.05 -p servo_max:=0.95 > ${LOG%.csv}_driver.log 2>&1 &
  DRV=$!; sleep 3
  kill -0 $DRV 2>/dev/null || { echo "!!! the VESC driver exited:"; tail -5 ${LOG%.csv}_driver.log; exit 1; }
  grep -iE "error|fail" ${LOG%.csv}_driver.log | head -3 || true
fi
python3 /dev/fd/3 $LOG "${DRYRUN:-}" 3<<'PY'
import sys, time, select, rclpy
from std_msgs.msg import Float64
log, dry = sys.argv[1], bool(sys.argv[2])
GAIN, OFF, LIM = -1.2135, 0.5304, 0.34
servo_of = lambda d: GAIN * d + OFF
rclpy.init(); n = rclpy.create_node('steer_sweep')
pub = n.create_publisher(Float64, 'commands/servo/position', 10)
echo = {'v': None}
n.create_subscription(Float64, 'sensors/servo_position_command', lambda m: echo.update(v=m.data), 10)
cur = [OFF]
def send(v, secs):                      # hold v for secs, publishing at 50 Hz
    t = time.monotonic()
    while time.monotonic() - t < secs:
        pub.publish(Float64(data=float(v))); rclpy.spin_once(n, timeout_sec=0.0); time.sleep(0.02)
def ramp(v, rate=0.15):                 # servo units per second: centre to the end of travel in ~3 s
    a = cur[0]; k = max(1, int(abs(v - a) / rate / 0.02))
    for i in range(1, k + 1):
        send(a + (v - a) * i / k, 0.02)
    cur[0] = v
def ask(timeout):
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.readline().strip() if r else None
f = open(log, 'w'); f.write('side,delta_rad,servo_cmd,servo_applied,answer\n')
ramp(OFF); send(OFF, 1.0)
print(f"centre: servo {OFF:.4f}, driver applied {echo['v']}" + ('  (DRYRUN: no driver)' if dry else ''))
steps = [0.20, 0.26, 0.30, 0.313, 0.32, 0.33, 0.34]
try:
    for side, sgn in (('LEFT', +1), ('RIGHT', -1)):
        print(f"\n=== {side} ===")
        for d in steps:
            dl = sgn * d; s = servo_of(dl)
            ramp(s); echo['v'] = None; send(s, 0.3)
            print(f"  {side} {d:.3f} rad ({d*57.2958:.1f} deg): servo {s:.4f}, driver applied {echo['v']}"
                  f"{'  <- beyond the current vesc.yaml limit' if not 0.15 <= s <= 0.85 else ''}")
            t0 = time.monotonic(); ans = None
            print("    Enter = smooth | b = binding | or measured deg + Enter (back to centre in 8 s) > ", end='', flush=True)
            while ans is None and time.monotonic() - t0 < 8.0:
                send(s, 0.1); ans = ask(0.0)
            if ans is None: print("(timeout)"); ans = 'timeout'
            f.write(f"{side},{dl:+.3f},{s:.4f},{echo['v']},{ans}\n"); f.flush()
            ramp(OFF); send(OFF, 0.5)
            if ans.lower().startswith('b'):
                print(f"    binding noted: {side} ends at {d:.3f} rad"); break
finally:
    ramp(OFF); send(OFF, 1.0); f.close()
    print(f"\nback at centre. log: {log}")
    n.destroy_node(); rclpy.shutdown()
PY
