#!/bin/bash
# steer_cal.sh (2026-11-04), ON THE CAR: autonomous steering calibration (user: "do an autonomous calibration, i don't want
# to have to reset it"). Drives the figure-eight (both turn directions in one lap) slowly with pure pursuit + a small steering
# wiggle, starting wherever the car is on/near the path, then fits the wheel angle the yaw rate implies (kinematic: at
# 0.5 m/s the lateral acceleration is ~0.25 m/s^2, the tires barely slip) against the commanded one:
#     actual = g * commanded + b        and prints the vesc.yaml values that make actual = commanded:
#     steering_angle_to_servo_gain' = G / g,   steering_angle_to_servo_offset' = O - G * b / g
# It does NOT change vesc.yaml (that is applied afterwards, then car_stack.sh restarted).
#   bash ~/car_runs/steer_cal.sh [laps = 2] [speed = 0.5]      car near the figure-eight, pointing along it; Enter to go
#   bash ~/car_runs/steer_cal.sh fit <log.npz>                 refit an existing log
# Needs car_stack.sh up and net_runner stopped (slow_lap.sh checks both). Enter stops it.
PLAN=mocap_fig8R100_b0_free_a33
if [ "$1" != fit ]; then
  DITHER=0.04 bash ~/car_runs/slow_lap.sh $PLAN ${2:-0.5} ${1:-2} near || exit 1
  LOG=$(ls -t ~/car_runs/slow_laps/*_$PLAN.npz | head -1)
else LOG=$2; fi
CFG=$HOME/lla_ws/install/f1tenth_stack/share/f1tenth_stack/config/vesc.yaml
G=$(grep -oP '^\s*steering_angle_to_servo_gain:\s*\K[-0-9.]+' $CFG); O=$(grep -oP '^\s*steering_angle_to_servo_offset:\s*\K[-0-9.]+' $CFG)
SMIN=$(grep -oP '^\s*servo_min:\s*\K[-0-9.]+' $CFG); SMAXS=$(grep -oP '^\s*servo_max:\s*\K[-0-9.]+' $CFG)
python3 - "$LOG" "$G" "$O" "$SMIN" "$SMAXS" <<'PY'
import sys, numpy as np
f, G, O, smin, smax = sys.argv[1], *map(float, sys.argv[2:])
d = np.load(f, allow_pickle=True); R = d['rec']; L = 0.330
t, v, r, st = R[:, 0], np.hypot(R[:, 4], R[:, 5]), R[:, 6], R[:, 10]
print(f'\n=== steering fit: {f.split("/")[-1]} ({str(d["reason"])}, {len(R)} samples, wiggle {float(d["dither"]) if "dither" in d.files else 0:.2f} rad)')
best = None
for lag in range(0, 21):                                 # 0..500 ms: servo + chassis response
    de = np.arctan(L * r[lag:] / np.maximum(v[lag:], 1e-3)); dc = st[:len(st) - lag]
    m = (t[lag:] > 2.0) & (v[lag:] > 0.3) & (np.abs(np.gradient(dc)) < 0.006) & (np.abs(dc) < 0.315)   # not at the clip
    if m.sum() < 100: continue
    k = np.polyfit(dc[m], de[m], 1); res = np.sqrt(np.mean((np.polyval(k, dc[m]) - de[m]) ** 2))
    print(f'  lag {lag * 25:3d} ms: actual = {k[0]:.3f} x commanded {k[1]:+.4f} rad, rms residual {res:.4f} rad, n {m.sum()}')
    if best is None or res < best[0]: best = (res, lag, k, m.sum(), dc[m])
if best is None: print('!!! too few usable samples (did the laps run?)'); sys.exit(1)
res, lag, (g, b), nn, dcs = best
print(f'best lag {lag * 25} ms: actual = {g:.3f} x commanded {b:+.4f} rad (residual {res:.4f} rad); commanded range {dcs.min():+.3f}..{dcs.max():+.3f}, '
      f'left {np.sum(dcs > 0.05)} / right {np.sum(dcs < -0.05)} / near zero {np.sum(np.abs(dcs) <= 0.05)} samples')
G2, O2 = G / g, O - G * b / g
act = lambda s: g * (s - O) / G + b                      # the actual wheel angle at servo value s, today's map
print(f'vesc.yaml now:     steering_angle_to_servo_gain {G:.4f}, steering_angle_to_servo_offset {O:.4f}')
print(f'vesc.yaml CORRECT: steering_angle_to_servo_gain {G2:.4f}, steering_angle_to_servo_offset {O2:.4f}')
print(f'servo limits {smin} / {smax} are then the wheel angles {act(smin):+.3f} / {act(smax):+.3f} rad (model s_max 0.34)')
PY
