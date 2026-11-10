# Plastic, room-sized tracks (2026-11-07 night; written from matcha)

Why: the 11-05 / 11-07 plastic flights showed the real car's steering-limited curvature falls with speed (measured at full lock:
0.98 1/m at 1.4 m/s, 0.89 at 1.6, 0.72 at 1.9, 0.60 at 2.0-2.25). The old tracks' turns (R 1.15-1.35 m) cannot be driven at >= 2 m/s,
so square2fast / peanut / fig8R100 crashed or ran pace-limited. User: plastic must run >= 2 m/s; room ~3.5 m (x) x 5 m (y);
envelope widened to 0.7 m. Networks unchanged (freePPON_s0, rmaN_s0): no retraining.

## Tracks (make_big_tracks.py, make_fig8_track.py; plans trajopt_drift.py --a-max A --vstar 5.5, sideslip-free, as the runbook)
| role | plan | shape | speed (plan) |
|---|---|---|---|
| PRACTICE | `mocap_oval165_b0_free_a26` | stadium, turns R 1.65 m, straights along y, counter-clockwise | lap mean 2.29 m/s, corners 2.0 |
| HELD OUT | `mocap_bean165CW_b0_free_a26` | two R 1.65 lobes + gentle concave waist (9 cm), clockwise | lap mean 2.16 m/s |
| HELD OUT (extra, optional) | `mocap_fig8R115_b0_free_a24` | figure-eight, lobes R 1.15 m (the largest that fits 5 m) | lap mean 1.72 m/s |
Footprint (room frame, as designed): oval / bean x -2.40..0.90, y -2.70..2.10 (3.3 x 4.8 m), centred on the old box centre (-0.75, -0.30);
fig8 x -1.91..0.41, y -2.64..2.04. Centre lines: `~/car_runs/HW_TRACK_mocap_oval165_centerline.csv`, `..._bean165CW_...`; picture
`~/car_runs/HW_TRACK_big_1107.png`. Starts: oval (0.90, -1.05) heading +90 deg; bean (0.86, 0.07) heading -102 deg.
**Placement:** net_runner adds `~/car_runs/track_offset.txt` (now 0.5 0.0) to every plan. Set it so the footprint sits in the real
3.5 x 5 m area, and keep ~0.7 m clear around it (the envelope).
Simulation (12 plastic sim cars, zero-shot, no failures): oval PPO+DR 1.22 / RMA 1.52 (pace-limited, 85-88 % of plan speed), bean 0.79 / 0.64.

## 0. Once per session
1. Car on the network, then ON THE WORKSTATION: `bash ~/car_runs/push_big_to_car.sh`
   (copies the new tracks + plans to the car, adds `-p safety_ey:=0.7` to the car's car_runner.sh; backup car_runner.sh.bak_pre_0p7)
2. On the car (as before): T1 `bash ~/car_runs/car_stack.sh`; T2 `bash ~/car_runs/car_runner.sh plastic 1107big`
   (new session folder /home/kathy/car_sessions/1107big_plastic; its banner must say envelope 0.70)
3. Workstation, every terminal: `source ~/car_runs/car_env_big.sh` (= car_env.sh plastic 1107big + the new P / G / A, envelope 0.7)
   If the session mount is stale: `fusermount -u /home/kathy/car_sessions; bash ~/car_runs/mount_car_sessions.sh`
4. Slow lap to check placement + mocap coverage (on the car): `bash ~/car_runs/slow_lap.sh mocap_oval165_b0_free_a26 0.6 1`
   and `bash ~/car_runs/car_pos.sh mocap_oval165_b0_free_a26` -- watch for mocap gaps at the far ends in y.

## 1. Zero-shot baselines (needed whatever the simulation check says) -- 1 run x 5 laps each
```bash
source ~/car_runs/car_env_big.sh
for T in oval bean fig8; do        # drop fig8 if not flying it
  bash ~/car_runs/car_eval.sh --session-dir $S --policy runs/freePPON_s0/policy_final.npz --tag ppodr_zs_plastic_$T --plan ${P[$T]} --laps 5 --runs 1
  bash ~/car_runs/car_eval.sh --session-dir $S --policy runs/rmaN_s0/policy_final.npz --tag rma_zs_plastic_$T --plan ${P[$T]} --laps 5 --runs 1
done
```
Fly them in order (the runner announces each; place the car with car_pos.sh, Enter). Tell matcha after the first oval flight: it
checks corner grip / steering saturation against the plan. If the oval crashes in the corners even at 0.7 m: the a_max 2.4 plans
exist (`mocap_oval165_b0_free_a24`, corners 1.92 m/s, `mocap_bean165CW_b0_free_a24`).

## 2. Stage -- WAIT for matcha's go (simulation check, ~00:30: does oval practice improve the held-out bean / not hurt the others)
```bash
bash ~/car_runs/stage_run_big.sh ppodr plastic 1107big      # 3 trials x 2 laps on the oval, the runbook's SB command
bash ~/car_runs/stage_run_big.sh rma plastic 1107big        # then RMA (--z-path 1)
```
Outputs runs/hw/real_{freePPON,rmaN}_plasticbig_stage (separate from today's 1105b stages). If matcha says practise oval + fig8,
set G="mocap_oval165_b0_free_a26 mocap_fig8R115_b0_free_a24" before running (or matcha will send the exact command).

## 3. Evaluate the adapted networks (same protocol)
```bash
source ~/car_runs/car_env_big.sh
for NET in freePPON rmaN; do for T in oval bean fig8; do
  bash ~/car_runs/car_eval.sh --session-dir $S --policy runs/hw/real_${NET}_plasticbig_stage/real_policy.npz \
    --tag ${NET}_stage_plastic_$T --plan ${P[$T]} --laps 5 --runs 1
done; done
```

## 4. Score
`python ~/car_runs/car_score.py $S` (network part from the hand-over; nerr = max(rms e_y / 10 cm, (1 - pace) / 0.1), crash = 5).
