# Car hardware runbook, 2026-11-22 evening (written from matcha; replaces RUNBOOK_1116's two experiments; nothing here has flown yet)

Goal (user): one RMA arm and one PPO arm, one on rubber and one on plastic, one on the new rotated figure-8, each with a **large,
consistent offset (repeatable, not just a peak) that the stage visibly corrects** on video.

| # | network | tire | track (plan) | track offset | stage | expected zero-shot -> after 6 laps (sim, inferred for the car) |
|---|---|---|---|---|---|---|
| 1 | **PPO+DR** (`freePPON_s0`) | **rubber** | **new figure-8** `mocap_fig8R100_b0_free_x095_rot25` | **0.0 0.0** | **PACE = its zero-shot pace** | repeatable 27 -> 11-12 cm, worst segment 56 -> 20-21 cm, peak 68 -> 23-26 cm at x0.45 (rubber figure-8 match); pace 0.77 -> 0.71 |
| 2 | **RMA** (`rmaN_s0`) | **plastic** | **sq41free** `mocap_sq41R170CW_b0_free` (clockwise square, R 1.7 m) | 0.5 0.0 | **PACE = its zero-shot pace** | steady one-sided offset 23-29 -> 7-16 cm (worst 24-31 -> 8-19), peak 42-56 -> 20-29 cm; pace -0.03 to -0.04 |

User decisions 11-22: the arms above; **"pace target change is fine"**. One rule for BOTH arms: `--pace-target` = the arm's own zero-shot pace (rounded down to 0.05; `offset_score_1122.py` prints it).

## Why these two (all sim, 11-22; Lambda + matcha `~/car_screen_1122/ilc2real_ubefix`, logs `logs/screen1122/`, runs `runs/screen1122/`)
1. **Calibration (measured vs sim):** both hardware networks replayed on 20 plans x 9 friction levels with the hardware launch, 5 laps, scored
   like the hardware logs (`replay_cands_1122.py`; 720 runs). **Real rubber behaves like friction x0.4-0.5 of nominal** (peanut x1.5 RMA:
   hw 52 cm / +48 / pace 0.56, sim x0.5 51 / +49 / 0.55; rubber figure-8s ~x0.45), **not** the x1.667 stand-in and not nominal (x1.0 predicts
   14 cm for peanut x1.25 vs 48 real). Real plastic: figure-8s ~x0.4, sq41free ~x0.6-0.8. The networks' friction DR is ~+-20 %: the real
   rubber is outside it (a real sim-to-real gap). The sim's horizon scale matches the car's (peanut x1.25 2.2 vs 2.41 real; x1.5 2.6 vs 2.78).
2. **Stage screens** (this runbook's exact settings, 3 stage seeds, `--skip-below 0.5` and `--horizon-scale` from the sim zero-shot, 5-lap
   evals of every iteration; repeatable / worst segment / peak in cm, zero-shot -> after 3 iterations = 6 laps, per seed):

| arm | friction | zero-shot | after 6 laps | pace |
|---|---|---|---|---|
| PPO new figure-8 | x0.4 | 37 / 70 / 86 | 18/34/44, 17/33/42, 17/29/34 | 0.75 -> 0.69 |
| PPO new figure-8 | x0.45 | 27 / 56 / 68 | 17/29/33, 13/20/24, 17/28/34 | 0.77 -> 0.73 |
| **PPO new figure-8, PACE 0.77 (the rule)** | x0.45 | 27 / 56 / 68 | **12/21/26, 12/20/24, 11/20/23** | 0.77 -> 0.71 |
| PPO new figure-8 | x0.5 | 22 / 46 / 58 | 16/30/33, 15/27/31, 13/26/28 | 0.78 -> 0.76 |
| RMA sq41free, PACE 0.75 | x0.5 | 43 / 44 / 71 | 18/19/22, 13/14/33, 20/21/26 | 0.76 -> 0.72 |
| RMA sq41free, PACE 0.8 | x0.6 | 29 / 31 / 56 | 9/11/20, 9/11/26, 9/11/28 | 0.81 -> 0.77 |
| RMA sq41free, PACE 0.85 | x0.7 | 23 / 24 / 42 | 7/8/23, 16/19/29, 9/10/22 | 0.86 -> 0.83 |
| RMA sq41free, PACE 0.9 | x0.8 | 18 / 20 / 34 | 6/7/33, 6/7/30, 5/6/45 | 0.91 -> 0.88 |

   **Rejected** (same screens): every RMA arm with the plan's pace (pace target 1: diverges / crashes / no change: peanut x1.25 and x1.5,
   sq41free both tires, square x1.2); RMA peanut x1.25 with a pace target (works at x0.5 in 2 of 3 seeds, fragile at x0.3 / x0.4); RMA peanut
   x1.5 with pace target 0.55 (61/26/39); RMA on any figure-8 (zero-shot only 11-16 cm); RMA oval (29 -> 22-29); PPO square2fast_a45 (its
   11-08 37 cm is NOT reproduced by the sim: 5 cm; likely the no-braking car).
3. **Hardware evidence for arm 2:** 11-08 RMA plastic sq41free zero-shot (no braking) = repeatable 21 cm, worst 30, peak 53, lap scatter 3 cm,
   pace 0.79 (`offset_score_1122.py`); that day's stage (old settings, short horizon) took it to +15 cm.

## Gates (decide BEFORE looking at the stage result)
Score every zero-shot with `python3 ~/car_runs/offset_score_1122.py $S/res_<k>.npz`:
- **Consistent:** lap scatter < 1/3 of the repeatable offset; laps 1-4 agree.
- **Large enough for video:** repeatable >= ~20 cm (arm 2: one-sided, mean ~= repeatable) and worst segment >= ~30 cm.
- **5 laps, no wall stop.** If a gate fails, stop and report; do not stage.

## Before flying
- Battery read; Motive check: the car's mocap z 0.13-0.16 m (recalibrated 11-22).
- **Track offset per track** (car `~/car_runs/track_offset.txt`, first line, read by `car_runner.sh` at start: restart it after a change):
  - **figure-8: `0.0 0.0`.** The plan is already shifted/rotated to fit the room (RUNBOOK_1116: start (-0.50, 0.01) EKF, heading ~50-56 deg).
    At the default 0.5 the sim zero-shot at x0.4 leaves the tracked area by 0.16 m (x up to 2.38 > 2.22); at 0.0 it stays >= 0.29 m inside.
  - **sq41free: `0.5 0.0`** (as on 11-08: start (1.80, -0.31) EKF, heading -90 deg). Tight: the sim zero-shot at x0.6 touches the tracked
    area's -x edge (x -2.71 vs -2.69); the real 11-08 flights reached x -2.51, y -2.75. Watch the -x / -y corner.
  - The placement window's start check is authoritative.
- Plans: both `.npz` + `_K.npy` are on the workstation and the car (checked 11-22 19:20, 4/4).
- **Video:** tape the planned path on the floor from `~/car_runs/tape_1122/{fig8r100_offset0,sq41free_offset05}.csv` (every 0.25 m; EKF and
  Motive coordinates, Motive = -EKF). Phone video of the zero-shot and the final eval from the same spot; keep the mocap logs for the overlay.

## Sequence
```bash
# ARM 1: PPO, rubber, new figure-8 (car_runner.sh rubber running, track offset 0 0)
source ~/car_runs/car_env.sh rubber; Z=mocap_fig8R100_b0_free_x095_rot25
#   slow lap first (bash ~/car_runs/oncar.sh slow_lap.sh ...) to check the start and the tape
bash ~/car_runs/car_eval.sh --session-dir $S --policy runs/freePPON_s0/policy_final.npz --tag ppodr_zs_rubber_fig8r100 --plan $Z --laps 5 --runs 1
python3 ~/car_runs/offset_score_1122.py $S/res_<k>.npz                       # gates; prints PACE=<p>
python ~/ros2_ws/src/ilc_f1tenth/ilc2real/horizon_scale.py $S/res_<k>.npz    # -> HSCALE
PACE=<p> HSCALE=<x> bash ~/car_runs/stage_track_1122.sh ppodr rubber fig8r100   # 3 trials x 2 laps; Enter per trial
bash ~/car_runs/car_eval.sh --session-dir $S --policy runs/hw/real_freePPON_rubber_fig8r100_1122_h<x>_p<p>/real_policy.npz \
  --tag freePPON_stage_rubber_fig8r100 --plan $Z --laps 5 --runs 1
python3 ~/car_runs/offset_score_1122.py $S/res_<zero-shot>.npz $S/res_<final>.npz

# ARM 2: RMA, plastic, sq41free (swap tires; car_runner.sh plastic; track offset 0.5 0)
source ~/car_runs/car_env.sh plastic; Z=mocap_sq41R170CW_b0_free
bash ~/car_runs/car_eval.sh --session-dir $S --policy runs/rmaN_s0/policy_final.npz --tag rma_zs_plastic_sq41free --plan $Z --laps 5 --runs 1
python3 ~/car_runs/offset_score_1122.py $S/res_<k>.npz                       # gates; prints PACE=<p>
python ~/ros2_ws/src/ilc_f1tenth/ilc2real/horizon_scale.py $S/res_<k>.npz    # -> HSCALE
PACE=<p> HSCALE=<x> bash ~/car_runs/stage_track_1122.sh rma plastic sq41free # refuses to start without PACE
bash ~/car_runs/car_eval.sh --session-dir $S --policy runs/hw/real_rmaN_plastic_sq41free_1122_h<x>_p<p>/real_policy.npz \
  --tag rmaN_stage_plastic_sq41free --plan $Z --laps 5 --runs 1
```
- The final eval is flown exactly like the zero-shot (same start, 5 laps, similar battery). Report repeatable, worst segment, peak AND pace.
- Lap budget: `ITERS` defaults to 3 = 6 real laps per arm.

## Files (11-22)
- `~/car_runs/stage_track_1122.sh`: stage_track_1116.sh (unchanged, kept) + these two arms, `PACE=` (required for both), `ITERS` default 3.
- `~/car_runs/offset_score_1122.py`: the consistent-offset score + PACE.
- `~/car_runs/tape_1122/*.csv`: floor tape lines.
- Workstation `hw_stage.py` vs the screened copy: only `--tr-reject` / `--gain-current` / `--skip-until-frac` (defaults = screened behaviour)
  and the same `--horizon-scale`.

## Open
- DONE 19:26: PPO figure-8 with the same pace rule at x0.45: 27 -> 11-12 cm repeatable on all 3 seeds (better and tighter than the plan's
  pace, 13-17) -> one rule for both arms.
- One sim rerun of an identical screen changed one seed (45 vs 18 cm): read single seeds as approximate.
- Plastic and rubber grip are calibrated from 11-08 (no braking) and 11-22 (braking) flights; the zero-shots above are the real test.
