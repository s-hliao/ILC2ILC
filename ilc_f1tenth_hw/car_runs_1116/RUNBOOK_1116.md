# Car hardware runbook, 2026-11-16 (written from matcha; tracks updated 2026-11-19, rubber -> peanut x1.5 late 11-19; plastic -> rotated R1.0 figure-8 on 11-20)

Two stage experiments. **2026-11-19: both tracks moved one step faster** (sim speed ladder, `ilc2real_ubefix/screen_ladder_1116.sh`, same stage settings), so the zero-shot error is larger and the improvement more visible:

| # | tire | track | plan | lap time | sim: lap RMS lateral error per 2-lap iteration (3 stage seeds) |
|---|---|---|---|---|---|
| 1 | rubber | peanut **x1.5** (clockwise) | `mocap_peanutM_b0_free_x150` | 2.55 s | final network, 3 seeds: 34 -> 14-20 cm lap RMS, peak 52 -> 22-40 cm, never beyond 0.5 m; pace kept (0.79 -> 0.75-0.79 of the plan) |
| 2 | plastic | figure-8 R **1.0** m, **x0.95**, rotated 25 deg (11-20) | `mocap_fig8R100_b0_free_x095_rot25` | 5.37 s | final network, 3 seeds: lap 1 45 -> 17 / 19 / 24 cm, lap 2 21 -> 7.5 / 34 / 18 cm, peak 87 -> 27 / 52 / 35 cm; pace 0.74 -> 0.65 of the plan |

- **Why the R1.0 figure-8, rotated (11-20):** the R1.15 figure-8 left the mocap room (y down to -3.47 m). The R1.0 one at x0.95 is rotated 25 deg and shifted (0.05, 0.60) m; every replayed sim path, zero-shot (peak 87 cm off the line) included, stays >= 0.34 m inside the tracked area x [-2.69, 2.22], y [-2.86, 2.80] (11-08/09 mocap positions). Rotation changes nothing else: replays on the rotated and unrotated plans agree to < 0.7 cm. x0.90 was the alternative (lap 1 34 -> 18-20 cm, peak 67 -> 30-34 cm): smaller start, smaller gain.
- **RMA (sim replays with the launch, 11-20, 3 seeds):** rubber peanut x1.5: lap 1 24 -> 10 / 11.5 / 11 cm, peak 44 -> 20-25 cm, pace kept (0.78 -> 0.79-0.80): strong. Plastic rotated figure-8: RMA starts much better than PPO (lap 1 23 cm, peak 46 cm) and gains less: lap 1 -> 11.5 / 18 / 17 cm, peak -> 20 / 37 / 33 cm, lap 2 9.6 -> 6.2 / 15.6 cm; one seed (9001) slowed to 0.55 of the plan's pace and made 0.45 laps in the replay window. **So: RMA on rubber, PPO on plastic** (the user's split) is also where each shows the most.

- **Why peanut x1.5 (11-19):** the user asked for a much larger zero-shot deviation (x1.25 was 12 cm), track-based, inside the mocap room (1.0 m envelope). x1.5 starts at 34 cm and the final network halves it on every seed. Faster plans (x1.65: 46-52 cm) did not converge reliably; figfast x1.5 improves less and partly by slowing down.
- **Speed:** x1.5 peaks at 5.55 m/s on the plan (the car runs ~79 % of the plan's pace). Check the VESC speed limit allows it.
- **Launch: keep 8 A / hand-over at 1.0 m/s.** A gentler plastic launch (4 A to 2.0 m/s) left a steady error the stage could not reduce (sim, 6 runs).
- **Sim numbers are replays of the saved networks with the hardware launch** (`ilc2real_ubefix/inspect_runs_1119.py`), not the trial logs.
- **Videos (sim, final network, hardware launch):** `src/ilc_f1tenth/videos/car_rubberpeanutx150final_stage_1119.mp4`, `car_plasticfig8x095final_stage_1119.mp4` (the old R1.15 plastic track; a video for the rotated R1.0 one is not made yet).
- **Previous tracks, kept as fallback** (still accepted by `stage_track_1116.sh` as `peanut125` / `peanut120` / `fig8x095` / `fig8x090`; the R1.15 figure-8s do not fit the room): peanut x1.2, 8.9 -> 4.4-5.9 cm; figure-8 x0.9, 32 -> 14-19 cm, both monotone in every seed.

- **Not used:** peanut x1.5, which needs 5.55 m/s and 13.5 m/s^2 (beyond the real car). figure-8 at full speed, where the plastic stand-in leaves the 1.0 m envelope zero-shot.
- **Stand-ins:** rubber = friction x1.667, plastic = friction x0.4. The real tires' gap is what we are about to measure.

## Settings: what changed vs 11-08

`stage_track_1116.sh` = `stage_track_v2.sh` with the sim screens' settings:

| setting | 11-08 | now |
|---|---|---|
| iterations | 3 | **4** (2 laps each = 8 laps per network per tire) |
| `--cap` | 0.4 | **0.8** |
| `--anchor-w` | 20 | **10** |
| pace row scale | 0.24 | **1.0** |
| `--tr-max` | default | **0.8** |
| also added | | `--rollback 0.7 --crash-rollback 1` |

- Masking is unchanged: `--skip-below 0.5`.
- **Lap budget:** 8 laps is above the usual 6. Set `ITERS=3` for 6 laps; the sim gain is mostly in by iteration 2-3.

## Before flying
1. **Install the car-side mocap fixes** from `src/ilc_f1tenth_hw/car_side_fix_1115/` (README there): the pose gate drops heading flips and position jumps. 56 of 255 flights were voided by these on 11-08/09.
2. **Fix the VESC wire; charge the batteries.**
3. **Plans:** `mocap_peanutM_b0_free_x150{.npz,_K.npy}` and `mocap_fig8R100_b0_free_x095_rot25{.npz,_K.npy}` are on the workstation (`~/ros2_ws/src/ilc_f1tenth/ilc2real/plans_nom06/`), identical to the sim's. dl-car was offline for the last copy: copy both there too if the car-side code reads plans.
   - `ssh car "ls ~/ILC2ILC/ilc_f1tenth/ilc2real/plans_nom06 | grep -c 'peanutM_b0_free_x150\|fig8R100_b0_free_x095_rot25'"` should print 4.
   - **Check the start pose:** the rotated figure-8 starts at (-0.50, 0.01) m, heading 50 deg (0.88 rad). Place the car there.
4. **Copy `stage_track_1116.sh`** to `~/car_runs/` again (already there on the workstation, updated late 11-19: accepts `peanut150`, `peanut125` and `fig8x095`; 11-20 adds `fig8r100`).

## Sequence, per tire
```bash
source ~/car_runs/car_env.sh <tire>
# rubber: Z=mocap_peanutM_b0_free_x150 TR=peanut150     plastic: Z=mocap_fig8R100_b0_free_x095_rot25 TR=fig8r100
bash ~/car_runs/car_eval.sh --session-dir $S --policy runs/freePPON_s0/policy_final.npz --tag ppodr_zs_<tire>_$TR --plan $Z --laps 5 --runs 1
bash ~/car_runs/stage_track_1116.sh ppodr <tire> $TR          # 4 trials x 2 laps; Enter per trial
bash ~/car_runs/car_eval.sh --session-dir $S --policy runs/hw/real_freePPON_<tire>_${TR}_1116/real_policy.npz \
  --tag freePPON_stage_<tire>_$TR --plan $Z --laps 5 --runs 1
```
(RMA: the same with `rma` / `runs/rmaN_s0`.)

## Video
**Phone video of every trial**, plus the mocap logs. matcha renders the sim-style video from them: car on the track + lateral-offset trace per iteration, as in `car_plasticfig8x095_stage_1116.mp4`.
