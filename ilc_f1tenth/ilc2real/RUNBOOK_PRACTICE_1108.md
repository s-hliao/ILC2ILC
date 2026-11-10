# Car hardware, PRACTICE-ONLY protocol (2026-11-08 night; written from matcha) -- supersedes RUNBOOK_BIG_1107.md

**Design (user, 2026-11-08):** no held-out tracks on the car (the quadruped carries held-out goals). Each tire practises each track
directly and is scored on that same track: few-shot convergence. Same two tracks for both tires, so the behaviour contrast is clean:
**rubber converges in grip, plastic converges by sliding** ("with plastic the car would drift given the corrections").
Tracks: the OLD room box, the existing tape (square2fast, figure-eight fig8R100). The room-sized oval / peanut are dropped.
Envelope 0.7 m (`push_big_to_car.sh`). Networks unchanged: freePPON_s0 (PPO+DR), rmaN_s0 (RMA).

| tire | fig8 plan | square plan | behaviour |
|---|---|---|---|
| rubber | `mocap_fig8R100_b0_free_x120` (3.12 m/s) | `mocap_square2fast_b0_free_x120` (3.34 m/s, peak 9.4 m/s^2 < rubber grip 9.81) | grip |
| plastic | `mocap_fig8R100_b0_free` (2.60 m/s) | `mocap_square2fast_b0_free` (2.79 m/s, 5.5 m/s^2 vs plastic ~2.5-3.3) | sliding |

**Plastic: WAIT for matcha's go** (simulation check that practice at these speeds improves the score while sideslip rises; ~02:30).
Fallback: square2fast `_a45`, already flown 11-07 (1105b_plastic trials 9-25) -- then no plastic re-fly.
Simulation (rubber, scored on the practised fig8 x120, 6 laps): ours PPO+DR -23.6 % (18/18 runs), RMA -26.3 % (18/18); FADA -4 % / +1 %.

## 0. Once
1. Car on the network, then ON THE WORKSTATION: `bash ~/car_runs/push_big_to_car.sh` (0.7 m envelope on the car's runner; restart the runner after)
2. On the car: T1 `bash ~/car_runs/car_stack.sh`; T2 `bash ~/car_runs/car_runner.sh rubber` (session /home/kathy/car_sessions/<MMDD>_rubber; banner: envelope 0.70)
3. Workstation, every terminal: `source ~/car_runs/car_env.sh rubber` (plastic later: `car_env.sh plastic`). Stale mount:
   `fusermount -u /home/kathy/car_sessions; bash ~/car_runs/mount_car_sessions.sh`

## 1. Per track (rubber: fig8 FIRST, then square). Z = the track's plan from the table above
```bash
source ~/car_runs/car_env.sh rubber
Z=mocap_fig8R100_b0_free_x120; TR=fig8          # then: Z=mocap_square2fast_b0_free_x120; TR=square
# a) zero-shot, 1 run x 5 laps per network
bash ~/car_runs/car_eval.sh --session-dir $S --policy runs/freePPON_s0/policy_final.npz --tag ppodr_zs_rubber_$TR --plan $Z --laps 5 --runs 1
bash ~/car_runs/car_eval.sh --session-dir $S --policy runs/rmaN_s0/policy_final.npz   --tag rma_zs_rubber_$TR   --plan $Z --laps 5 --runs 1
# b) the stage on this track, one network at a time (3 trials x 2 laps each; place the car with car_pos.sh, Enter per trial)
bash ~/car_runs/stage_track.sh ppodr rubber $TR
bash ~/car_runs/stage_track.sh rma rubber $TR
# c) the adapted networks, same track
for NET in freePPON rmaN; do bash ~/car_runs/car_eval.sh --session-dir $S --policy runs/hw/real_${NET}_rubber_${TR}_1108/real_policy.npz \
  --tag ${NET}_stage_rubber_$TR --plan $Z --laps 5 --runs 1; done
```
Per tire: 2 tracks x (2 zero-shot + 2 x 3 stage trials + 2 evaluations) = 20 flights, ~1.5 h. Plastic: the same with `car_env.sh plastic`,
`Z=mocap_square2fast_b0_free; TR=square` / `Z=mocap_fig8R100_b0_free; TR=fig8`, tags `_plastic_`, after matcha's go.

## 2. Score
`python ~/car_runs/car_score.py $S` (network part from the hand-over; nerr = max(rms e_y / 10 cm, (1 - pace) / 0.1), crash = 5).
matcha adds the behaviour metric: body sideslip |beta| = |atan2(vy, vx)| from x_true, zero-shot vs after the stage, per tire.
