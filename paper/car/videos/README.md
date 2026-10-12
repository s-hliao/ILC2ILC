# F1TENTH ILC2Real car videos

Top-down views of the recorded runs (`src/ilc_f1tenth/ilc2real/trajectories/`, record_car.py) on the five multi-body "real" cars, MP4 (H.264;
pause and scrub in any player), real time. Grey: the plan; the path is coloured by |sideslip| (light 0 to dark 35 deg);
blue: the car and its heading; red: its velocity (the angle between them is the sideslip). Regenerate:
`python make_car_videos.py --root <results root>` after `record_car.py`.

| folder | contents |
|---|---|
| `00_compare_all_methods/<car>_<plan>.mp4` | every method side by side on the beta-25 plan of one track, 5 laps without reset, with the sideslip against the plan's |
| `01_ours_stopping_rule/` | the headline: our learner + DR (task objective), snapshot chosen by the sim-only stopping rule: the hardware stage (10 laps), the 8 evaluation plans zero-shot, after 10 (low budget) and after 48 real laps (asymptotic budget) |
| `02_ours_plus_dr_scratch/`, `03_ours_plus_dr_finetune/`, `04_ours_nominal_sim/` | our learner + DR (B: from scratch, A: DR fine-tune) and on the nominal sim only: hardware stage, zero-shot / after 10 laps |
| `05_ppo_dr/` | PPO+DR, then our hardware stage: zero-shot and after 10 real laps |
| `06_rma_best_nonoracle/` ... `09_fada_best_oracle/` | the RMA / FADA baselines: per method the best non-oracle and the best oracle variant (`src/paper/baselines.json`), at its own best real-data budget |
| `10_plan_lqr/` | the plans' LQR |

Files: `hardware_stage_<car>.mp4` (one panel per training plan, iteration by iteration, earlier iterations faded) and
`<when>_eval_<car>.mp4` (the 8 evaluation plans at once: beta 25 / 21 / 14 / 0 on both tracks, 21 and 14 never trained
on; 5 laps without reset). Cars: real_nom, real_mass (+25 % mass), real_mu (friction x0.8), real_act (actuators), real_lag
(tire stiffness / steering lag).
