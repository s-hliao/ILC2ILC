# F1TENTH ILC2Real car videos

Top-down views of the recorded runs (`../trajectories/`, record_car.py) on the five multi-body "real" cars, MP4 (H.264;
pause and scrub in any player), real time. Grey: the plan; the path is coloured by |sideslip| (light 0 to dark 35 deg);
blue: the car and its heading; red: its velocity (the angle between them is the sideslip). Regenerate:
`python make_car_videos.py --root <results root>` after `record_car.py`.

| folder | contents |
|---|---|
| `00_compare_all_methods/<car>_<plan>.mp4` | every method side by side on the beta-25 plan of one track, 3 laps without reset, with the sideslip against the plan's |
| `01_ours/` | ours (nominal-sim learner): the hardware stage (24 laps), the 8 evaluation plans zero-shot and after it |
| `02_ours_plus_dr_finetune/`, `03_ours_plus_dr_scratch/` | our learner + DR (A: DR fine-tune, B: DR from scratch): hardware stage, before / after |
| `04_ppo_dr/` | PPO+DR, then our hardware stage: zero-shot, after 2 and after 24 real laps |
| `05_rma/`, `07_plan_lqr/` | zero-shot only |
| `06_fada/` | FADA zero-shot and after its own 24-lap adaptation |

Files: `hardware_stage_<car>.mp4` (one panel per training plan, iteration by iteration, earlier iterations faded) and
`<when>_eval_<car>.mp4` (the 8 evaluation plans at once: beta 25 / 21 / 14 / 0 on both tracks, 21 and 14 never trained
on; 3 laps without reset). Cars: real_nom, real_mass (+25 % mass), real_mu (friction x0.8), real_act (actuators), real_lag
(tire stiffness / steering lag).
