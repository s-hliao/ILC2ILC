# Nov 5 plastic zero-shot trials (workstation ~/car_runs/archive_20261105/car_sessions/1105b_plastic), analysed 2026-11-07 ~20:00
Script: scratchpad hw_analyze.py; network part only (from the hand-over). MEASURED unless marked.
| req | tag | plan a_max | corners: plan a_req p50 / p90 | measured |v r| p50 / p90 / max | speed / plan | steer cmd beyond 0.34 rad | result |
| 00000 | rma s2f    | 4.5 | 4.38 / 5.35 | 2.27 / 2.64 / 2.86 | 0.74 | 64 % | 5 laps, nerr 2.65 (no launch) |
| 00001 | ppodr s2f  | 4.5 | 4.41 / 5.34 | 2.14 / 2.42 / 2.75 | 0.72 | 37 % | 5 laps, 2.85 |
| 00002 | rma fig8   | 3.3 | 3.14 / 3.78 | 1.91 / 2.52 / 3.74 | 0.80 | 81 % | 5 laps, 1.94 |
| 00008 | ppodr fig8 | 3.3 | 3.13 / 3.68 | 1.83 / 2.30 / 4.06 | 0.77 | 76 % | wall at 3.64 laps |
| 00004 | rma peanut | 3.3 | 3.22 / 4.36 | 2.39 / 3.16 / 4.19 | 0.92 | 65 % | wall 1.7 s after hand-over (lap 0.37, v 2.29 vs plan 2.01) |
| 00005 | ppodr peanut | 3.3 | 3.21 / 4.44 | 2.32 / 3.62 / 4.39 | 0.84 | 46 % | wall 1.5 s after hand-over (lap 0.35, v 2.49 vs plan 2.12) |
Sustained plastic grip ~2.3-2.6 m/s^2 (mu_eff ~0.24-0.27); the plans ask 3.1-4.4 (p50) in corners; commands exceed the steering limit
37-81 % of the time. Sim plastic cars: mu 0.24-0.40; the lowest (ax:mu=0.4000) zero-shot 1.80-1.87, no failure (fig1_eval_bases.json).
INFERRED (not tested): the plastic plans are beyond the real tire's grip -> the stage has no actuation authority there
(project_actuation_precondition); the real car is also worse than the lowest-grip sim car (walls vs none) -- cause not isolated.

## 2026-11-07 session (1105b_plastic, trials 00009-00025; scored 22:30 by scratchpad hw_score.py = car_score.py, network part, 1 run x 5 laps)
| track | PPO+DR zs -> stage | RMA zs -> stage |
| square2fast (practised) | 2.77 -> 2.51 (-9 %) | 2.58 -> 2.33 (-10 %) |
| fig8R100 | 5.00 wall (3.64 laps) -> 1.85 | 1.94 -> 5.00 wall (1.60 laps) |
| peanut (held-out) | 5.00 wall -> 5.00 wall (0.40 laps) | 5.00 wall -> 5.00 wall (0.42 laps) |
Stage (summary.json): task cost J PPO+DR 951 -> 724 (-24 %, predicted 651), RMA 883 -> 724 (-18 %, predicted 594) after update 1;
update 2 rejected on both (rho -0.98 / -0.64, rollback, cap 0.4 -> 0.2) -> the evaluated networks = after update 1. 6.16 real laps each.
square2fast nerr is pace-limited (pace 0.72-0.77, rms e_y 0.07-0.09): corners |v r| 2.2-2.5 vs the plan's 4.4 m/s^2.
