# DistILC paper material

Everything the paper uses, outside the code packages: the abstract, the related-work list, every figure and the
supplementary videos, for both robots. The scripts that make them stay in their packages (they import package code and
read run data); each writes here.

| path | contents | made by |
|---|---|---|
| `abstract.txt` | the abstract draft | |
| `baselines.json` | the RMA / FADA rows every figure shows: per robot, of each method, the best non-oracle and the best oracle variant (best success at any real-data budget), and the low / asymptotic real-data budgets (quadruped 24 / 96 jumps per robot) | `select_baselines.py` (rerun when a candidate run finishes; then `ilc_mjx/scripts/eval_picks.py` flies the quadruped picks under the perturbations, `ilc_f1tenth/ilc2real/car_stop_axes.sh` the car's) |
| `REFERENCES.md` | related work, grouped by relation to DistILC, read-status tags, contrasts, open items | by hand |
| `quadruped/figures/` | `01`-`13`: methods by perturbation kind, per perturbation / robot, sim sweeps, goal-plane maps, real-jump budget, tolerance, sim training, per axis, what the stage absorbs, trained x tested, axis interactions, real-data matching | `ilc_mjx/paper/ilc2real_figs.py` (13: `quad_figures.py`) |
| `quadruped/figures_summary/` | `fig1`-`fig8`: factorial, methods, budget, hardware-stage arms, sim variants, efficiency, goal maps, real-data matching | `ilc_mjx/paper/quad_figures.py` |
| `quadruped/transfer_overview/` | the transfer overview pages | `ilc_mjx/paper/transfer_overview.py --save` |
| `quadruped/plane/` | the goal-plane grid | `ilc_mjx/paper/plane_grid.py` |
| `quadruped/videos/` | MP4s, one folder per method (see its README) | `ilc_mjx/scripts/make_animations.py` |
| `car/figures/` | `fig1`-`fig15` at the car's real budget (0 / 2 / 10 laps; success and 5-lap completion), `traj_*` trajectory plots | `ilc_f1tenth/ilc2real/car_figures.py`, `car_study_figs.py`, `stop_analysis.py` (fig13), `car_traj_figs.py` |
| `car/figures/fig16_ladder_per_car.png` | the low-friction fix: per-car b25 success (5 cars + friction x0.9 / x0.7 / x0.6), ours without the ladder, the reference ladder (10 laps) on sim seeds 0 / 2 / 1, exact-dynamics RL oracles at 5,000 laps | `ilc_f1tenth/ilc2real/ladder_figs.py` |
| `car/figures/fig17_exact_matching.png` | the car under EXACT dynamics: b25 success and crashes vs real laps per car, RMA (true latent) / PPO+DR on each multi-body car, FADA (exact IDMs, + fixed lookahead, + no walls), ours (9 laps) | `ilc_f1tenth/ilc2real/ladder_figs.py` |
| `car/figures/fig18_ladder_rungs.png` | the rung (nominal / 0.8x / 0.6x-friction plan) each car settled on per track and seed, with its b25 success; the rungs' speed price (90 % / 78 %) | `ilc_f1tenth/ilc2real/ladder_figs.py` |
| `car/figures/old_24lap/` | the superseded 24-lap car figures | (archive) |
| `car/videos/` | MP4s, one folder per method (see its README) | `ilc_f1tenth/ilc2real/make_car_videos.py` |

Regenerate (from `src/`):

```bash
python3 paper/select_baselines.py            # first: the picked baselines and the budgets
# quadruped (the ilcmjx env; reads ~/ilc_ws/log/dilc/plane, outside the repo)
~/miniconda3/envs/ilcmjx/bin/python ilc_mjx/paper/ilc2real_figs.py
~/miniconda3/envs/ilcmjx/bin/python ilc_mjx/paper/quad_figures.py
# car (the f1t env; reads ilc_f1tenth/ilc2real/runs)
cd ilc_f1tenth/ilc2real && source env_tight.sh
~/miniconda3/envs/f1t/bin/python car_figures.py && ~/miniconda3/envs/f1t/bin/python car_study_figs.py
~/miniconda3/envs/f1t/bin/python stop_analysis.py && ~/miniconda3/envs/f1t/bin/python car_traj_figs.py
```

Results notes stay with the code: `ilc_f1tenth/ilc2real/NOTES.md` (car), `~/ilc_ws/log/dilc/plane/NOTES.md` (quadruped).

## pogo/ (toy 2-D SLIP pogo stick, 2026-10-13)
- figures/fig1..fig8 (methods, budget, coverage, positivity, controls, cost, goal plane, data matching) -- src/ilc_pogo/figures.py
- videos/*.gif (matplotlib animations) -- src/ilc_pogo/animate_pogo.py; results in src/ilc_pogo/NOTES.md

## final/ (paper-ready figures, 2026-10-28)
- final/*.{pdf,png}: the figures as they go into the paper; each generator lives in its own subfolder of final/ and writes
  its outputs to final/ itself.
- method_pogo_ilc.{pdf,png} (methodology schematic: batches at V goals -> policy update -> ... -> converged; hand-set,
  no run data) -- final/pogo_method/method_diagram.py (run from src/ with any python that has matplotlib, e.g.
  ~/miniconda3/envs/mj_quad/bin/python paper/final/pogo_method/method_diagram.py)
- pogo_inout.{pdf,png} (pogo goal generalization -- goals flown / inside their hull / outside it -- at matched sim hours
  (11 h, zero-shot), matched real rollouts (36) and full budgets; the tables' protocol and normalized error, 6 sticks, 3
  seeds) -- final/pogo_inout/inout_figs.py from inout_matched.json (~/ilc_ws/log/pogo_servo/inout_matched.py +
  inout_matched_data.py)
- pogo_budget_{sim,real}.{pdf,png} (normalized error vs simulated hours, zero-shot / vs real rollouts per robot, by
  coverage tier A / B / C) -- final/pogo_budget/budget_figs.py from budget_data.json (~/ilc_ws/log/pogo_servo/budget_data.py)
