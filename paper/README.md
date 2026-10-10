# DistILC paper material

Everything the paper uses, outside the code packages: the abstract, the related-work list, every figure and the
supplementary videos, for both robots. The scripts that make them stay in their packages (they import package code and
read run data); each writes here.

| path | contents | made by |
|---|---|---|
| `abstract.txt` | the abstract draft | |
| `REFERENCES.md` | related work, grouped by relation to DistILC, read-status tags, contrasts, open items | by hand |
| `quadruped/figures/` | `01`-`13`: methods by perturbation kind, per perturbation / robot, sim sweeps, goal-plane maps, real-jump budget, tolerance, sim training, per axis, what the stage absorbs, trained x tested, axis interactions, real-data matching | `ilc_mjx/paper/ilc2real_figs.py` (13: `quad_figures.py`) |
| `quadruped/figures_summary/` | `fig1`-`fig8`: factorial, methods, budget, hardware-stage arms, sim variants, efficiency, goal maps, real-data matching | `ilc_mjx/paper/quad_figures.py` |
| `quadruped/transfer_overview/` | the transfer overview pages | `ilc_mjx/paper/transfer_overview.py --save` |
| `quadruped/plane/` | the goal-plane grid | `ilc_mjx/paper/plane_grid.py` |
| `quadruped/videos/` | MP4s, one folder per method (see its README) | `ilc_mjx/scripts/make_animations.py` |
| `car/figures/` | `fig1`-`fig15` at the car's real budget (0 / 2 / 10 laps; success and 5-lap completion), `traj_*` trajectory plots | `ilc_f1tenth/ilc2real/car_figures.py`, `car_study_figs.py`, `stop_analysis.py` (fig13), `car_traj_figs.py` |
| `car/figures/old_24lap/` | the superseded 24-lap car figures | (archive) |
| `car/videos/` | MP4s, one folder per method (see its README) | `ilc_f1tenth/ilc2real/make_car_videos.py` |

Regenerate (from `src/`):

```bash
# quadruped (the ilcmjx env; reads ~/ilc_ws/log/dilc/plane, outside the repo)
~/miniconda3/envs/ilcmjx/bin/python ilc_mjx/paper/ilc2real_figs.py
~/miniconda3/envs/ilcmjx/bin/python ilc_mjx/paper/quad_figures.py
# car (the f1t env; reads ilc_f1tenth/ilc2real/runs)
cd ilc_f1tenth/ilc2real && source env_tight.sh
~/miniconda3/envs/f1t/bin/python car_figures.py && ~/miniconda3/envs/f1t/bin/python car_study_figs.py
~/miniconda3/envs/f1t/bin/python stop_analysis.py && ~/miniconda3/envs/f1t/bin/python car_traj_figs.py
```

Results notes stay with the code: `ilc_f1tenth/ilc2real/NOTES.md` (car), `~/ilc_ws/log/dilc/plane/NOTES.md` (quadruped).
