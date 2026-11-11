# Figure data (2026-11-10)

`quad_fig456_evals.tar.gz`: every per-jump evaluation file (`eval_planeval*.json`, 3400 files, 86 MB uncompressed) that the quadruped
figures 4, 5 and 6 are built from, with their paths relative to `~/ilc_ws/log/dilc/plane` (`fig4/eval/`, `fig5/`, `stage_fl/*_SRB8_*`,
`stage_fl/rt_srb_*`, `stage_fl/fadaNPq_s*`, `stage_fl/rt_fadaNPq_*`). To rebuild the figures on a fresh checkout:

    mkdir -p ~/ilc_ws/log/dilc/plane && tar xzf quad_fig456_evals.tar.gz -C ~/ilc_ws/log/dilc/plane
    cd ../budget2 && python3 quad_fig4s_data.py quad_fig4s.json && python3 budget_figs2.py quad_fig4s.json quad_fig4
    python3 quad_fig5s_data.py quad_fig5s.json && python3 fig5_best_iter.py quad_fig5s.json quad_fig5s_best.json && python3 budget_figs2.py quad_fig5s_best.json quad_fig5
    cd ../fig6 && python3 fig6_retarget_srb.py

Car figures 1 / 2: their sources are already in `budget2/fig2_src_prac/` and `budget2/fig2_src_x8/`. Pogo: `pogo_budget/budget_data.json`
holds every plotted point (the per-robot summaries stay in `~/ilc_ws/log/pogo_servo/runs`).
