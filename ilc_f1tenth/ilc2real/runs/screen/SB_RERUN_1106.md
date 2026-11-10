# SB rerun of every car stage cell (2026-11-06 ~17:35; user: "adopt SB"), written before any result

STEP_SCREEN_1106.md's rule picked SB (--cap 0.4: the trust region starts at its maximum; --fit-stall 0: the refit runs to
--fit-max-steps 8000 unless it realizes 90 % of the step). Every car stage cell of the paper is rerun with SB, otherwise exactly the
original protocol's flags, on the fixed simulator (ilc2real_fix: slip floor, F1T_SPIN_RHO=3), --iters 4 (0-8 real laps; the tables
report it 3 = 6 laps, the hardware protocol; figure 2 shows 0-8), --save-its 1, RMA with --z-path 1 (CAR_SIGNIF_1106.md C).
Folders runs/screen/SB_<protocol>_<net>_s<seed>_ss<stage seed> on ube (sb_stage.sh / sb_queue.sh). The S0 runs stay and are reported
as the step-size ablation.

| protocol | cars | practised plan (anchors) | trials | seeds (net x stage) | scored on |
|---|---|---|---|---|---|
| plastic | the 12 plastic | square2fast_a45 (square2fast / figfast _a45) | 2 laps | 0-2 x 9001-9003 | held-out peanut _a33 |
| rubber | the 2 rubber | fig8R100_x120 (square2fast / fig8R100 _x120) | 2 laps | 0-2 x 9001-9006 | held-out peanut _x120 |
| ethzp | the 12 plastic | ethzmobil43_a33 (as plastic) | 1 lap | 0-2 x 9001-9003 | ethzmobil43_a33 + held-out peanut _a33 |
| ethzr | the 2 rubber | ethzmobil43 (as rubber) | 1 lap | 0-2 x 9001-9003 | ethzmobil43 + held-out peanut _x120 |
| ethzr110 | the 2 rubber | ethzmobil43_x110 (as rubber) | 1 lap | 0-2 x 9001-9003 | ethzmobil43_x110 + held-out peanut _x120 |
| peax150 | the 2 rubber | peanutM_x150 (as rubber) | 2 laps | 0-2 x 9001 | peanutM_x150 |
| tier | the 36 tier cars (A / B / C) | square2fast_b0_free (square2fast / figfast _b0_free) | 2 laps | 0-2 x 9001-9002 | held-out peanut _b0_free |
Networks: PPO + DR (freePPON) and RMA (rmaN) for every protocol; figure 2's DistILC_sim / DistILC_sim + DR (figure 1's ours_s<s> /
oursdr_s<s>) on tier / plastic / rubber, seeds 0-2 x 9001. Scoring: ladder_eval.py (seed 701, 4 starts, 5 laps; 1 lap on ETHZMobil),
LADDER_ITS 0-4, statistic = car_final_score.py's (per run the mean over its cars of the % change vs it 0; bootstrap over runs).
The hardware runbook's car stage gets --cap 0.4 --fit-stall 0 (hw_stage.py's --fit-stall synced from ilc2real_fix).
