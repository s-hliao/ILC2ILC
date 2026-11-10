# Car refit-parameterization screen (2026-11-06 ~18:30; Atkeson meeting: "the fine-tuning algorithm was chosen heuristically,
# have we tried ablating it" -> user: "that's fine"), written before any result

## Question
How sensitive is the car stage to the refit's knobs (the anchored supervised fit onto the GN-step targets)? Every knob below
was set once and never swept on the car. Last-layer-only / other parameterizations are excluded (user: "not the point").

## Prior sweeps (found 18:40, before launch)
- Car anchor weight 1 / 5 / 20 (+ 5 on all 15 bank plans), car_anchor.sh 2026-10-30, ours + DR and PPO + DR, seeds 0-2, the
  old simulator and _v08 plans: weight 1 drifted the other goal (square2fast 0.3-0.5 cm at 20 vs 2.8 cm at 1) -> 20 adopted.
  RMA rubber 2026-11-01: anchor weight 2 / 5 / 10 not adopted. So AW5 is dropped; AW0 / AW80 extend the range on today's setup.
- Pogo: refit 800 vs 3000 steps, lr 1e-4 vs 3e-4 (success unchanged); age decay 0 + refit to convergence (-8 to -12 pts).
- Car age decay and refit steps / lr: never swept.

## Arms (each = SB exactly, ONE knob changed; sb_stage.sh's flags, the extra flag appended so argparse's last value wins)
| arm | change vs SB | what it tests |
|---|---|---|
| AW0  | --anchor-w 0     | no anchor to the network's sim behaviour on the anchor plans |
| AW80 | --anchor-w 80    | stronger anchor |
| AD0  | --age-decay 0    | refit onto the newest trials only (SB: 0.5 = weights halve per iteration of age) |
| AD25 | --age-decay 0.25 | faster forgetting |
| AD1  | --age-decay 1    | every trial weighted equally, however old |
| FIT  | --steps 3000 --lr 3e-4 | a stronger refit (SB: 800 Adam steps, lr 1e-4) |

## Cell
Plastic held-out protocol (sb_stage.sh plastic): PPO + DR (freePPON), network seeds 0-2 x stage seeds 9001-9002 = 6 runs per arm,
36 runs, folders runs/screen/RF_<arm>_plastic_freePPON_s<s>_ss<ss> on ube (ilc2real_fix, the fixed simulator), --iters 4.
Baseline = the SB runs SB_plastic_freePPON_s<s>_ss<ss> with the same seeds (paired).

## Evaluation (as the main tables)
ladder_eval.py, seed 701, 4 starts, 5 laps, held-out peanut _a33, the 12 plastic cars, LADDER_ITS 0-4, F1T_SPIN_RHO=3.
Statistic: per run the mean over its cars of the % score change it 3 vs it 0 (car_final_score.py's); paired difference arm - SB
over the 6 seed pairs, bootstrap CI. Also reported: it 1 / 2 / 4, spin-out share at it 3.

## Rule
Every arm is reported (the refit-sensitivity ablation). An arm is called better than SB only if its paired difference has the
upper end of the Bonferroni-corrected CI (6 arms: 99.2 %) below 0 AND its spin-out share at it 3 is not higher than SB's.
Changing the method's refit settings would be a method change: asked to the user before anything is rerun; otherwise SB stays.
Scheduling: the screen starts only after the SB queue has launched all 146 of its runs, at most 6 stage runs at once in total.
