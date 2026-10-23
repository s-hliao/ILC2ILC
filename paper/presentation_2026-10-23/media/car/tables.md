### Car (F1TENTH 25-degree drift, 8 cars incl. friction x0.9/0.7/0.6) - b25 success %, validation seed 301, 3 seeds

| arm | real laps / car | b25 success % (mean +- sd) | crash % | all plans % | test 2701 b25 % |
|---|---|---|---|---|---|
| ours, zero-shot | 0 | 40.8 +- 4.0 | 45.8 | 49.9 | 40.1 |
| ours + DistILC ladder stage | 10 | 89.9 +- 14.2 | 1.9 | 94.9 | 91.1 |
| PPO+DR (task reward), zero-shot | 0 | 49.8 +- 9.3 | 28.1 | 55.3 | 48.6 |
| PPO+DR + our ladder stage | 10 | 85.9 +- 1.1 | 7.5 | 86.2 | 85.9 |
| PPO+DR on our rungs (no ILC) | 0 | 79.5 +- 10.0 | 8.3 | 87.0 | 80.0 |
| RMA (task), zero-shot | 0 | 39.1 +- 3.5 | 31.6 | 52.1 | 38.5 |
| RMA + our ladder stage | 10 | 75.2 +- 4.5 | 10.8 | 79.5 | - |

**Paired b25 differences (cluster = car x plan, 16 clusters, 95 % bootstrap) and direction ablations**

| comparison | delta b25 (pts) |
|---|---|
| ours + ladder - PPO+DR + ladder | +4.0 [-6.8, +16.7] |
| ours + ladder - RMA + ladder | +14.8 [-1.4, +32.5] |
| RMA + ladder - RMA zero-shot | +36.1 [+18.1, +54.9] |
| older setup (mu 0.2, seed 0, 24 laps): stage 79 % / open Jacobian 79 % / flipped Jacobian 51 % |  |

Sources: src/ilc_f1tenth/ilc2real/runs/hw/redo_val_table.txt, NOTES.md (success as scored by reeval.py: 12 runs x 5 chained laps). No random-direction arm; the flipped Jacobian is the control.
