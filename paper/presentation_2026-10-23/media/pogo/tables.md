### Pogo (2-D SLIP hopper, 6 robots incl. 3 outside the DR family) - success %, frozen test seed 3301, 3 seeds

| arm | real trials / robot | success % [95 % CI] | apex err (cm) | |dv| (m/s) |
|---|---|---|---|---|
| ours, zero-shot | 0 | 7.6 [1.4, 15.3] | 3.81 | 0.102 |
| ours + DistILC stage | 36 | 90.8 [84.2, 96.5] | 0.81 | 0.027 |
| ours+DR, zero-shot | 0 | 48.8 [37.7, 59.7] | 1.61 | 0.090 |
| ours+DR + DistILC stage | 36 | 97.6 [94.8, 99.8] | 0.47 | 0.023 |
| PPO+DR, zero-shot | 0 | 73.6 [62.2, 84.0] | 0.71 | 0.068 |
| PPO+DR + DistILC stage | 36 | 97.2 [94.6, 99.3] | 0.38 | 0.023 |
| PPO fine-tuned on the robot | 48 / 384 | 79.5 / 94.4 | 0.60 / 0.38 | 0.059 / 0.034 |
| FADA (own adaptation) | 24 | 95.1 [91.7, 97.9] | 0.79 | 0.030 |
| RMA, zero-shot (privileged) | 0 | 98.6 [96.5, 100] | 0.34 | 0.020 |

**Ablations of the stage's update direction (validation, nominal start s0, 24 trials)**

| arm | success % |
|---|---|
| stage, Gauss-Newton step through the sim Jacobian | 79.7 |
| open-loop Jacobian (no feedback term) | 5.2 |
| flipped Jacobian (the 180-degree direction control) | 0.0 |
| + trust region / no rollback | 77.1 / 79.7 |
| sim-stage final snapshot instead of the stopping-rule pick | 34.4 |

Sources: src/ilc_pogo/results_table_test3301.txt (success as scored by common.hop_metrics / ci.py), src/ilc_pogo/NOTES.md (controls). No random-direction arm in this domain; the flipped Jacobian is the direction control.
