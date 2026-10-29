# Pogo tier C: outside the model class (servo leg, test seed 4301, 3 seeds)

Inclusion (fixed before the run): positivity holds, median cos >= 0.5 on our stages' own trials (ours nominal + ours + DR, pooled). nerr = max(RMS apex / 2 cm, |dv| / 0.1 m/s), a fall = 5; <= 1 is a success.

| robot | positivity (median cos, ours nominal + DR) | trial pairs | tier C |
|---|---|---|---|
| dead | 0.89 | 180 | yes |
| bias | 0.80 | 162 | yes |
| lag | 0.85 | 180 | yes |
| ground | 0.79 | 180 | yes |
| legm | 0.87 | 180 | yes |
| dead2 | 0.83 | 180 | yes |
| dead3 | 0.78 | 174 | yes |
| lag2 | 0.82 | 102 | yes |
| lag3 | 0.81 | 96 | yes |
| ground2 | 0.23 | 150 | no |
| ground3 | 0.27 | 180 | no |
| legm2 | 0.78 | 180 | yes |
| legm3 | 0.74 | 180 | yes |

### Tier C (positivity holds)

| arm | real trials / robot | nerr [95 % CI] | success | falls | dead / bias / lag / ground / legm / dead2 / dead3 / lag2 / lag3 / legm2 / legm3 |
|---|---|---|---|---|---|
| Ours (nominal sim), zero-shot | 0 | 3.41 [3.26, 3.54] | 0 % | 6.6 % | 2.73 / 2.29 / 2.16 / 3.41 / 2.87 / 4.04 / 5.35 / 2.78 / 3.38 / 3.86 / 4.64 |
| Ours + DR, zero-shot | 0 | 1.46 [1.34, 1.57] | 39 % | 0.0 % | 0.73 / 1.88 / 0.78 / 0.47 / 1.45 / 0.95 / 1.13 / 1.26 / 1.70 / 2.45 / 3.23 |
| PPO + DR, zero-shot | 0 | 1.52 [1.39, 1.65] | 40 % | 0.0 % | 0.69 / 1.60 / 0.42 / 1.00 / 1.62 / 1.37 / 2.22 / 0.56 / 0.90 / 2.74 / 3.60 |
| RMA (DR, history), zero-shot | 0 | 1.19 [1.05, 1.34] | 59 % | 0.7 % | 0.27 / 2.11 / 0.90 / 0.11 / 1.12 / 0.33 / 0.82 / 0.57 / 0.37 / 2.63 / 3.88 |
| FADA (DR; published), zero-shot | 0 | 1.60 [1.47, 1.72] | 35 % | 0.0 % | 0.78 / 1.47 / 0.53 / 1.18 / 1.69 / 1.51 / 2.38 / 0.65 / 1.01 / 2.77 / 3.59 |
| FADA (DR; published) + 24 | 24 | 1.46 [1.33, 1.60] | 45 % | 0.8 % | 0.61 / 2.03 / 0.42 / 0.83 / 1.40 / 1.26 / 2.00 / 0.48 / 0.70 / 2.61 / 3.76 |
| Ours (nominal) + 12 trials | 12 | 0.94 [0.86, 1.03] | 64 % | 1.1 % | 0.55 / 0.35 / 0.61 / 0.49 / 0.46 / 0.63 / 1.79 / 1.31 / 2.02 / 0.67 / 1.43 |
| Ours (nominal) + 36 trials | 36 | 0.52 [0.44, 0.61] | 96 % | 2.3 % | 0.42 / 0.40 / 0.42 / 0.45 / 0.26 / 0.45 / 0.87 / 0.86 / 0.96 / 0.33 / 0.30 |
| Ours + DR + 12 trials | 12 | 1.04 [0.95, 1.13] | 60 % | 0.0 % | 0.49 / 1.43 / 0.48 / 0.57 / 0.54 / 0.61 / 0.68 / 1.34 / 1.96 / 1.27 / 2.06 |
| Ours + DR + 36 trials | 36 | 0.87 [0.79, 0.95] | 71 % | 0.0 % | 0.47 / 0.92 / 0.63 / 0.91 / 0.33 / 0.52 / 0.88 / 1.28 / 1.91 / 0.57 / 1.11 |
| PPO + DR + our stage, 12 trials | 12 | 0.58 [0.52, 0.63] | 86 % | 0.0 % | 0.37 / 0.42 / 0.27 / 0.33 / 0.21 / 0.56 / 0.87 / 0.41 / 0.97 / 0.42 / 1.52 |
| PPO + DR + our stage, 36 trials | 36 | 0.39 [0.35, 0.42] | 94 % | 0.0 % | 0.32 / 0.33 / 0.26 / 0.38 / 0.17 / 0.35 / 0.49 / 0.43 / 0.97 / 0.16 / 0.37 |

### Excluded robots (positivity fails; appendix)

| arm | real trials / robot | nerr [95 % CI] | success | falls | ground2 / ground3 |
|---|---|---|---|---|---|
| Ours (nominal sim), zero-shot | 0 | 4.97 [4.93, 5.00] | 0 % | 96.9 % | 4.94 / 5.00 |
| Ours + DR, zero-shot | 0 | 2.81 [2.32, 3.33] | 19 % | 5.7 % | 1.62 / 4.00 |
| PPO + DR, zero-shot | 0 | 3.58 [3.13, 4.02] | 0 % | 3.1 % | 2.26 / 4.90 |
| RMA (DR, history), zero-shot | 0 | 1.47 [1.10, 1.85] | 52 % | 1.0 % | 0.39 / 2.54 |
| FADA (DR; published), zero-shot | 0 | 3.70 [3.20, 4.23] | 0 % | 5.7 % | 2.35 / 5.04 |
| FADA (DR; published) + 24 | 24 | 2.76 [2.42, 3.12] | 0 % | 4.2 % | 1.82 / 3.71 |
| Ours (nominal) + 12 trials | 12 | 4.71 [4.46, 4.91] | 0 % | 91.7 % | 4.43 / 5.00 |
| Ours (nominal) + 36 trials | 36 | 4.81 [4.55, 5.06] | 2 % | 89.1 % | 4.63 / 4.99 |
| Ours + DR + 12 trials | 12 | 2.38 [1.89, 2.93] | 28 % | 6.2 % | 1.03 / 3.73 |
| Ours + DR + 36 trials | 36 | 2.38 [1.91, 2.87] | 22 % | 6.8 % | 1.14 / 3.61 |
| PPO + DR + our stage, 12 trials | 12 | 2.10 [1.76, 2.43] | 22 % | 2.1 % | 1.10 / 3.10 |
| PPO + DR + our stage, 36 trials | 36 | 1.81 [1.54, 2.09] | 19 % | 2.6 % | 1.15 / 2.47 |

