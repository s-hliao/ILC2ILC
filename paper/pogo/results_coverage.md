# Pogo by coverage tier (frozen test pass, seed 4301, servo leg, 3 seeds)

nerr = per run max(RMS apex error / 2 cm, |mean speed error| / 0.1 m/s), a fall = 5; mean over runs with a cluster-bootstrap 95 % CI over (seed, robot, goal). <= 1 is a success. Tiers: **A** inside the DR range (12 robots: each DR axis alone, both directions, at 0.75 of its half-range); **B** outside the range of a DR parameter (12: the same at 1.5 of the half-range); **C** outside the model class: prog + the 11 tier C robots that pass positivity (results_tierc.md: dead, bias, lag, ground, legm, dead2, dead3, lag2, lag3, legm2, legm3). Source: coverage_table.py.

| arm | real trials / robot | A nerr [95 % CI] | B nerr [95 % CI] | C nerr [95 % CI] | A success | B success | C success |
|---|---|---|---|---|---|---|---|
| Ours (nominal sim), zero-shot | 0 | 1.72 [1.62, 1.83] | 3.14 [2.86, 3.45] | 3.24 [3.10, 3.39] | 20 % | 7 % | 0 % |
| Ours + DR, zero-shot | 0 | 0.56 [0.52, 0.60] | 0.91 [0.84, 0.97] | 1.40 [1.29, 1.51] | 88 % | 68 % | 43 % |
| PPO + DR, zero-shot | 0 | 0.64 [0.60, 0.67] | 1.25 [1.19, 1.32] | 1.43 [1.31, 1.55] | 87 % | 32 % | 45 % |
| RMA (DR, history), zero-shot | 0 | 0.13 [0.12, 0.14] | 0.20 [0.18, 0.22] | 1.11 [0.98, 1.24] | 100 % | 100 % | 62 % |
| FADA (DR; published), zero-shot | 0 | 0.75 [0.71, 0.80] | 1.37 [1.30, 1.44] | 1.49 [1.37, 1.61] | 72 % | 30 % | 40 % |
| FADA (DR; published) + 24 | 24 | 0.65 [0.61, 0.69] | 1.19 [1.13, 1.26] | 1.37 [1.25, 1.50] | 84 % | 38 % | 50 % |
| Ours (nominal) + 12 trials | 12 | 0.45 [0.43, 0.47] | 0.99 [0.82, 1.22] | 0.88 [0.80, 0.97] | 100 % | 79 % | 67 % |
| Ours (nominal) + 24 trials | 24 | 0.42 [0.40, 0.44] | 0.74 [0.65, 0.84] | 0.54 [0.47, 0.62] | 100 % | 88 % | 94 % |
| Ours (nominal) + 36 trials | 36 | 0.42 [0.40, 0.44] | 0.89 [0.77, 1.02] | 0.50 [0.43, 0.58] | 100 % | 86 % | 96 % |
| Ours + DR + 12 trials | 12 | 0.48 [0.45, 0.51] | 0.65 [0.60, 0.69] | 0.99 [0.91, 1.07] | 95 % | 86 % | 64 % |
| Ours + DR + 24 trials | 24 | 0.49 [0.46, 0.52] | 0.58 [0.54, 0.63] | 0.88 [0.80, 0.96] | 95 % | 89 % | 70 % |
| Ours + DR + 36 trials | 36 | 0.49 [0.45, 0.52] | 0.59 [0.54, 0.64] | 0.82 [0.74, 0.90] | 94 % | 88 % | 74 % |
| PPO + DR + our stage, 12 trials | 12 | 0.30 [0.29, 0.32] | 0.55 [0.51, 0.60] | 0.55 [0.49, 0.60] | 99 % | 87 % | 87 % |
| PPO + DR + our stage, 24 trials | 24 | 0.30 [0.29, 0.32] | 0.48 [0.43, 0.53] | 0.39 [0.35, 0.43] | 99 % | 89 % | 95 % |
| PPO + DR + our stage, 36 trials | 36 | 0.32 [0.30, 0.34] | 0.48 [0.43, 0.53] | 0.37 [0.34, 0.41] | 99 % | 89 % | 95 % |
| PPO + DR + PPO on robot, 24 | 24 | 0.56 [0.53, 0.60] | 1.17 [1.11, 1.24] | 1.39 [1.26, 1.52] | 93 % | 36 % | 45 % |
| PPO + DR + PPO on robot, 36 | 36 | 0.53 [0.50, 0.57] | 1.12 [1.06, 1.19] | 1.36 [1.23, 1.49] | 94 % | 41 % | 48 % |
| PPO + DR + PPO on robot, 96 | 96 | 0.44 [0.41, 0.47] | 0.99 [0.93, 1.05] | 1.34 [1.21, 1.47] | 98 % | 54 % | 49 % |
| PPO + DR + PPO on robot, 384 | 384 | 0.20 [0.18, 0.21] | 0.53 [0.48, 0.58] | 1.14 [1.01, 1.27] | 100 % | 83 % | 55 % |
