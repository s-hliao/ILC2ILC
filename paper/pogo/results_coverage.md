# Pogo by coverage tier (frozen test pass, seed 4301, servo leg, 3 seeds)

nerr = per run max(RMS apex error / 2 cm, |mean speed error| / 0.1 m/s), a fall = 5; mean over runs with a cluster-bootstrap 95 % CI over (seed, robot, goal). <= 1 is a success. Tiers: **A** inside the DR range (nom, mass, spring); **B** outside the range of a DR parameter (act, offset); **C** outside the model class: prog + the 11 tier C robots that pass positivity (results_tierc.md: dead, bias, lag, ground, legm, dead2, dead3, lag2, lag3, legm2, legm3). Source: coverage_table.py.

| arm | real trials / robot | A nerr [95 % CI] | B nerr [95 % CI] | C nerr [95 % CI] | A success | B success | C success |
|---|---|---|---|---|---|---|---|
| Ours (nominal sim), zero-shot | 0 | 2.37 [2.15, 2.60] | 2.70 [2.48, 2.97] | 3.24 [3.09, 3.38] | 3 % | 0 % | 0 % |
| Ours + DR, zero-shot | 0 | 0.45 [0.38, 0.52] | 1.47 [1.20, 1.76] | 1.40 [1.30, 1.51] | 93 % | 43 % | 43 % |
| PPO + DR, zero-shot | 0 | 0.70 [0.60, 0.78] | 1.12 [0.93, 1.32] | 1.43 [1.30, 1.55] | 79 % | 50 % | 45 % |
| RMA (DR, history), zero-shot | 0 | 0.10 [0.09, 0.12] | 0.13 [0.11, 0.14] | 1.11 [0.99, 1.24] | 100 % | 100 % | 62 % |
| FADA (DR; published), zero-shot | 0 | 0.88 [0.79, 0.97] | 1.14 [0.98, 1.29] | 1.49 [1.37, 1.62] | 57 % | 49 % | 40 % |
| FADA (DR; published) + 24 | 24 | 0.70 [0.62, 0.79] | 1.07 [0.90, 1.25] | 1.37 [1.25, 1.50] | 84 % | 52 % | 50 % |
| Ours (nominal) + 12 trials | 12 | 0.60 [0.53, 0.68] | 0.80 [0.50, 1.26] | 0.88 [0.80, 0.97] | 97 % | 92 % | 67 % |
| Ours (nominal) + 24 trials | 24 | 0.49 [0.44, 0.55] | 0.72 [0.40, 1.22] | 0.54 [0.48, 0.63] | 99 % | 93 % | 94 % |
| Ours (nominal) + 36 trials | 36 | 0.47 [0.42, 0.52] | 0.51 [0.36, 0.76] | 0.50 [0.42, 0.58] | 99 % | 94 % | 96 % |
| Ours + DR + 12 trials | 12 | 0.51 [0.44, 0.59] | 0.95 [0.81, 1.10] | 0.99 [0.91, 1.08] | 90 % | 49 % | 64 % |
| Ours + DR + 24 trials | 24 | 0.61 [0.53, 0.69] | 0.63 [0.56, 0.71] | 0.88 [0.80, 0.96] | 83 % | 90 % | 70 % |
| Ours + DR + 36 trials | 36 | 0.69 [0.59, 0.79] | 0.54 [0.48, 0.61] | 0.82 [0.74, 0.90] | 79 % | 94 % | 74 % |
| PPO + DR + our stage, 12 trials | 12 | 0.26 [0.24, 0.28] | 0.35 [0.31, 0.39] | 0.55 [0.50, 0.60] | 100 % | 100 % | 87 % |
| PPO + DR + our stage, 24 trials | 24 | 0.28 [0.25, 0.30] | 0.32 [0.28, 0.36] | 0.39 [0.36, 0.43] | 100 % | 100 % | 95 % |
| PPO + DR + our stage, 36 trials | 36 | 0.31 [0.28, 0.33] | 0.31 [0.28, 0.35] | 0.37 [0.34, 0.41] | 100 % | 100 % | 95 % |
| PPO + DR + PPO on robot, 24 | 24 | 0.68 [0.60, 0.76] | 0.99 [0.83, 1.17] | 1.39 [1.26, 1.51] | 81 % | 50 % | 45 % |
| PPO + DR + PPO on robot, 36 | 36 | 0.64 [0.58, 0.71] | 0.93 [0.77, 1.08] | 1.36 [1.24, 1.48] | 90 % | 52 % | 48 % |
| PPO + DR + PPO on robot, 96 | 96 | 0.56 [0.50, 0.63] | 0.77 [0.66, 0.88] | 1.34 [1.21, 1.47] | 95 % | 67 % | 49 % |
| PPO + DR + PPO on robot, 384 | 384 | 0.22 [0.20, 0.25] | 0.19 [0.17, 0.22] | 1.14 [1.01, 1.28] | 100 % | 100 % | 55 % |
