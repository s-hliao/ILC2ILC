# Car results, one drivable plan per track (d60), validation seed 301

### 8 cars, training tracks

| arm | seeds | crash (%) | lateral err (cm) | pace (%) | success (%) |
|---|---|---|---|---|---|
| LQR on the plan | 1 | 62 | 7.3 | 104 | 22 |
| Ours (nominal sim), zero-shot | 3 | 68 | 8.3 | 102 | 23 |
| Ours (nominal sim) + 10 laps | 3 | 68 | 6.9 | 104 | 28 |
| Ours + DR, zero-shot | 3 | 65 | 10.0 | 102 | 19 |
| Ours + DR + 10 laps | 3 | 64 | 7.4 | 104 | 30 |
| PPO + DR, zero-shot | 3 | 14 | 7.0 | 116 | 76 |
| PPO + DR + our stage, 10 laps | 3 | 15 | 5.7 | 114 | 77 |
| PPO + DR (5 tracks), zero-shot | 3 | 22 | 7.0 | 116 | 72 |

### 8 cars, held-out track (squareH)

| arm | seeds | crash (%) | lateral err (cm) | pace (%) | success (%) |
|---|---|---|---|---|---|
| LQR on the plan | 1 | 62 | 7.9 | 103 | 25 |
| Ours (nominal sim), zero-shot | 3 | 62 | 7.6 | 103 | 24 |
| Ours (nominal sim) + 10 laps | 3 | 61 | 8.7 | 102 | 22 |
| Ours + DR, zero-shot | 3 | 56 | 10.7 | 103 | 23 |
| Ours + DR + 10 laps | 3 | 61 | 13.9 | 102 | 14 |
| PPO + DR, zero-shot | 3 | 2 | 5.8 | 117 | 98 |
| PPO + DR + our stage, 10 laps | 3 | 8 | 5.5 | 115 | 92 |
| PPO + DR (5 tracks), zero-shot | 3 | 10 | 5.1 | 121 | 90 |

### low-friction cars (x0.6, x0.7, real_mu), training tracks

| arm | seeds | crash (%) | lateral err (cm) | pace (%) | success (%) |
|---|---|---|---|---|---|
| LQR on the plan | 1 | 8 | 7.3 | 104 | 58 |
| Ours (nominal sim), zero-shot | 3 | 15 | 8.3 | 102 | 61 |
| Ours (nominal sim) + 10 laps | 3 | 15 | 6.9 | 104 | 74 |
| Ours + DR, zero-shot | 3 | 10 | 9.9 | 102 | 50 |
| Ours + DR + 10 laps | 3 | 8 | 7.3 | 104 | 81 |
| PPO + DR, zero-shot | 3 | 15 | 9.2 | 109 | 57 |
| PPO + DR + our stage, 10 laps | 3 | 15 | 6.8 | 108 | 62 |
| PPO + DR (5 tracks), zero-shot | 3 | 11 | 7.3 | 110 | 72 |

### 4 harsh cars, training tracks

| arm | seeds | crash (%) | lateral err (cm) | pace (%) | success (%) |
|---|---|---|---|---|---|
| LQR on the plan | 1 | 66 | 16.2 | 107 | 3 |
| Ours (nominal sim), zero-shot | 3 | 90 | 8.7 | 103 | 8 |
| Ours (nominal sim) + 10 laps | 3 | 88 | 8.9 | 104 | 9 |
| Ours + DR, zero-shot | 3 | 79 | 19.0 | 105 | 5 |
| Ours + DR + 10 laps | 3 | 76 | 13.6 | 108 | 9 |
| PPO + DR, zero-shot | 3 | 19 | 7.5 | 115 | 67 |
| PPO + DR + our stage, 10 laps | 3 | 25 | 6.2 | 113 | 67 |
| PPO + DR (5 tracks), zero-shot | 3 | 29 | 6.4 | 115 | 61 |
