# Car results, one drivable plan per track (d80e5), validation seed 301

### 8 cars, training tracks

| arm | seeds | crash (%) | lateral err (cm) | pace (%) | success (%) |
|---|---|---|---|---|---|
| LQR on the plan | 1 | 3 | 8.1 | 101 | 70 |
| Ours (nominal sim), zero-shot | 3 | 8 | 8.6 | 102 | 49 |
| Ours (nominal sim) + 10 laps | 3 | 10 | 7.4 | 102 | 54 |
| Ours + DR, zero-shot | 3 | 8 | 8.3 | 102 | 58 |
| Ours + DR + 10 laps | 3 | 8 | 7.1 | 102 | 64 |
| PPO + DR, zero-shot | 3 | 11 | 7.8 | 104 | 70 |
| PPO + DR + our stage, 10 laps | 3 | 14 | 5.6 | 103 | 78 |
| PPO + DR (5 tracks), zero-shot | 3 | 10 | 8.9 | 104 | 61 |

### 8 cars, held-out track (squareH)

| arm | seeds | crash (%) | lateral err (cm) | pace (%) | success (%) |
|---|---|---|---|---|---|
| LQR on the plan | 1 | 16 | 13.3 | 101 | 34 |
| Ours (nominal sim), zero-shot | 3 | 21 | 13.1 | 101 | 34 |
| Ours (nominal sim) + 10 laps | 3 | 21 | 18.4 | 98 | 27 |
| Ours + DR, zero-shot | 3 | 16 | 10.9 | 101 | 33 |
| Ours + DR + 10 laps | 3 | 17 | 13.7 | 99 | 28 |
| PPO + DR, zero-shot | 3 | 15 | 4.9 | 108 | 84 |
| PPO + DR + our stage, 10 laps | 3 | 14 | 4.6 | 108 | 85 |
| PPO + DR (5 tracks), zero-shot | 3 | 30 | 2.7 | 110 | 69 |

### low-friction cars (x0.6, x0.7, real_mu), training tracks

| arm | seeds | crash (%) | lateral err (cm) | pace (%) | success (%) |
|---|---|---|---|---|---|
| LQR on the plan | 1 | 4 | 12.7 | 97 | 33 |
| Ours (nominal sim), zero-shot | 3 | 18 | 9.9 | 96 | 40 |
| Ours (nominal sim) + 10 laps | 3 | 24 | 8.1 | 96 | 49 |
| Ours + DR, zero-shot | 3 | 18 | 8.5 | 96 | 58 |
| Ours + DR + 10 laps | 3 | 17 | 7.4 | 96 | 61 |
| PPO + DR, zero-shot | 3 | 21 | 9.7 | 96 | 35 |
| PPO + DR + our stage, 10 laps | 3 | 24 | 8.0 | 95 | 54 |
| PPO + DR (5 tracks), zero-shot | 3 | 22 | 10.3 | 97 | 31 |

### 4 harsh cars, training tracks

| arm | seeds | crash (%) | lateral err (cm) | pace (%) | success (%) |
|---|---|---|---|---|---|
| LQR on the plan | 1 | 3 | 4.9 | 101 | 84 |
| Ours (nominal sim), zero-shot | 3 | 12 | 7.4 | 100 | 59 |
| Ours (nominal sim) + 10 laps | 3 | 15 | 7.7 | 101 | 61 |
| Ours + DR, zero-shot | 3 | 6 | 6.4 | 100 | 65 |
| Ours + DR + 10 laps | 3 | 2 | 6.5 | 100 | 71 |
| PPO + DR, zero-shot | 3 | 20 | 8.8 | 101 | 51 |
| PPO + DR + our stage, 10 laps | 3 | 19 | 6.5 | 100 | 71 |
| PPO + DR (5 tracks), zero-shot | 3 | 20 | 9.4 | 103 | 53 |
