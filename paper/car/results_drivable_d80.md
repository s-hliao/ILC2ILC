# Car results, one drivable plan per track (d80), validation seed 301

### 8 cars, training tracks

| arm | seeds | crash (%) | lateral err (cm) | pace (%) | success (%) |
|---|---|---|---|---|---|
| LQR on the plan | 1 | 16 | 7.9 | 102 | 70 |
| Ours (nominal sim), zero-shot | 3 | 15 | 8.3 | 102 | 48 |
| Ours (nominal sim) + 10 laps | 3 | 18 | 6.5 | 102 | 57 |
| Ours + DR, zero-shot | 3 | 15 | 8.2 | 102 | 57 |
| Ours + DR + 10 laps | 3 | 16 | 6.9 | 102 | 58 |
| PPO + DR, zero-shot | 3 | 14 | 7.8 | 104 | 70 |
| PPO + DR + our stage, 10 laps | 3 | 17 | 5.6 | 103 | 74 |
| PPO + DR (5 tracks), zero-shot | 3 | 12 | 8.9 | 104 | 61 |

### 8 cars, held-out track (squareH)

| arm | seeds | crash (%) | lateral err (cm) | pace (%) | success (%) |
|---|---|---|---|---|---|
| LQR on the plan | 1 | 16 | 13.3 | 101 | 34 |
| Ours (nominal sim), zero-shot | 3 | 25 | 13.1 | 101 | 34 |
| Ours (nominal sim) + 10 laps | 3 | 47 | 9.3 | 99 | 23 |
| Ours + DR, zero-shot | 3 | 16 | 10.9 | 101 | 33 |
| Ours + DR + 10 laps | 3 | 23 | 13.6 | 100 | 30 |
| PPO + DR, zero-shot | 3 | 15 | 4.9 | 108 | 84 |
| PPO + DR + our stage, 10 laps | 3 | 15 | 4.9 | 108 | 84 |
| PPO + DR (5 tracks), zero-shot | 3 | 30 | 2.7 | 110 | 69 |

### low-friction cars (x0.6, x0.7, real_mu), training tracks

| arm | seeds | crash (%) | lateral err (cm) | pace (%) | success (%) |
|---|---|---|---|---|---|
| LQR on the plan | 1 | 38 | 4.7 | 99 | 33 |
| Ours (nominal sim), zero-shot | 3 | 28 | 8.5 | 97 | 40 |
| Ours (nominal sim) + 10 laps | 3 | 29 | 7.4 | 96 | 51 |
| Ours + DR, zero-shot | 3 | 28 | 7.7 | 97 | 58 |
| Ours + DR + 10 laps | 3 | 31 | 6.5 | 96 | 58 |
| PPO + DR, zero-shot | 3 | 26 | 9.6 | 96 | 35 |
| PPO + DR + our stage, 10 laps | 3 | 32 | 7.4 | 96 | 46 |
| PPO + DR (5 tracks), zero-shot | 3 | 28 | 10.2 | 97 | 31 |

### 4 harsh cars, training tracks

| arm | seeds | crash (%) | lateral err (cm) | pace (%) | success (%) |
|---|---|---|---|---|---|
| LQR on the plan | 1 | 3 | 4.9 | 101 | 84 |
| Ours (nominal sim), zero-shot | 3 | 16 | 7.3 | 100 | 59 |
| Ours (nominal sim) + 10 laps | 3 | 18 | 7.1 | 101 | 59 |
| Ours + DR, zero-shot | 3 | 11 | 6.3 | 100 | 65 |
| Ours + DR + 10 laps | 3 | 9 | 6.5 | 101 | 73 |
| PPO + DR, zero-shot | 3 | 28 | 8.6 | 101 | 51 |
| PPO + DR + our stage, 10 laps | 3 | 26 | 6.3 | 100 | 65 |
| PPO + DR (5 tracks), zero-shot | 3 | 27 | 9.3 | 102 | 51 |
