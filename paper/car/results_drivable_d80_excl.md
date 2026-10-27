# Car results, one drivable plan per track (d80), validation seed 301

### 8 cars, training tracks (without ax:mu=0.6)

| arm | seeds | crash (%) | lateral err (cm) | pace (%) | success (%) |
|---|---|---|---|---|---|
| LQR on the plan | 1 | 4 | 7.9 | 102 | 80 |
| Ours (nominal sim), zero-shot | 3 | 7 | 8.0 | 102 | 55 |
| Ours (nominal sim) + 10 laps | 3 | 10 | 6.4 | 102 | 65 |
| Ours + DR, zero-shot | 3 | 7 | 8.0 | 102 | 65 |
| Ours + DR + 10 laps | 3 | 8 | 6.8 | 102 | 66 |
| PPO + DR, zero-shot | 3 | 6 | 7.7 | 104 | 80 |
| PPO + DR + our stage, 10 laps | 3 | 10 | 5.6 | 103 | 85 |
| PPO + DR (5 tracks), zero-shot | 3 | 4 | 8.8 | 104 | 70 |

### 8 cars, held-out track (squareH) (without ax:mu=0.6)

| arm | seeds | crash (%) | lateral err (cm) | pace (%) | success (%) |
|---|---|---|---|---|---|
| LQR on the plan | 1 | 4 | 13.3 | 101 | 39 |
| Ours (nominal sim), zero-shot | 3 | 14 | 13.1 | 101 | 39 |
| Ours (nominal sim) + 10 laps | 3 | 39 | 9.3 | 99 | 26 |
| Ours + DR, zero-shot | 3 | 8 | 10.8 | 101 | 38 |
| Ours + DR + 10 laps | 3 | 17 | 13.3 | 100 | 35 |
| PPO + DR, zero-shot | 3 | 11 | 4.9 | 109 | 89 |
| PPO + DR + our stage, 10 laps | 3 | 11 | 4.9 | 109 | 89 |
| PPO + DR (5 tracks), zero-shot | 3 | 21 | 2.7 | 110 | 79 |

### low-friction cars (x0.6, x0.7, real_mu), training tracks (without ax:mu=0.6)

| arm | seeds | crash (%) | lateral err (cm) | pace (%) | success (%) |
|---|---|---|---|---|---|
| LQR on the plan | 1 | 6 | 4.7 | 99 | 50 |
| Ours (nominal sim), zero-shot | 3 | 6 | 5.3 | 99 | 60 |
| Ours (nominal sim) + 10 laps | 3 | 8 | 3.4 | 99 | 77 |
| Ours + DR, zero-shot | 3 | 6 | 4.1 | 99 | 88 |
| Ours + DR + 10 laps | 3 | 10 | 4.1 | 98 | 88 |
| PPO + DR, zero-shot | 3 | 6 | 9.1 | 98 | 52 |
| PPO + DR + our stage, 10 laps | 3 | 15 | 6.3 | 97 | 69 |
| PPO + DR (5 tracks), zero-shot | 3 | 8 | 10.1 | 99 | 46 |

### 4 harsh cars, training tracks (without ax:mu=0.6)

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
