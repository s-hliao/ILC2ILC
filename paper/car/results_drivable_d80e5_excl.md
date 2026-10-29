# Car results, one drivable plan per track (d80e5), validation seed 301

### 7 cars, training tracks (without ax:mu=0.6)

| arm | seeds | crash (%) | lateral err (cm) | pace (%) | success (%) |
|---|---|---|---|---|---|
| LQR on the plan | 1 | 2 | 7.9 | 102 | 80 |
| Ours (nominal sim), zero-shot | 3 | 2 | 8.3 | 102 | 56 |
| Ours (nominal sim) + 10 laps | 3 | 5 | 6.8 | 102 | 61 |
| Ours + DR, zero-shot | 3 | 2 | 8.1 | 102 | 66 |
| Ours + DR + 10 laps | 3 | 4 | 7.0 | 102 | 73 |
| PPO + DR, zero-shot | 3 | 6 | 7.7 | 104 | 80 |
| PPO + DR + our stage, 10 laps | 3 | 8 | 5.5 | 103 | 89 |
| PPO + DR (5 tracks), zero-shot | 3 | 4 | 8.8 | 104 | 70 |

### 7 cars, held-out track (squareH) (without ax:mu=0.6)

| arm | seeds | crash (%) | lateral err (cm) | pace (%) | success (%) |
|---|---|---|---|---|---|
| LQR on the plan | 1 | 4 | 13.3 | 101 | 39 |
| Ours (nominal sim), zero-shot | 3 | 10 | 13.1 | 101 | 39 |
| Ours (nominal sim) + 10 laps | 3 | 10 | 18.4 | 98 | 31 |
| Ours + DR, zero-shot | 3 | 8 | 10.8 | 101 | 38 |
| Ours + DR + 10 laps | 3 | 10 | 13.6 | 99 | 32 |
| PPO + DR, zero-shot | 3 | 11 | 4.9 | 109 | 89 |
| PPO + DR + our stage, 10 laps | 3 | 10 | 4.6 | 108 | 90 |
| PPO + DR (5 tracks), zero-shot | 3 | 21 | 2.7 | 110 | 79 |

### low-friction cars (x0.6, x0.7, real_mu), training tracks (without ax:mu=0.6)

| arm | seeds | crash (%) | lateral err (cm) | pace (%) | success (%) |
|---|---|---|---|---|---|
| LQR on the plan | 1 | 0 | 7.5 | 98 | 50 |
| Ours (nominal sim), zero-shot | 3 | 2 | 6.4 | 98 | 60 |
| Ours (nominal sim) + 10 laps | 3 | 10 | 4.6 | 98 | 73 |
| Ours + DR, zero-shot | 3 | 2 | 5.5 | 98 | 88 |
| Ours + DR + 10 laps | 3 | 6 | 4.3 | 98 | 92 |
| PPO + DR, zero-shot | 3 | 6 | 9.1 | 98 | 52 |
| PPO + DR + our stage, 10 laps | 3 | 10 | 6.2 | 97 | 81 |
| PPO + DR (5 tracks), zero-shot | 3 | 8 | 10.1 | 99 | 46 |

### 4 harsh cars, training tracks (without ax:mu=0.6)

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
