# Stopping rule for the sim stage (car)

Zero-shot transfer to the five multi-body cars (beta 25 + 0 plans, 80 runs per snapshot, SE ~5 points) against SIM-ONLY validation metrics, per snapshot (every 5 iterations). Regret = best real transfer (3-point running mean) minus the picked snapshot's, in points.

## our learner + DR (B) (`v3drB_s0_long`)

Real transfer 65-80 % (beta 25: 40-70 %), best 50 iterations (running mean 79.6 %).

| sim metric (held-out plans) | Spearman with real |
|---|---|
| drwide/-rms_ey | +0.93 |
| dr/-rms_ey | +0.87 |
| dist/-rms_ey | +0.74 |
| nominal/-rms_ey | +0.68 |
| dr/success | +0.65 |
| dr+drwide/success | +0.60 |
| drwide/success | +0.58 |
| mean non-nominal/success | +0.57 |
| dr/-fail | -0.15 |
| delay/-rms_ey | -0.30 |
| pstart/-fail | -0.35 |
| pstart/success | -0.35 |
| drwide/-fail | -0.75 |
| pstart/-rms_ey | -0.77 |

| rule | picks (its) | real % | beta 25 % | regret (points) |
|---|---|---|---|---|
| last snapshot (90 its) | 90 | 71.2 | 52.5 | 6.7 |
| fixed 30 iterations (the pipeline) | 30 | 75.0 | 60.0 | 6.7 |
| argmax nominal/heldout/success | 0 | 65.0 | 40.0 | 14.6 |
| early stop nominal/heldout/success | 0 | 65.0 | 40.0 | 14.6 |
| early stop (patience 30 its) nominal/heldout/success | 0 | 65.0 | 40.0 | 14.6 |
| argmax nominal/heldout/-rms_ey | 75 | 78.8 | 67.5 | 2.5 |
| early stop nominal/heldout/-rms_ey | 0 | 65.0 | 40.0 | 14.6 |
| early stop (patience 30 its) nominal/heldout/-rms_ey | 0 | 65.0 | 40.0 | 14.6 |
| argmax pstart/heldout/success | 0 | 65.0 | 40.0 | 14.6 |
| early stop pstart/heldout/success | 0 | 65.0 | 40.0 | 14.6 |
| early stop (patience 30 its) pstart/heldout/success | 0 | 65.0 | 40.0 | 14.6 |
| argmax dr/heldout/success | 30 | 75.0 | 60.0 | 6.7 |
| early stop dr/heldout/success | 30 | 75.0 | 60.0 | 6.7 |
| early stop (patience 30 its) dr/heldout/success | 30 | 75.0 | 60.0 | 6.7 |
| argmax drwide/heldout/success | 40 | 76.2 | 62.5 | 3.3 |
| early stop drwide/heldout/success | 40 | 76.2 | 62.5 | 3.3 |
| early stop (patience 30 its) drwide/heldout/success | 40 | 76.2 | 62.5 | 3.3 |
| argmax dr+drwide/heldout/success | 40 | 76.2 | 62.5 | 3.3 |
| early stop dr+drwide/heldout/success | 40 | 76.2 | 62.5 | 3.3 |
| early stop (patience 30 its) dr+drwide/heldout/success | 40 | 76.2 | 62.5 | 3.3 |
| argmax mean non-nominal/heldout/success | 40 | 76.2 | 62.5 | 3.3 |
| early stop mean non-nominal/heldout/success | 30 | 75.0 | 60.0 | 6.7 |
| early stop (patience 30 its) mean non-nominal/heldout/success | 30 | 75.0 | 60.0 | 6.7 |
| argmax drwide/heldout/-rms_ey | 45 | 80.0 | 70.0 | 0.8 |
| early stop drwide/heldout/-rms_ey | 40 | 76.2 | 62.5 | 3.3 |
| early stop (patience 30 its) drwide/heldout/-rms_ey | 40 | 76.2 | 62.5 | 3.3 |

## ours (nominal sim) (`v3nom_s0_long`)

Real transfer 65-78 % (beta 25: 40-80 %), best 60 iterations (running mean 76.7 %).

| sim metric (held-out plans) | Spearman with real |
|---|---|
| mean non-nominal/success | +0.83 |
| pstart/-rms_ey | +0.81 |
| dr/success | +0.78 |
| dr+drwide/success | +0.76 |
| drwide/success | +0.73 |
| dr/-rms_ey | +0.69 |
| dist/-rms_ey | +0.66 |
| drwide/-rms_ey | +0.63 |
| nominal/-rms_ey | +0.37 |
| delay/-fail | -0.06 |
| delay/success | -0.06 |
| delay/-rms_ey | -0.29 |
| dr/-fail | -0.56 |
| drwide/-fail | -0.57 |
| pstart/-fail | -0.68 |
| pstart/success | -0.68 |

| rule | picks (its) | real % | beta 25 % | regret (points) |
|---|---|---|---|---|
| last snapshot (90 its) | 90 | 68.8 | 72.5 | 7.9 |
| fixed 30 iterations (the pipeline) | 30 | 67.5 | 45.0 | 7.9 |
| argmax nominal/heldout/success | 5 | 65.0 | 40.0 | 11.7 |
| early stop nominal/heldout/success | 5 | 65.0 | 40.0 | 11.7 |
| early stop (patience 30 its) nominal/heldout/success | 5 | 65.0 | 40.0 | 11.7 |
| argmax nominal/heldout/-rms_ey | 50 | 75.0 | 65.0 | 2.1 |
| early stop nominal/heldout/-rms_ey | 5 | 65.0 | 40.0 | 11.7 |
| early stop (patience 30 its) nominal/heldout/-rms_ey | 5 | 65.0 | 40.0 | 11.7 |
| argmax pstart/heldout/success | 5 | 65.0 | 40.0 | 11.7 |
| early stop pstart/heldout/success | 5 | 65.0 | 40.0 | 11.7 |
| early stop (patience 30 its) pstart/heldout/success | 5 | 65.0 | 40.0 | 11.7 |
| argmax dr/heldout/success | 80 | 72.5 | 80.0 | 4.2 |
| early stop dr/heldout/success | 5 | 65.0 | 40.0 | 11.7 |
| early stop (patience 30 its) dr/heldout/success | 80 | 72.5 | 80.0 | 4.2 |
| argmax drwide/heldout/success | 70 | 76.2 | 80.0 | 0.4 |
| early stop drwide/heldout/success | 5 | 65.0 | 40.0 | 11.7 |
| early stop (patience 30 its) drwide/heldout/success | 70 | 76.2 | 80.0 | 0.4 |
| argmax dr+drwide/heldout/success | 80 | 72.5 | 80.0 | 4.2 |
| early stop dr+drwide/heldout/success | 5 | 65.0 | 40.0 | 11.7 |
| early stop (patience 30 its) dr+drwide/heldout/success | 80 | 72.5 | 80.0 | 4.2 |
| argmax mean non-nominal/heldout/success | 70 | 76.2 | 80.0 | 0.4 |
| early stop mean non-nominal/heldout/success | 5 | 65.0 | 40.0 | 11.7 |
| early stop (patience 30 its) mean non-nominal/heldout/success | 70 | 76.2 | 80.0 | 0.4 |
| argmax drwide/heldout/-rms_ey | 90 | 68.8 | 72.5 | 7.9 |
| early stop drwide/heldout/-rms_ey | 5 | 65.0 | 40.0 | 11.7 |
| early stop (patience 30 its) drwide/heldout/-rms_ey | 5 | 65.0 | 40.0 | 11.7 |

