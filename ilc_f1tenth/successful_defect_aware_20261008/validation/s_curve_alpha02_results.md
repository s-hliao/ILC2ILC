# Defect-aware ILC: 20 updates

Reexecuted saved blend NMPC epoch-0 controls as fresh initial rollout; unchanged reference and effective costs; 35 Hz fixed-deadline execution. Nine-state blend dynamics plus previous-trial fixed defects, no simulator internal dynamics/hidden states. Formal acceptance uses defect-aware cost only; original nominal predictions are diagnostics. Actual measured performance is stored separately. Baseline code hashes unchanged.

| Epoch | Measured cost | Position RMSE (m) | Selected alpha |
|---|---:|---:|---:|
| 0 | 187.989663 | 0.341460 | - |
| 1 | 151.439992 | 0.323390 | 0.2 |
| 2 | 128.919598 | 0.311609 | 0.2 |
| 3 | 110.969053 | 0.294216 | 0.2 |
| 4 | 71.198528 | 0.198692 | 0.2 |
| 5 | 63.366655 | 0.181978 | 0.2 |
| 6 | 57.025908 | 0.162578 | 0.2 |
| 7 | 55.275431 | 0.158629 | 0.2 |
| 8 | 52.955267 | 0.151464 | 0.2 |
| 9 | 52.415088 | 0.149505 | 0.2 |
| 10 | 51.211622 | 0.143993 | 0.2 |
| 11 | 51.427919 | 0.144732 | 0.2 |
| 12 | 51.380324 | 0.144599 | 0.2 |
| 13 | 50.629598 | 0.140691 | 0.2 |
| 14 | 51.013412 | 0.141815 | 0.2 |
| 15 | 50.850044 | 0.140346 | 0.2 |
| 16 | 50.340457 | 0.139098 | 0.2 |
| 17 | 50.191527 | 0.138174 | 0.2 |
| 18 | 50.182072 | 0.137740 | 0.2 |
| 19 | 50.112738 | 0.137098 | 0.2 |
| 20 | 50.302825 | 0.137776 | 0.2 |

Best epoch: 19. All plots include reference, initial, best and final. Steering is the controller command state, not independently measured actuator angle.
