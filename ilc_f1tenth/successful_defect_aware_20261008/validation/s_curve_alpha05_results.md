# Defect-aware ILC: 10 updates

Reexecuted saved blend NMPC epoch-0 controls as fresh initial rollout; unchanged reference and effective costs; 35 Hz fixed-deadline execution. Nine-state blend dynamics plus previous-trial fixed defects, no simulator internal dynamics/hidden states. Formal acceptance uses defect-aware cost only; original nominal predictions are diagnostics. Actual measured performance is stored separately. Baseline code hashes unchanged.

| Epoch | Measured cost | Position RMSE (m) | Selected alpha |
|---|---:|---:|---:|
| 0 | 189.861198 | 0.346170 | - |
| 1 | 108.731189 | 0.291674 | 0.5 |
| 2 | 57.670526 | 0.163711 | 0.5 |
| 3 | 52.454827 | 0.147284 | 0.5 |
| 4 | 51.517912 | 0.143079 | 0.5 |
| 5 | 50.991117 | 0.139360 | 0.5 |
| 6 | 50.437544 | 0.138042 | 0.5 |
| 7 | 50.931720 | 0.140093 | 0.5 |
| 8 | 50.510983 | 0.137050 | 0.5 |
| 9 | 50.606295 | 0.137508 | 0.5 |
| 10 | 50.321677 | 0.134753 | 0.5 |

Best epoch: 10. All plots include reference, initial, best and final. Steering is the controller command state, not independently measured actuator angle.
