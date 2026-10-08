# Defect-aware ILC: 10 updates

New figure-eight reference initialized using original blend NMPC function; fixed reference across all trials and effective costs; 35 Hz fixed-deadline execution. Nine-state blend dynamics plus previous-trial fixed defects, no simulator internal dynamics/hidden states. Formal acceptance uses defect-aware cost only; original nominal predictions are diagnostics. Actual measured performance is stored separately. Baseline code hashes unchanged.

| Epoch | Measured cost | Position RMSE (m) | Selected alpha |
|---|---:|---:|---:|
| 0 | 12571.209721 | 1.825472 | - |
| 1 | 2364.025015 | 0.784008 | 0.5 |
| 2 | 1747.529001 | 0.670373 | 0.5 |
| 3 | 1535.207762 | 0.623452 | 0.5 |
| 4 | 1185.141288 | 0.533492 | 0.5 |
| 5 | 1132.918586 | 0.514301 | 0.5 |
| 6 | 1041.924831 | 0.482897 | 0.5 |
| 7 | 1038.900950 | 0.482073 | 0.0 |
| 8 | 1057.436069 | 0.487356 | 0.0 |
| 9 | 1004.370527 | 0.471277 | 0.0 |
| 10 | 1019.224531 | 0.475753 | 0.0 |

Best epoch: 9. All plots include reference, initial, best and final. Steering is the controller command state, not independently measured actuator angle.

Initialization caveat: NMPC returned status2 (not converged). Figure-eight reference follows notebook cells8/9, scaled3, length36.15m, nominal2m/s with smooth start/stop, duration19.6s. Six updates accepted, epochs7-10 rejected and repeat epoch6 controls. Best measured epoch9 is a repeat-rollout observation, not a new learned control. Residual improvements on these repeats reflect variability.
