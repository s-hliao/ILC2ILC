# ORCA study (2026-11-05), written before any result

User: "i need more extreme example, can you run training on harder track like ethzurich orca, also don't stop the trial at
0.5 m deviation, just mark it as deviation time."

- Track: the ETH Zurich ORCA centreline (github.com/alexliniger/MPCC, C++/Params/track.json) scaled x4.3 (1:43 -> 1:10):
  `orca43`, 76.8 m per lap, tightest radius 0.80 m, 1610 deg of turning per lap. SIMULATION ONLY (does not fit the room).
- Plan: `orca43_b0_free_a33`, trajopt_drift.py with the held-out peanut's settings (beta 0, w_beta 0, a_max 3.3, --vstar 5.5 as every plastic plan, nominal mu 0.6).
- Stage: orca_stage.sh = the plastic stage's exact arguments (PLSL_*/args.json, checked flag by flag), practising orca43;
  12 plastic cars; PPO+DR and RMA, network seeds 0-2, stage seed 9001 first (6 runs); more replicates if the first is informative.
- F1T_DEV_TIME=1 in stage and evaluation: leaving the 0.5 m envelope is timed, not stopped; spin-outs and stalls still end a run.
  On the real car the envelope stays a stop.
- Evaluation: orca_eval.py, the car table's protocol (seed 701, 4 runs, noise), orca43 1 lap + held-out peanut 5 laps, it 0 vs 3.
  Reported for every cell: score change, rms e_y, pace, deviation time, failures. Whether the gain is pace or tracking is reported.
- Any figure "example" is the largest drop and is labelled as selected, with the median beside it.
