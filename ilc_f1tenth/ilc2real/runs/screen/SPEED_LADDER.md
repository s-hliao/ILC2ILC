# Rubber speed ladder (2026-11-05), written before any result

User: "it's fine to start with higher error because it leads to a more visible result, the whole point is i don't want it to be able
to track well initially post-domain shift"; "this can be done especially on rubber".

- Runs: the 36 existing rubber stage runs (SL50x120*: replicates 1-6 x network seeds 0-2 x PPO+DR / RMA; practised fig8R100 _x120,
  2 rubber cars), it 0 (zero-shot) vs it 3 (6 trial laps). No new training.
- Held-out track (user, 2026-11-05: "evaluate in sim on orca, and do peanut/fig8/squarefast on hardware"): ORCA (orca43, sim only),
  its uncapped plan (trajopt_drift as the rubber peanut's: no a_max, vstar 5.5) uniformly time-scaled (scale_plan.py) x1.00, x1.10,
  x1.20, x1.30, x1.40, x1.50. Corner grip of rubber vs the plans' friction: sqrt(1.0 / 0.6) ~ 1.29, so the ladder crosses the grip limit.
- Evaluation: ladder_eval.py = the car table's protocol (seed 701, 4 runs, sensing noise), 1 ORCA lap (77 m), F1T_DEV_TIME=1.
- Hardware is unchanged: peanut (held-out), fig8 (rubber practice), square2fast (plastic practice).
- Reported for EVERY rung, both networks: score change with bootstrap CI over runs, zero-shot and stage score, rms e_y, pace, time
  outside the 0.5 m tube, failures. The figure is the whole curve (stage gain vs initial error); no rung is picked after the fact.
- Added before any result (user: "if this works, may need to timescale the speeds for the on-hardware deployments"): the same ladder on
  the held-out PEANUT (the hardware track): mocap_peanutM_b0_free x1.10, x1.15, x1.20, x1.30, x1.40, x1.50, 5 laps. Choosing a hardware
  speed: the fastest rung whose stage change has CI upper < 0 AND whose stage run spends no more time outside the 0.5 m tube than the
  zero-shot (the room's envelope is a stop on hardware). Then a hardware check before flying: planned top speed / lateral acceleration
  against the car's limits, a desk rehearsal of the runbook at that speed. The user decides.

## Plastic ladder (user 2026-11-05: "what about plastic hardware speed ladder?"), written before any result
- Runs: the 18 plastic stage runs (PLSL*, practised square2fast _a45, 12 plastic cars), it 0 vs 3.
- Held-out plans: the hardware plan mocap_peanutM_b0_free_a33 (3.3 m/s^2 cap) time-scaled x1.05, x1.10, x1.15, x1.20 (scale_plan.py;
  lateral acceleration x k^2: plastic grip ~ 0.4 g = 3.9 m/s^2, so x1.10 already plans 4.0 m/s^2), plus x1.00 (the existing a33), 5 laps.
- Same evaluation, same reporting (every rung, both networks), same hardware rule as rubber: fastest rung with CI upper < 0 and no more
  time outside the tube than zero-shot; the user decides.
