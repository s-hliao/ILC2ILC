# Car RMA: the stage's feedback model vs the policy's true reaction (2026-11-06 ~02:00; logs/car_rma_impulse.py / .out)
hw_stage / ilc_core build the closed loop with K = d pi / d o, z_hat HELD FIXED (z_hat = phi(last 40 x [obs[:8], previous action])).
Measured in the car sim (real_car.run, noise-free, zero-shot rmaN_s0 / freePPON_s0, held-out peanut; one action impulse +-0.05 at
periods 40, 80, ...; channels steer and current): the share of the policy's true mean-action change the stage's model misses, by periods
after the impulse 1 / 2 / 4 / 8 / 16 (median):
  RMA rubber (ax:mu=1.667, 4 impulse sites x 2 channels)   0.38 / 0.17 / 0.14 / 0.45 / 0.84
  RMA plastic (n04_real_nom, 8 x 2)                        0.37 / 0.30 / 0.33 / 0.73 / 1.05
  PPO+DR, both (control: memoryless)                       0 at every lag
With z_hat's path (J_z dz, z_hat recomputed along each rollout) the error is 0.000: the miss is entirely the adaptation module's path.
The same class of error as the quad's (ARMB_VARIANTS_1102.md, --fb-memory). Not fixed: the fix (the chain through phi's 40-step
history) changes the car method and every RMA cell would be rerun; the user decides.
