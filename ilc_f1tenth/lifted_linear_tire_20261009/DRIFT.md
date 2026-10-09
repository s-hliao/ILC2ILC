# Using the ILC for continuous drift: design notes (untested)

## Why the current ILC cannot drift as it is

1. **Drift equilibria are open-loop unstable.** A steady drift has saturated rear tires and countersteer, and it is a saddle equilibrium (Voser, Hindiyeh and Gerdes; Goh and Gerdes). The current scheme flies the learned controls open-loop for the whole trial, so any deviation grows within about 1 s and no two trials are alike. The ILC has to sit around a stabilizing feedback.
2. **The linear tire has no drift.** With F_y = C·α there is no saturation, so there is no drift equilibrium, and past the peak the sensitivities have the wrong sign. The model behind G needs saturating tires, and at the rear it needs combined slip (drive force using up lateral capacity). The Fiala model in `dynamics.py` has both.
3. **Continuous running means no resets.** Each lap starts wherever the previous one ended, and laps are not time-synchronized. A time-indexed error turns small phase drift into large position errors, which is already the dominant leftover term on the S-curve (along-track error 0.125 m against 0.044 m cross-track).
4. **Sideslip isn't measured.** The odometry contract has vy = 0. Drift needs β, and on hardware that has to come from mocap (`optitrack_node.py`). In sim, the MB bridge would need to publish vy.
5. **Speed-command actuation can't produce power-over.** The gym front end turns a speed command into body acceleration. Drifting needs rear wheel spin, so the VESC needs to be in current (torque) mode, which is the 9-state model's `current` state used directly. On hardware this means a current command instead of `speed`.
6. **The MB tire is too soft at the limit.** With the Froude-scaled set, the lateral force peaks at about 35° of slip, while real RC tires peak at about 8–12°. A drift study needs MF coefficients fitted to the RC tire, ideally from mocap drift logs.

## Proposed structure (quadruped analog in brackets)

1. **TO [full-body TO]:** Fiala-model trajectory optimization (`trajopt.py`) for an entry into drift followed by one steady drift lap.
   - It is periodic in path coordinates: the end state equals the start state up to the lap angle.
   - It is constrained to the friction circle and steering limits.
   - It is initialized from the analytic drift equilibrium for the target radius and sideslip (solve f(x, u) = 0 with β fixed) [the SRB guess].
2. **Stabilizer:** time-varying LQR around the TO plan, from the Fiala linearization, running at 40 Hz on mocap state. Trials only become repeatable once this loop is closed.
3. **ILC on the closed loop [Stages I–III]:** learn the feedforward u_ff that the closed loop adds to the LQR.
   - Closed-loop lifted G from x⁺ = (A − BK)x + B·u_ff, with A and B from Fiala along the measured lap.
   - Error indexed by path position (arc length, or polar angle on a circle) rather than time: lateral offset, sideslip error, speed error. That removes the phase-drift term.
   - Repetitive (lap-to-lap) update: the QP is solved while the next lap runs and applied one lap later. At about 200 samples per lap the box QP takes well under a second.
4. **Sensitivity correction [secant]:** Broyden updates on G from each lap's measured response, because Fiala sensitivities near saturation will be off, as the SRB pitch sensitivity was.
5. **Safeguard [Stage III rollback]:** abort a lap on spin-out (β or yaw rate beyond limits) and hand over to a recovery controller. Score each lap on measured cost, roll back to the best feedforward with a heavier step weight, and add `best_refresh` averaging, since laps under feedback are noisier than open-loop sim trials.

## Suggested first experiment

A **steady drift circle** with low-dimensional ILC:
- The TVLQR holds R and β.
- The ILC learns only the steady countersteer and drive current, i.e. 2 parameters per lap, updated by a secant step.

This is the drift version of Stage III (a few landing rows against the forces). It needs the mocap state, the current-mode actuation and the stabilizer, but not the full lifted machinery. Then extend to a full feedforward profile on a drift figure-eight (transitions between equilibria), where the lifted QP earns its keep.
