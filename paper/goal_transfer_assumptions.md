# When does a hardware-stage correction transfer between goals?

Prompted by the criticism (2026-10-12) that the goal space is nonlinear while DistILC assumes transfer between goals.

This note gives a first-order statement of what has to hold for a correction learned at flown goals to improve an unflown
goal, and maps each assumption to the evidence we have. Analysis code: `src/ilc_mjx/paper/transfer_radius.py`
(`log/dilc/plane/transfer_radius.json`; figures `quadruped/figures/14_transfer_radius.png`, `car/figures/fig19_transfer_radius.png`).
The single-goal experiment: `log/dilc/plane/fixbox/single_goal_transfer.sh`.

## Setup

- **Policy and goals.** The goal-conditioned network is u = π_θ(o, g). A real trial at goal g has an outcome error e_g(θ)
  (landing error for the quadruped; path and pace error for the car).
- **The update at a flown goal g_i.** It is one damped Gauss-Newton ILC step through the NOMINAL simulator's lifted
  Jacobian G_i^sim, taken along the measured trial:

      u*_i = u_i − (G_iᵀ Q G_i + R)⁻¹ G_iᵀ Q e_i        (G_i = G_i^sim)

  The network is then refit, θ → θ', to the targets u*_i at the flown goals, with a trust region: anchors at sim states,
  damping, and the sim-checked gate.
- **The effect at an unflown goal g.** To first order the refit changes the output there by
  Δu_g ≈ Σ_i K_θ((o,g), (o_i,g_i)) Δu_i, where K_θ is the kernel the network's parameter update induces between inputs
  (its tangent kernel). The real error then changes by Δe_g ≈ G_g^real Δu_g.

**Transfer** means the unflown goal improves: e_gᵀ Q Δe_g < 0.

## Assumptions (sufficient, to first order)

| # | Assumption | What it means physically | Breaks when |
|---|---|---|---|
| **A1** | **Per-goal descent.** At each flown goal the nominal-sim Jacobian gives a descent direction for the real system: e_iᵀ Q G_i^real (G_i^simᵀ Q G_i^sim + R)⁻¹ G_i^simᵀ Q e_i > 0 (the standard approximate-model ILC condition, e.g. ‖I − G^real L‖ < 1 locally) | the simulator gets the direction of a correction right, not its exact size | gross model error, e.g. a ~30 % thrust-to-weight error (our "very wrong sim": fails) |
| **A2** | **The error is systematic and smooth in the goal**, within a mode: e_g = ē(g) + ε_g with ‖ē(g) − ē(g_i)‖ ≤ L_e ‖g − g_i‖ | the real-vs-sim discrepancy is robot-wide (thrust deficit, delay, friction bias), not specific to one goal | errors specific to a goal: reachability limits, box-edge contact, an early touchdown on one box height |
| **A3** | **Jacobian continuity, the same hybrid mode.** ‖G_g − G_{g_i}‖ ≤ L_G ‖g − g_i‖, for both the sim and the real Jacobian; requires g and g_i to share the contact-mode sequence and roughly its timing | nearby goals respond to a force change in the same way | across modes (a flat vs a box landing, a different touchdown order): G is discontinuous in g |
| **A4** | **Representational coupling and coverage.** The network spreads a flown goal's correction to g: K_θ(g, g_i) > 0 and decaying smoothly with distance; the flown trials cover the state region g's trajectory visits | the network shares structure across goals where their trajectories overlap | g's trajectory leaves the flown trials' state region; the kernel is local in goal |
| **A5** | **Consistency, no interference.** The corrections needed at the flown goals and at g are compatible: one parameter change satisfies them approximately (a low-rank shared correction), and the trust region does not pin g's region | one fix serves all goals | flown goals need opposite corrections (e.g. a box goal vs a flat goal); updates at some goals worsen others |
| **A6** | **Feasibility.** g is achievable on the real system within actuation limits | there is something to converge to | the goal is beyond the robot (r5's box goals) |
| **A7** | **Signal above trial noise.** Per-trial outcome noise is smaller than the correction it should reveal | the measured error carries the systematic part | ~1 cm corrections under 0.5–3 cm trial-to-trial noise with one trial per goal |

**Resulting statement.** Under A1–A5 within one mode, the first-order improvement at an unflown goal holds for
‖g − g_i‖ < r*, with

    r* ∝ (descent margin at g_i) / (L_G ‖e‖ + ‖G‖ L_e + kernel decay)

So transfer is local: a finite radius in the goal space, set by how fast the error (A2), the Jacobian (A3) and the
network's coupling (A4) change with the goal. Across a mode boundary there is no first-order guarantee, and
interference (A5) can make held-out goals worse. A6 and A7 bound what any per-trial method can show, ours or per-goal ILC.

## Evidence

Each line gives the domain and the source, then what it shows and which assumption it bears on.

**Car** (98 hardware-stage arms vs their zero-shot, 3,600 rows)
- The RMS e_y gain falls with sideslip distance to the nearest trained plan on the same track: +0.8 cm (0–5°), +0.3 cm
  (5–12°), +0.1 cm (12–30°). A finite radius of ~5–12°: A2, A3 and A4 hold within a smooth family.
- The ladder's rung choice on b25 carries to every plan of a track (friction ×0.8: 0 → 100 % on all plans). A
  discrete, robot-wide correction transfers.

**Quadruped** (49 arms, r1/s1/r4, 1,072 test-goal rows)
- Unflown test goals (≤ 6 cm from a flown goal) gain +0.7 to +1.1 cm |e_x|, vs +1.36 cm on the flown goals: about
  50–80 % of the flown gain carries over.
- Direction transfer is r = +0.34 for flat test goals vs **−0.08 for box test goals**. Flat: A2 and A3 partly hold. Box:
  the correction is not the neighbour's (A2/A3 fail across box heights).
- `hw_v8s1it10`: the training box goals improved (r1 J 36 → 2.7) while the held-out boxes fell from 22 % to 10 %. A
  violation of A5 (interference).
- Test-box bias grows with box height (−58 cm per m) and distance (−20 cm per m). A violation of A2 (goal-specific error).
- r5: its box goals stay 11–20 cm short whatever is done. A violation of A6.
- One jump per goal per iteration, std 0.5–3 cm. A7 (why the DR (B) stage shows +2 points).
- Single-goal stages (one flown goal, all 8 test goals, 1–20 cm, both modes): *running*. This is the direct radius
  measurement, by mode.

## How to write it

**Claim.** The sim stage provides the global goal structure (zero-shot 41.7 % on the quadruped's held-out goals). The
hardware stage's correction transfers **locally**:
- within a contact mode;
- within a radius set by how smoothly the robot's error and response vary with the goal (A2–A4);
- most clearly for robot-wide discrepancies (actuation, friction): the car's family, flat jumps, the ladder's rung choice.

**Limitation.** It does not cross contact-mode boundaries (box vs flat, box heights): there the per-goal error is not
the neighbour's (A2/A3) and updates can interfere (A5). Per-goal ILC is then better on the goals it flies, at the cost of
flying every goal.

**Practical rule.** Fly at least one goal per contact mode and per box-height bin. The radius measurement (single-goal
stages) gives its size.
