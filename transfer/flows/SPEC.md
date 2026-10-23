# Flow analysis: how sim-to-sim transfer corrupts the closed-loop vector field, and how DistILC re-corrects it

Requested by the user on 2026-10-22: "I want to see how the vector field gets corrupted in sim-to-sim transfer, then
how our method re-corrects it over several iterations within a bounded set of goals, for all of our cases, and
starting from any policy."

One worker per domain (pogo, car, quad) writes into `log/dilc/flows/<domain>/`. The cross-domain figure is built
afterwards from each domain's `summary.json`.

## The object
- **The closed-loop flow.** A policy pi (goal-conditioned), a goal g and a system s together define the closed-loop
  map
  - F^s_pi(x, g): the state after one step of the domain's natural map, started from state x.
  - The vector field is V^s_pi(x, g) = F^s_pi(x, g) - x.
- **The systems.**
  - s = sim: the nominal model the policy was trained in.
  - s = real twin: each "real" robot / car / stick. They are perturbed simulators, hence "sim-to-sim".
- **The reference flow.** V^sim_{pi_0}(x, g): the sim-trained start policy in its own world, the flow the designer
  intended.
  - Where a domain has a plan or reference trajectory, also report the field error against the plan's flow, and say
    which one you use.
- **The corruption** at iteration 0: dV_0 = V^real_{pi_0} - V^sim_{pi_0}.
- **The re-correction:** dV_i = V^real_{pi_i} - V^sim_{pi_0} for the hardware-stage iterations i = 0..N, where pi_i
  is the network after i iterations.
  - Expect it to shrink along the states the goals visit, inside the bounded goal set (the trained goals and their
    hull).
  - Show whether it shrinks or grows away from the trained goals, and off the visited tube.
- **The states x.** A tube around the trajectories that the sim pi_0 and the real pi_i visit at each goal (the
  visited states plus perturbations of a stated size), normalized by the domain's state scales.
- **Goal sets.** A domain-specific choice is fine, but report all three:
  - IN = the stage's training goals;
  - HULL = validation goals inside their convex hull / neighbourhood;
  - OUT = validation goals outside it.

## Deliverables (per domain, in `log/dilc/flows/<domain>/`)
1. **A script** `flows_<domain>.py` (reproducible, CLI). It computes the fields and metrics from saved per-iteration
   checkpoints.
2. **Figures.**
   - **(a) Phase portraits.** A quiver or stream plot in the domain's natural 2-D projection: sim pi_0, real pi_0
     (corrupted), real pi_i at 2–3 iterations, and real pi_N. Show it for one representative robot and goal, plus
     a robot outside the DR family if the domain has one. Mark the goal / fixed point / reference.
   - **(b) Field error vs iteration.** Normalized mean |dV_i| on IN / HULL / OUT for every start policy, with CIs
     over robots (and seeds where they exist).
   - **(c) A task-level link.** The fixed point, equilibrium or landing error vs iteration, so the field picture ties
     to the success metric.
3. **`summary.json`** with this exact schema, so the domains can be merged:
   ```json
   {"domain": "pogo", "map": "apex-to-apex return map", "proj": ["apex h", "vx"],
    "starts": {"<start policy name>": {"iters": [0,1,...], "robots": ["..."],
        "field_err": {"IN": [[per-robot values per iter]...], "HULL": [...], "OUT": [...]},
        "task_err":  {"IN": [...], "HULL": [...], "OUT": [...]}}},
    "notes": "..."}
   ```
4. **`FLOWS.md`.** What was computed and the main numbers (with CIs), plus an honest reading. Keep it short, in the
   style of the repo's other NOTES files: tables and bullets, no filler.

## Start policies ("starting from any policy")
Use every start policy the domain already has a hardware-stage path for: ours (nominal), ours+DR, PPO+DR, RMA, and
FADA's source if applicable.
- If per-iteration checkpoints do not exist, rerun the hardware stage on the VALIDATION goals with per-iteration
  saving. Add a flag if needed: append-only, defaults unchanged.
- Keep the stage's frozen recipe for each domain; do not retune it.

## Rules (binding)
- **Validation only. Never evaluate on spent test seeds:**
  - pogo 1301 / 2301 / 3301;
  - car 1701 / 2701 / 3701 (car val = 301);
  - quad planefinal (seed 701).
- **Resources:** ≤ 8 CPU processes per worker (≤ 32 in total on the machine, which is shared).
  - GPUs 1–3 are taken by the quad sim retraining (2026-10-22). Use `JAX_PLATFORMS=cpu` unless told otherwise.
  - GPU 0 is someone else's.
- **Shell:**
  - no foreground sleeps; use background waiters;
  - never `pkill -f` / `pgrep -f` a pattern that appears in your own command line (kill by PID);
  - never edit a bash script while it runs.
- **Repo:** do NOT commit. New flags in existing scripts are append-only, with defaults that reproduce old runs.
- **Method constraints** (for any new runs): one network, no residual policy, no system ID on the robots, the sim
  stays nominal (DR only as the existing variant).
