# Quad flow analysis (staged 2026-10-22; not yet run)

Spec: `../SPEC.md`. Script: `flows_quad.py` (its docstring has the definitions). Runner: `../run_all.sh` (the quad
block) or the four commands below.

## What it computes
- **The map:** one control sample (10 ms) of the push. The SRB state x = (CoM x, z, pitch, vx, vz, pitch rate) at
  contact sample k goes to k + 1, for k < Nc = 60.
- **The fields, at the states the robot actually visited:**

| field | definition |
|---|---|
| real | what the robot did: its TRUE recorded pose (`rec_true_*`), with velocities from a centred 20 ms slope |
| sim0 | the nominal GPU sim's one-sample flight from the same state (restored as in `deploy.py --jac-at real`) under the start network, at the robot's own observation |
| simk | the same under the iteration-k network: how much the network itself changed the sim field |
| plan | the goal's reference flow x_ref[k+1] − x_ref[k] |

- **Errors** (state-scale units, mean over the push):

| error | definition |
|---|---|
| `field_err` | \|V_real − V_plan\|, goal-referenced (primary) |
| `field_err_sim` | \|V_real − V_sim0\|, the corruption / SPEC reference |
| `policy_change` | \|V_simk − V_sim0\| |
| `tube_err` | \|x_real − x_ref\| |
| `task_err` | the jump's score; also ok5 |

- **Goal sets:**
  - IN = the 6 training goals.
  - HULL = the 4 planeval goals inside their hull: (0.5625, 0), (0.53, 0.11), (0.51, 0.13), (0.56, 0.12).
  - OUT = the 4 outside: (0.4375, 0), (0.52, 0.16), (0.4625, 0.06), (0.6125, 0.07).
  - Validation seed 301 only; no test goals.

## Start policies (existing stage runs with it0..it4 networks, 3 seeds × 4 robots = 12 cells each)

| name | dirs |
|---|---|
| ours+DR | `plane/tr/recipe_sd*` (drB_s2 it30 + the recipe) |
| ours+DR random-dir (control) | `plane/tr/ctl_random_sd*` |
| PPO+DR | `plane/baseline_audit/after/hw24_on_ppo_dr_fix_s0_sd*` (fixed PPO+DR + our stage) |
| ours+DR on clean twins | `plane/clean/recipe_sd*` (no timing / sensing: the case where the stage converges) |
| random-dir on clean twins | `plane/clean/ctl_random_sd*` |
| ours+DR pool, stage_c_* arms | picked up automatically once `run_remaining_quad.sh` has run them |

## Run
```
cd /home/henry/ilc_ws/log/dilc/flows/quad
export ILC_QUAD_OVERLAY=/ilc_ws/log/dilc/plane/fixbox/overlay JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_ilc DILC_FLY_JOBS=8
PY=~/miniconda3/envs/ilcmjx/bin/python
$PY flows_quad.py list                 # starts found, goal sets
$PY flows_quad.py fly --par 3          # container ilc_quad: 5 starts x 12 cells x 5 networks x 14 goals x 2 jumps
$PY flows_quad.py compute --gpu 1      # one GPU
$PY flows_quad.py aggregate            # summary.json, summary_extra.json, fig_field_err / _sim / task_err / ok5
$PY flows_quad.py portraits --portrait-start "ours+DR" --portrait-robot real_r4 --portrait-goal 0.6 0.1
```

**Cost:** `fly` is 300 (cell, network) flights of 28 jumps each, 8 400 jumps in total. At about 8 jumps / min per
3 CPUs, that is several hours with `--par 3 DILC_FLY_JOBS=8`; raise both on a bigger machine. `compute` is one GPU
compile (~10 min) plus ~1 s per jump.

## Smoke test (matcha, 2026-10-22)
One cell (`smoke/recipe_sd9001/real_r4`, iteration 0, 1 jump per goal) went through `fly` and the per-jump fields:
see `smoke.log`, `diag.py` and `diag2.log`. The `smoke/` dir is not part of the analysis.
- **Fly:** 14 jumps in 8 s; the recordings are saved with `rec_true_*`.
- **Fields:** ~2 s per jump after a ~5 min compile.
- **Bug found and fixed:** the restored stance feet sat exactly on the floor, carried no load in the first sample,
  and the sim's one-sample flight fell freely (Δvz = −g dt on every goal: a field gap of 1.0 sx in vz).
  - They are now put at the nominal sim's soft-contact depth (`Fields.foot_z`, as `deploy.py --jac-at real` does).
  - After the fix, the sim0-vs-real gap per component (in sx) is x 0.04, z 0.02, pitch 0.04, vx 0.11, vz 0.22,
    pitch rate 0.30.
  - That is the same order as real-vs-plan (0.09 / 0.06 / 0.11 / 0.08 / 0.09 / 0.26) on real_r4 at goal (0.6, 0).

## Caveats
- `sim0` uses the robot's recorded observation, whose previous-action entries come from the network that flew it,
  not from pi_0. The difference is small, but the reference is not an exact pi_0 rollout.
- The real velocities come from a 20 ms centred slope of the true pose, and the restored sim states project the
  stance feet onto the floor. Both are needed for a clean one-sample flight, and both differ a little from the
  sim's own states. Compare `field_err_sim` at iteration 0 with the sim-vs-sim gap (`policy_change` at k = 0 is 0
  by construction).
- The jumps are random by timing: about 5 cm sd in landing x. The field errors are means over 2 jumps × 14 goals ×
  12 cells per point.
