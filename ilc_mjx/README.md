# ilc_mjx: the jump on the GPU

Many Go1 jumps in parallel in MuJoCo MJX (JAX). It is for training at scale: value-gradient pretraining
with finite-difference Jacobians, and the domain-randomization baselines. It is a separate package from
`ilc_quad`, which keeps the CPU simulator, the ROS controller and the robot bridges. Evaluation stays on
`ilc_quad`'s CPU asynchronous real-like robots, so every result trained here is a cross-simulator transfer.

What it models (deliberately simpler than `ilc_quad`'s CPU sim):

- **Synchronous and nominal.** No reality model: no delays, jitter, dropped commands or mocap. The policy
  reads the true SRB state.
- **Foot–floor contacts only,** and the constraint solver at 4 / 8 iterations (MJX's cost grows with
  contact pairs and iterations). With ilc_quad's full collision model and 100 iterations, MJX matches CPU
  MuJoCo exactly but runs about 700x slower.
- **Landing with a joint PD** to the home pose, as in Nguyen et al., "Mastering Agile Jumping Skills from
  Simple Practices with Iterative Learning Control" (arXiv 2408.02619). This is the controller's
  `landing_controller:=pd`; there is no QP balance controller.
- **The rest of the jump is ilc_quad's controller:** the TO torque plus `J^T R^T (u - u_ref)`, damping on
  stance legs, PD on swing legs, and levelled feet late in flight. The force policy is `dilc_execute.Driver`'s.

Finite-difference Jacobians (VG-SAC-FD) are batch entries. At each contact sample, every jump flies ten
copies of itself to the next sample: one per action channel and one per SRB coordinate. The state
perturbations are the minimum-norm change in the planar mass metric with the feet in contact held fixed.

## Layout

| File | Contents |
|---|---|
| `ilc_mjx/model.py` | The Go1 as `ilc_quad.QuadModel` builds it, on the device |
| `ilc_mjx/planar.py` | `PlanarQuadModel`'s kinematics in JAX (`scripts/check_planar.py` checks it against CasADi) |
| `ilc_mjx/robot.py` | `ilc_jump_sim`'s index maps without ROS |
| `ilc_mjx/jump.py` | `JumpEnv`: batched rollouts and their FD Jacobians |
| `ilc_mjx/serve.py` | Rollout server for `dilc_train.py` (`--gpu-rollouts`), in `dilc_execute serve`'s protocol |
| `scripts/feasibility.py` | CPU vs MJX trajectories and throughput |
| `scripts/smoke.py` | A batch of policy jumps, with and without FD |

## Environment

The `ilcmjx` conda environment on the host is a clone of `jaxenv` (JAX 0.9 with CUDA) plus MuJoCo
3.12 / MJX (matching the container's MuJoCo) and CasADi. `COLCON_IGNORE` keeps the ROS build out.

```bash
~/miniconda3/envs/ilcmjx/bin/python scripts/check_planar.py
~/miniconda3/envs/ilcmjx/bin/python scripts/smoke.py --gpu 1
# training: dilc_train.py ... --gpu-rollouts 0 --gpu-batch 128 [--fd-jac 0.05 0.1]
```
