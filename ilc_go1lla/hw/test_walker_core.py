"""Check hw/walker_core.py against the MJX walker (2026-11-21). CPU:
    CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu ~/venvs/pg0/bin/python hw/test_walker_core.py --npz hw/policies/nominal.npz
On a simulated walk (nominal robot and a perturbed one): the command law, the observation rebuilt from the signals the robot has
(trunk pose, world velocity, body angular velocity, joints), the numpy policy on the logged observations, the WalkLog field
conventions (tau / q_after one step on)."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ilc_go1lla import walkenv as WE   # noqa: E402
import walker_core as C                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--params', default='runs/walker_nom_s0/params.pkl')
ap.add_argument('--npz', required=True)
a = ap.parse_args()

assert np.allclose(C.HOME, np.asarray(WE.make_env()._default_pose)), 'home pose'
pol = C.Policy(a.npz)
ok = True
for case, pert in (('nominal', {}), ('payload4', dict(payload=(4.0, -0.10, 0.08)))):
    w = WE.Walker(a.params, stand_at_goal=True, **pert)
    th = np.radians(7.6)
    g = np.array([2.5 * np.cos(th), 2.5 * np.sin(th), 1.0], np.float32)
    n = C.walk_steps(g)
    out = w.rollout(g, n=n)
    cmd = np.stack([C.command(out['p'][k], C.yaw_of_R(C.quat_to_R(out['quat'][k])), g, k * C.DT) for k in range(n)])
    obs = []
    for k in range(n):
        R = C.quat_to_R(out['quat'][k])
        last = np.zeros(12) if k == 0 else out['a'][k - 1]
        obs.append(C.build_obs(R.T @ out['v'][k, :3], out['v'][k, 3:6], R, out['qj'][k], out['qd'][k], last, out['cmd'][k]))
    obs = np.stack(obs)
    k_sw = int(np.ceil(C.handover_time(g) / C.DT))
    act = np.stack([pol(o) for o in out['obs_state'][:k_sw]])
    e_cmd = np.abs(cmd - out['cmd']).max()
    e_obs = np.abs(obs - out['obs_state']).max(0)
    e_act = np.abs(act - out['a'][:k_sw]).max()
    print(f'{case}: n {n} hand-over step {k_sw}  max|cmd| err {e_cmd:.2e}  max|obs| err by block: lin {e_obs[0:3].max():.2e} '
          f'gyro {e_obs[3:6].max():.2e} grav {e_obs[6:9].max():.2e} q {e_obs[9:21].max():.2e} dq {e_obs[21:33].max():.2e} '
          f'last_a {e_obs[33:45].max():.2e} cmd {e_obs[45:48].max():.2e}  max|a| err {e_act:.2e}')
    # trunk-motion blocks (lin, gyro, grav): MJX reads its sensors one physics substep (4 ms) before the logged state, so they
    # differ here by design; the builder's math is checked against MuJoCo's sensors at the same state in hw/test_obs_sensors.py
    ok &= e_cmd < 1e-4 and e_obs[9:].max() < 1e-4 and e_act < 1e-4
    assert np.allclose(out['a'][k_sw:], 0), 'stand controller after the hand-over'
    # WalkLog conventions vs the simulated tau / q_after
    assert np.allclose(out['q_after'][:-1], out['qj'][1:], atol=1e-5), 'q_after = next qj'
# order maps
can = np.arange(12)
assert np.array_equal(can[C.PG_FROM_CAN][C.CAN_FROM_PG], can)
assert [C.PG_LEGS[i // 3] for i in range(12)] == [C.CAN_LEGS[j // 3] for j in C.PG_FROM_CAN]
print('ALL OK' if ok else '!!! MISMATCH')
