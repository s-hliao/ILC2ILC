"""hw/walker_core.build_obs vs MuJoCo's own sensors at the SAME state (CPU mj_forward on the logged qpos/qvel), 2026-11-21:
separates the builder's math (frames, imu-site lever arm) from the MJX observation's timing."""
import os, sys
import numpy as np, mujoco
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ilc_go1lla import walkenv as WE
import walker_core as C
w = WE.Walker('runs/walker_nom_s0/params.pkl', stand_at_goal=True, payload=(4.0, -0.10, 0.08))
m = w.env.mj_model; d = mujoco.MjData(m)
g = np.array([2.5 * np.cos(0.13), 2.5 * np.sin(0.13), 1.0], np.float32)
out = w.rollout(g, n=C.walk_steps(g))
sid = lambda n: m.sensor_adr[m.sensor(n).id]
err = []
for k in range(0, len(out['p']), 7):
    d.qpos[:] = np.r_[out['p'][k], out['quat'][k], out['qj'][k]]; d.qvel[:] = np.r_[out['v'][k], out['qd'][k]]
    mujoco.mj_forward(m, d)
    lin = d.sensordata[sid('local_linvel'):sid('local_linvel') + 3]; gyr = d.sensordata[sid('gyro'):sid('gyro') + 3]
    grav = d.site_xmat[m.site('imu').id].reshape(3, 3).T @ np.array([0, 0, -1.0])
    R = C.quat_to_R(out['quat'][k])
    o = C.build_obs(R.T @ out['v'][k, :3], out['v'][k, 3:6], R, out['qj'][k], out['qd'][k], np.zeros(12), np.zeros(3))
    err.append([np.abs(o[0:3] - lin).max(), np.abs(o[3:6] - gyr).max(), np.abs(o[6:9] - grav).max(),
                np.abs(out['obs_state'][k][0:3] - lin).max(), np.abs(out['obs_state'][k][3:6] - gyr).max()])
e = np.array(err).max(0)
print(f'builder vs MuJoCo sensors at the same state: lin {e[0]:.1e} gyro {e[1]:.1e} grav {e[2]:.1e}')
print(f'MJX obs vs MuJoCo sensors at the logged state:  lin {e[3]:.1e} gyro {e[4]:.1e}  (the MJX observation timing)')
