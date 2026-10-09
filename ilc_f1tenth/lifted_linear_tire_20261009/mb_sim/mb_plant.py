"""
The fork's multi-body model behind the stock f110_gym front end, for the ROS bridge
(mb_bridge.py) and offline checks. Per 0.01 s physics tick, as f110_gym RaceCar.update_pose:
the steering command goes through a 2-sample delay buffer, pid(speed, steer) turns the
commands into (steering velocity, acceleration), and the dynamics are integrated -- here MB
with RK4 substeps (the tire and wheel-spin modes are far stiffer than the single-track
model's; see SUBSTEP).

Below MB's kinematic switch (|vx| < 0.5 m/s) slips are zero and the tires carry no
longitudinal force, so the drive torque would spin the light RC wheels up freely
(m R_w a / I_y_w ~ 1e4 rad/s^2) and hand the dynamic model a huge slip at 0.5 m/s; the
wheels are held at rolling speed there instead, as init_mb sets them.
"""
import numba
import numpy as np

from fork_import import vehicle_dynamics_mb, init_mb, pid_steer, pid_accl, mb_vector, mb_index
from mb_params import F1TENTH_MB_PARAMETERS

KIN_THRESH = 0.5            # multi_body.py
SUBSTEP = 2.5e-4            # s; RK4 on the wheel-spin mode needs h < ~2.8 I_w v / (K_x R_w^2)

_f_mb = numba.njit(cache=True)(vehicle_dynamics_mb)


@numba.njit(cache=True)
def _integrate(x, u, p, dt, h, R_w, T_f, T_r):
    n = max(1, int(round(dt / h)))
    h = dt / n
    for _ in range(n):
        k1 = _f_mb(x.copy(), u, p)
        k2 = _f_mb(x + 0.5 * h * k1, u, p)
        k3 = _f_mb(x + 0.5 * h * k2, u, p)
        k4 = _f_mb(x + h * k3, u, p)
        x = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        for i in range(23, 27):
            if x[i] < 0.0:
                x[i] = 0.0
        if abs(x[3]) < KIN_THRESH:            # rolling wheels below the kinematic switch
            c, s = np.cos(x[2]), np.sin(x[2])
            vf = x[10] + p[3] * x[5]                  # lateral speed at the front axle
            x[23] = max(0.0, (x[3] + 0.5 * T_f * x[5]) * c + vf * s) / R_w
            x[24] = max(0.0, (x[3] - 0.5 * T_f * x[5]) * c + vf * s) / R_w
            x[25] = max(0.0, x[3] + 0.5 * T_r * x[5]) / R_w
            x[26] = max(0.0, x[3] - 0.5 * T_r * x[5]) / R_w
    return x


class MBPlant:
    def __init__(self, params=F1TENTH_MB_PARAMETERS, timestep=0.01, substep=SUBSTEP,
                 steer_buffer_size=2):
        self.params = params
        self.p = mb_vector(params)
        self.timestep = timestep
        self.substep = substep
        self.steer_buffer_size = steer_buffer_size
        self.R_w = self.p[mb_index('R_w')]
        self.T_f, self.T_r = self.p[mb_index('T_f')], self.p[mb_index('T_r')]
        self.reset((0.0, 0.0, 0.0))

    def reset(self, pose):
        x0 = np.zeros(7)
        x0[0], x0[1], x0[4] = pose
        self.x = init_mb(x0, self.p)
        self.steer_buffer = np.empty((0,))

    def step(self, raw_steer, speed):
        """One physics tick with commands (steering angle, speed), as RaceCar.update_pose."""
        if self.steer_buffer.shape[0] < self.steer_buffer_size:
            steer = 0.0
            self.steer_buffer = np.append(raw_steer, self.steer_buffer)
        else:
            steer = self.steer_buffer[-1]
            self.steer_buffer = self.steer_buffer[:-1]
            self.steer_buffer = np.append(raw_steer, self.steer_buffer)
        P = self.params
        sv = pid_steer(steer, self.x[2], P.sv_max)
        accl = pid_accl(speed, self.x[3], P.a_max, P.v_max, P.v_min)
        self.x = _integrate(self.x, np.array([sv, accl]), self.p, self.timestep, self.substep,
                            self.R_w, self.T_f, self.T_r)
        if not np.all(np.isfinite(self.x)):
            raise FloatingPointError(f'MB state diverged: {self.x}')

    def odom(self):
        """(x, y, yaw, linear.x, linear.y, yaw rate) as the stock bridge publishes them:
        linear.x the longitudinal speed, linear.y 0 (f110_gym never fills it)."""
        x = self.x
        return x[0], x[1], x[4], x[3], 0.0, x[5]

    def truth(self):
        """Hidden states for logs only: [vx, vy, steering angle, roll, pitch, wheel speeds]."""
        x = self.x
        return np.r_[x[3], x[10], x[2], x[6], x[8], x[23:27]]
