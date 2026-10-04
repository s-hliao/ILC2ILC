import rclpy
import time
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
import casadi as ca
from scipy.linalg import expm
from params import F110
from tf_transformations import euler_from_quaternion
from trajectory_generator import generate_s_curve
from blend_solver import setup_mpc, _get_tire_params, linear_model
from trajectory_analysis import compare_trajectories
from tqdm import tqdm
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from geometry_msgs.msg import PoseWithCovarianceStamped
class F1tenth_ILC(Node):
    def __init__(self):
        super().__init__("F1tenth_ILC_Node")
        self.first_6_state = None
        self.odom_count = 0
        self.pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )
        self.reset_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )
        self.pose_sub = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.pose_callback,
            10
        )
    def pose_callback(self, msg):
        # state: x, y, phi(yaw), vx, vy, omega(yaw rate), omega_w(wheel angular velocity),
        # current, delta(steer)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        quaternion = [
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ]
        _, _, phi = euler_from_quaternion(quaternion)
        v_x = msg.twist.twist.linear.x
        v_y = msg.twist.twist.linear.y
        omega = msg.twist.twist.angular.z
        self.first_6_state = np.array([x, y , phi, v_x, v_y, omega])
        self.odom_count += 1
    def reset_car(self, initial_state):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        x = initial_state[0]
        y = initial_state[1]
        phi = initial_state[2]
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = 0.0
        # yaw -> quaternion
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = np.sin(phi / 2.0)
        msg.pose.pose.orientation.w = np.cos(phi / 2.0)
        self.reset_pub.publish(msg)
    def publish_control(self, steer, v):
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steer
        msg.drive.speed = v
        self.pub.publish(msg)
def rollout(node, new_controls, initial_state, dt, gear_ratio, pole_pairs, lam, mass, rw, max_steer_angle):
    new_trajectory = []
    steer = initial_state[8]
    velocity = initial_state[3]
    I = initial_state[7]
    # publish control and record state
    for t in tqdm(range(new_controls.shape[0]), desc="Rollout", leave=False):
        slew_rate = new_controls[t, 0]
        steer_rate = new_controls[t, 1]
        state = np.zeros(9)
        state[0:6] = node.first_6_state
        state[6] = node.first_6_state[3]/rw
        state[7] = I
        state[8] = steer
        new_trajectory.append(state)
        I = I + slew_rate * dt
        tau_drive = gear_ratio * 1.5 * pole_pairs * lam * I
        acc = tau_drive/(mass*rw)
        # publish control
        velocity = velocity + acc * dt
        velocity = max(velocity, 0.0)
        steer = steer + steer_rate * dt
        steer = np.clip(steer, -max_steer_angle, max_steer_angle)
        if (
            not np.isfinite(velocity)
            or not np.isfinite(steer)
        ):
            raise RuntimeError(
                f"invalid command at step {t}: "
                f"velocity={velocity}, steer={steer}"
            )
        if t < 5:
            print(
                f"step {t}: "
                f"slew={slew_rate:.4f}, "
                f"steer_rate={steer_rate:.4f}, "
                f"I={I:.4f}, "
                f"velocity={velocity:.4f}, "
                f"steer={steer:.4f}"
            )
        old_count = node.odom_count
        node.publish_control(
            steer,
            velocity
        )
        step_start = time.time()
        got_new_odom = False
        while True:
            elapsed = time.time() - step_start
            rclpy.spin_once(
                node,
                timeout_sec=0.005
            )
            if node.odom_count > old_count:
                got_new_odom = True
            # only proceed when BOTH conditions are satisfied
            if elapsed >= dt and got_new_odom:
                break
            if elapsed > 0.1:
                raise RuntimeError(
                    f"odom timeout at step {t}"
                )
        if (
            not np.all(np.isfinite(node.first_6_state))
            or abs(node.first_6_state[5]) > 20.0
        ):
            raise RuntimeError(
                f"\nVehicle diverged at step {t}\n"
                f"state = {node.first_6_state}\n"
                f"v_cmd = {velocity:.4f}\n"
                f"steer = {steer:.4f}\n"
                f"steer_rate = {steer_rate:.4f}\n"
                f"I = {I:.4f}"
            )
    state = np.zeros(9)
    state[0:6] = node.first_6_state
    state[6] = node.first_6_state[3]/rw
    state[7] = I
    state[8] = steer
    new_trajectory.append(state)
    return np.array(new_trajectory)
def build_vcmd_regularization(
    controls,
    actual_trajectory,
    initial_state,
    dt,
    gear_ratio,
    pole_pairs,
    lam,
    mass,
    rw
):
    N = controls.shape[0]
    nu = 2

    accel_per_current = (
        gear_ratio
        * 1.5
        * pole_pairs
        * lam
        / (mass * rw)
    )

    v_cmd = np.zeros(N)
    H_cmd = np.zeros((N, N * nu))

    I = initial_state[7]
    velocity = initial_state[3]

    for k in range(N):
        I += controls[k, 0] * dt
        velocity += accel_per_current * I * dt
        v_cmd[k] = velocity

        for j in range(k + 1):
            H_cmd[k, 2 * j] = (
                accel_per_current
                * dt**2
                * (k - j + 1)
            )

    actual_vx = actual_trajectory[1:, 3]
    v_cmd_gap = v_cmd - actual_vx

    return v_cmd_gap, H_cmd, v_cmd


def reset_and_wait(
    node,
    initial_state,
    timeout=1.0,
    max_attempts=3
):
    for attempt in range(max_attempts):
        node.publish_control(0.0, 0.0)
        stop_start = time.time()
        while time.time() - stop_start < 0.1:
            rclpy.spin_once(
                node,
                timeout_sec=0.01
            )
        node.reset_car(initial_state)
        start = time.time()
        while time.time() - start < timeout:
            rclpy.spin_once(
                node,
                timeout_sec=0.01
            )
            if node.first_6_state is None:
                continue
            state = node.first_6_state
            pos_error = np.linalg.norm(
                state[:2]
                - initial_state[:2]
            )
            yaw_error = np.arctan2(
                np.sin(
                    state[2]
                    - initial_state[2]
                ),
                np.cos(
                    state[2]
                    - initial_state[2]
                )
            )
            vel_error = np.linalg.norm(
                state[3:6]
                - initial_state[3:6]
            )
            if (
                pos_error < 0.05
                and abs(yaw_error) < 0.05
                and vel_error < 0.05
            ):
                print(
                    "Reset succeeded:",
                    state
                )
                return True
        print(
            f"Reset attempt "
            f"{attempt + 1} failed"
        )
    return False
def create_jacobian_functions(params_car):
    x = ca.MX.sym('x', 9)
    u = ca.MX.sym('u', 2)
    p = ca.MX.sym('p', 5)
    f = linear_model(
        x,
        u,
        p,
        params_car,
        exact=False
    )
    Ac = ca.jacobian(f, x)
    Bc = ca.jacobian(f, u)
    Ac_func = ca.Function(
        "Ac_func_noilc",
        [x, u, p],
        [Ac]
    )
    Bc_func = ca.Function(
        "Bc_func_noilc",
        [x, u, p],
        [Bc]
    )
    return Ac_func, Bc_func
def build_lifted_matrix(
    trajectory,
    controls,
    ref_trajectory,
    tire_params,
    Ac_func,
    Bc_func,
    dt,
    memory_steps=10
):
    N = controls.shape[0]
    nx = 9
    nu = 2
    ny = 5
    G = np.zeros((N * ny, N * nu))
    state_sensitivity = np.zeros((nx, N * nu))
    max_real_continuous = -np.inf
    max_discrete_radius = 0.0
    for k in range(N):
        Ac = np.array(
            Ac_func(
                trajectory[k],
                controls[k],
                tire_params
            ),
            dtype=float
        )
        Bc = np.array(
            Bc_func(
                trajectory[k],
                controls[k],
                tire_params
            ),
            dtype=float
        )
        continuous_eigs = np.linalg.eigvals(Ac)
        max_real_continuous = max(
            max_real_continuous,
            np.max(np.real(continuous_eigs))
        )
        # exact ZOH discretization of the local linear model
        M = np.zeros((nx + nu, nx + nu))
        M[:nx, :nx] = Ac
        M[:nx, nx:] = Bc
        Md = expm(M * dt)
        A = Md[:nx, :nx]
        B = Md[:nx, nx:]
        discrete_radius = np.max(
            np.abs(np.linalg.eigvals(A))
        )
        max_discrete_radius = max(
            max_discrete_radius,
            discrete_radius
        )
        # propagate existing sensitivity one step
        state_sensitivity = A @ state_sensitivity
        # add direct effect of u_k on x_{k+1}
        state_sensitivity[
            :,
            k * nu:(k + 1) * nu
        ] += B
        # only keep the most recent control sensitivities
        oldest_control = max(
            0,
            k - memory_steps + 1
        )
        if oldest_control > 0:
            state_sensitivity[
                :,
                :oldest_control * nu
            ] = 0.0
        phi_ref = ref_trajectory[k + 1, 2]
        # outputs: along-track, cross-track, phi, omega, vx
        C = np.zeros((ny, nx))
        C[0, 0] = np.cos(phi_ref)
        C[0, 1] = np.sin(phi_ref)
        C[1, 0] = -np.sin(phi_ref)
        C[1, 1] = np.cos(phi_ref)
        C[2, 2] = 1.0
        C[3, 5] = 1.0
        C[4, 3] = 1.0
        G[
            k * ny:(k + 1) * ny,
            :
        ] = C @ state_sensitivity
    print(
        f"max continuous real eig={max_real_continuous:.3e}, "
        f"max discrete spectral radius={max_discrete_radius:.3e}"
    )
    return G
def main():
    #car parameters
    params_car = F110()
    tire_params = _get_tire_params(params_car)
    mass = params_car['mass']
    Iz   = params_car['Iz']
    lf   = params_car['lf']
    lr   = params_car['lr']
    rw   = params_car['rw']     # rear wheel radius [m]
    Iw   = params_car['Iw']     # rear driveline rotational inertia [kg m^2]
    Im = params_car['Im']
    pole_pairs  = params_car['pole_pairs']
    gear_ratio  = params_car['gear_ratio']
    lam = params_car['lambda']
    max_steer_angle = params_car['max_steer']
    #ILC parameters
    epochs = 30
    noilc_learning_rate = 0.25
    r_update = 100.0
    s_correction = 0.1
    memory_steps = 10
    v_cmd_gap_scale = 1.0  # [m/s], smaller = stronger regularization
    STEER_ANGLE_MAX = 0.4189  # rad, about 24 deg
    rclpy.init()
    node = F1tenth_ILC()
    dt = 0.05
    ref_trajectory = generate_s_curve(
        params_car=params_car,
        path_length=8.0,
        vx_ref=3.0,
        dt=dt,
        kappa_max=0.20,
        hard=False
    )
    N = ref_trajectory.shape[0] - 1
    horizon = N * dt
    # errors: along-track, cross-track, phi, omega, vx
    error_scale = np.array([
        1.0,
        1.0,
        0.5,
        1.0,
        1.0
    ])
    Q_stage = np.diag(
        1.0 / (error_scale**2)
    )
    Q = np.kron(
        np.eye(N),
        Q_stage
    )
    # normalize the two controls by their physical ranges
    control_scale = np.array([
        300.0,
        params_car['max_steer_vel']
    ])
    R_stage = r_update * np.diag(
        1.0 / (control_scale**2)
    )
    S_stage = s_correction * np.diag(
        1.0 / (control_scale**2)
    )
    R = np.kron(
        np.eye(N),
        R_stage
    )
    S = np.kron(
        np.eye(N),
        S_stage
    )
    solver = setup_mpc(steps=N, horizon=horizon, solver_config="test", build=True)
    p_car = F110()
    tire_params = _get_tire_params(p_car)
    # Stage-wise parameters: [Cf, Cr, muf, mur, Cro, x_ref(6)] for k = 0..N
    for k in range(N + 1):
        p_k = np.concatenate([tire_params, ref_trajectory[k, :6]])
        solver.set(k, "p", p_k)
    # Initial state: x, y, phi, vx, vy, omega, omega_w, current, delta
    x0 = ref_trajectory[0].copy()                 # 9 insteaf of 10?
    '''
    x0[3] = 3.0                       # vx
    x0[6] = x0[3] / p_car['rw']       # omega_w, consistent with vx (no initial slip)
    '''
    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)
    # Warm start
    for k in range(N + 1):
        solver.set(k, "x", ref_trajectory[k])
    for k in range(N):
        solver.set(k, "u", np.zeros(2))
    status = solver.solve()
    if status != 0:
        print(f"Solver returned status {status} (nonzero => check solver.print_statistics())")
        solver.print_statistics()
    else:
        print("Solve succeeded.")
    cost = solver.get_cost()
    print(f"Cost: {cost:.4f}")
    print("\nPredicted trajectory (x, y, phi, vx, vy, omega, omega_w, current, delta):")
    for k in range(N + 1):
        xk = solver.get(k, "x")
        print(f"  k={k:2d}: {np.array2string(xk, precision=3, suppress_small=True)}")
    cur_trajectory = solver.get_flat("x").reshape(N+1, 9)
    cur_controls = solver.get_flat("u").reshape(N, 2)
    nominal_controls = cur_controls.copy()
    control_correction = np.zeros_like(nominal_controls)
    # build NOILC Jacobian functions after the nominal solve
    Ac_func, Bc_func = create_jacobian_functions(
        params_car
    )
    print(f"First control [slew_rate, steer_vel]: {cur_controls[0]}")
    print("First 5 nominal controls:")
    print(cur_controls[:5])
    initial_state = ref_trajectory[0].copy()
    error_history = []
    for i in tqdm(range(epochs), desc="ILC epochs"):
        #reset initial pose and control
        # first stop the car
        reset_ok = reset_and_wait(
            node,
            initial_state
        )
        if not reset_ok:
            print(f"Epoch {i}: reset failed, skip rollout")
            continue
        print("state after reset:", node.first_6_state)
        #apply the new controls in f1tenth and get a new trajectory
        new_trajectory = rollout(node, cur_controls, initial_state, dt, gear_ratio, pole_pairs, lam, mass, rw, max_steer_angle)
        error = ref_trajectory - new_trajectory
        # position error for this epoch
        position_error = np.linalg.norm(
            error[:, 0:2],
            axis=1
        )
        position_rmse = np.sqrt(
            np.mean(position_error**2)
        )
        error_history.append(position_rmse)
        vx_rmse = np.sqrt(
            np.mean(error[:, 3]**2)
        )
        vx_error = error[:, 3]
        # path-frame position error
        phi_ref = ref_trajectory[:, 2]
        dx = error[:, 0]
        dy = error[:, 1]
        along_track_error = (
            np.cos(phi_ref) * dx
            + np.sin(phi_ref) * dy
        )
        cross_track_error = (
            -np.sin(phi_ref) * dx
            + np.cos(phi_ref) * dy
        )
        phi_error = np.arctan2(
            np.sin(ref_trajectory[:, 2] - new_trajectory[:, 2]),
            np.cos(ref_trajectory[:, 2] - new_trajectory[:, 2])
        )
        omega_error = (
            ref_trajectory[:, 5]
            - new_trajectory[:, 5]
        )
        x_rmse = np.sqrt(
            np.mean(error[:, 0]**2)
        )
        y_rmse = np.sqrt(
            np.mean(error[:, 1]**2)
        )
        at_rmse = np.sqrt(
            np.mean(along_track_error**2)
        )
        ct_rmse = np.sqrt(
            np.mean(cross_track_error**2)
        )
        phi_rmse = np.sqrt(
            np.mean(phi_error**2)
        )
        omega_rmse = np.sqrt(
            np.mean(omega_error**2)
        )
        print(
            f"\nEpoch {i}: "
            f"pos={position_rmse:.4f} | "
            f"x={x_rmse:.4f}, "
            f"y={y_rmse:.4f} | "
            f"at={at_rmse:.4f}, "
            f"ct={ct_rmse:.4f}, "
            f"phi={phi_rmse:.4f}, "
            f"omega={omega_rmse:.4f} | "
            f"vx={vx_rmse:.4f}"
        )
        print(
            f"omega error mean={np.mean(omega_error):.4f}, "
            f"omega error max={np.max(omega_error):.4f}, "
            f"omega error min={np.min(omega_error):.4f}"
        )
        # NOILC error vector: x_1 to x_N
        ilc_error = np.column_stack((
            along_track_error[1:],
            cross_track_error[1:],
            phi_error[1:],
            omega_error[1:],
            vx_error[1:]
        )).reshape(-1)
        # model sensitivity over the whole trajectory
        G = build_lifted_matrix(
            new_trajectory,
            cur_controls,
            ref_trajectory,
            tire_params,
            Ac_func,
            Bc_func,
            dt,
            memory_steps=memory_steps
        )
        print(
            f"G max={np.max(np.abs(G)):.3e}, "
            f"G finite={np.all(np.isfinite(G))}"
        )
        if not np.all(np.isfinite(G)):
            raise RuntimeError(
                "Lifted sensitivity matrix G contains NaN/Inf"
            )

        # Penalize v_cmd getting too far ahead of measured vx.
        # During this least-squares update, measured vx is treated as fixed.
        v_cmd_gap, H_cmd, v_cmd_profile = build_vcmd_regularization(
            cur_controls,
            new_trajectory,
            initial_state,
            dt,
            gear_ratio,
            pole_pairs,
            lam,
            mass,
            rw
        )

        print(
            f"v_cmd gap: "
            f"rmse={np.sqrt(np.mean(v_cmd_gap**2)):.3f}, "
            f"max={np.max(np.abs(v_cmd_gap)):.3f}, "
            f"v_cmd max={np.max(v_cmd_profile):.3f}, "
            f"vx max={np.max(new_trajectory[:, 3]):.3f}"
        )

        Q_sqrt = np.diag(
            np.sqrt(np.diag(Q))
        )
        R_sqrt = np.diag(
            np.sqrt(np.diag(R))
        )
        S_sqrt = np.diag(
            np.sqrt(np.diag(S))
        )
        W_gap_sqrt = (
            np.eye(N)
            / v_cmd_gap_scale
        )

        A_ls = np.vstack((
            Q_sqrt @ G,
            R_sqrt,
            S_sqrt,
            W_gap_sqrt @ H_cmd
        ))

        b_ls = np.concatenate((
            Q_sqrt @ ilc_error,
            np.zeros(2 * N),
            -S_sqrt @ control_correction.reshape(-1),
            -W_gap_sqrt @ v_cmd_gap
        ))
        delta_U, _, _, _ = np.linalg.lstsq(
            A_ls,
            b_ls,
            rcond=1e-6
        )
        delta_controls = delta_U.reshape(
            N,
            2
        )
        new_controls = (
            cur_controls
            + noilc_learning_rate * delta_controls
        )
        new_controls[:, 0] = np.clip(
            new_controls[:, 0],
            -300,
            300
        )
        new_controls[:, 1] = np.clip(
            new_controls[:, 1],
            -params_car['max_steer_vel'],
            params_car['max_steer_vel']
        )
        # keep correction consistent with the controls actually applied
        control_correction = (
            new_controls
            - nominal_controls
        )
        print(
            f"  learned correction: "
            f"slew max={np.max(np.abs(control_correction[:, 0])):.4f}, "
            f"steer max={np.max(np.abs(control_correction[:, 1])):.4f}"
        )
        slew_sat = np.sum(
            np.abs(new_controls[:, 0]) >= 299.9
        )
        steer_sat = np.sum(
            np.abs(new_controls[:, 1])
            >= params_car['max_steer_vel'] - 1e-4
        )
        print(
            f"  saturation: "
            f"slew={slew_sat}/{N}, "
            f"steer={steer_sat}/{N}"
        )
        #then update cur_trajectory and cur_control
        cur_controls = new_controls
        cur_trajectory = new_trajectory

    print("\n===== Final learned controls =====")
    print("k | slew_rate | steer_rate | corr_slew | corr_steer")
    final_correction = cur_controls - nominal_controls
    for k in range(N):
        print(
            f"{k:02d} | "
            f"{cur_controls[k, 0]:9.4f} | "
            f"{cur_controls[k, 1]:10.4f} | "
            f"{final_correction[k, 0]:9.4f} | "
            f"{final_correction[k, 1]:10.4f}"
        )

    print("\n===== Integrated final command profile =====")
    I_check = initial_state[7]
    v_cmd_check = initial_state[3]
    steer_check = initial_state[8]
    accel_per_current = (
        gear_ratio
        * 1.5
        * pole_pairs
        * lam
        / (mass * rw)
    )

    print("k | current | v_cmd | steer_angle")
    for k in range(N):
        I_check += cur_controls[k, 0] * dt
        v_cmd_check += accel_per_current * I_check * dt
        steer_check += cur_controls[k, 1] * dt
        steer_check = np.clip(
            steer_check,
            -max_steer_angle,
            max_steer_angle
        )

        print(
            f"{k:02d} | "
            f"{I_check:8.4f} | "
            f"{v_cmd_check:7.3f} | "
            f"{steer_check:10.4f}"
        )

    compare_trajectories(ref_trajectory, cur_trajectory, dt)
    node.destroy_node()
    rclpy.shutdown()
if __name__ == "__main__":
    main()
