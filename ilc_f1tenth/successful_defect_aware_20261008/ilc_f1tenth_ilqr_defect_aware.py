import rclpy
import time
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
from params import F110
from tf_transformations import euler_from_quaternion
from trajectory_generator import generate_s_curve
from blend_solver import setup_mpc, _get_tire_params
from iLQR_blend import discrete_blend_dynamics_model, state_difference
import iLQR_blend as baseline_ilqr
import casadi as ca
import argparse
import json
import threading
from pathlib import Path
from trajectory_analysis import compare_trajectories
from tqdm import tqdm
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped


class F1tenth_ILC(Node):

    def __init__(self):
        super().__init__("F1tenth_ILC_Node")
        self.first_6_state = None
        self.odom_count = 0
        self.pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.reset_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.pose_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.pose_callback, 10
        )

    def pose_callback(self, msg):
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

        self.first_6_state = np.array([x, y, phi, v_x, v_y, omega])
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
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = np.sin(phi / 2.0)
        msg.pose.pose.orientation.w = np.cos(phi / 2.0)

        self.reset_pub.publish(msg)

    def publish_control(self, steer, v):
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(steer)
        msg.drive.speed = float(v)
        self.pub.publish(msg)


def reset_and_wait(node, initial_state, timeout=1.0, max_attempts=3):
    for attempt in range(max_attempts):
        node.publish_control(0.0, 0.0)

        stop_start = time.time()
        while time.time() - stop_start < 0.1:
            rclpy.spin_once(node, timeout_sec=0.01)

        old_count = node.odom_count
        node.reset_car(initial_state)

        start = time.time()
        good_samples = 0

        while time.time() - start < timeout:
            rclpy.spin_once(node, timeout_sec=0.01)

            if node.first_6_state is None or node.odom_count <= old_count:
                continue

            state = node.first_6_state

            pos_error = np.linalg.norm(state[:2] - initial_state[:2])
            yaw_error = np.arctan2(
                np.sin(state[2] - initial_state[2]),
                np.cos(state[2] - initial_state[2])
            )
            vel_error = np.linalg.norm(state[3:6] - initial_state[3:6])

            if (
                pos_error < 0.05
                and abs(yaw_error) < 0.05
                and vel_error < 0.05
            ):
                good_samples += 1
                if good_samples >= 2:
                    print("Reset succeeded:", state)
                    return True
            else:
                good_samples = 0

        print(f"Reset attempt {attempt + 1} failed")

    return False


def rollout(
    node,
    controls,
    initial_state,
    dt,
    gear_ratio,
    pole_pairs,
    lam,
    mass,
    rw,
    max_steer_angle
):
    new_trajectory = []
    steer = initial_state[8]
    velocity = initial_state[3]
    I = initial_state[7]

    if node.first_6_state is None:
        raise RuntimeError("No odometry available before rollout.")

    for t in tqdm(range(controls.shape[0]), desc="Rollout", leave=False):
        slew_rate = controls[t, 0]
        steer_rate = controls[t, 1]

        state = np.zeros(9)
        state[0:6] = node.first_6_state
        state[6] = node.first_6_state[3] / rw
        state[7] = I
        state[8] = steer
        new_trajectory.append(state)

        I = I + slew_rate * dt
        tau_drive = gear_ratio * 1.5 * pole_pairs * lam * I
        acc = tau_drive / (mass * rw)

        velocity = velocity + acc * dt
        velocity = max(velocity, 0.0)

        steer = steer + steer_rate * dt
        steer = np.clip(steer, -max_steer_angle, max_steer_angle)

        if not np.isfinite(velocity) or not np.isfinite(steer):
            raise RuntimeError(
                f"Invalid command at step {t}: velocity={velocity}, steer={steer}"
            )

        old_count = node.odom_count
        node.publish_control(steer, velocity)

        step_start = time.time()
        got_new_odom = False

        while True:
            elapsed = time.time() - step_start
            rclpy.spin_once(node, timeout_sec=0.005)

            if node.odom_count > old_count:
                got_new_odom = True

            if elapsed >= dt and got_new_odom:
                break

            if elapsed > 0.1:
                raise RuntimeError(f"Odom timeout at step {t}")

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
    state[6] = node.first_6_state[3] / rw
    state[7] = I
    state[8] = steer
    new_trajectory.append(state)

    return np.array(new_trajectory)


def solve_nmpc_initial_solution(ref_trajectory, dt):
    N = ref_trajectory.shape[0] - 1
    horizon = N * dt

    solver = setup_mpc(
        steps=N,
        horizon=horizon,
        solver_config="test",
        build=True
    )

    p_car = F110()
    tire_params = _get_tire_params(p_car)

    # Stage-wise parameters: [Cf, Cr, muf, mur, Cro, x_ref(6)] for k = 0..N
    for k in range(N + 1):
        p_k = np.concatenate([
            tire_params,
            ref_trajectory[k, :6]
        ])
        solver.set(k, "p", p_k)

    # Initial state: x, y, phi, vx, vy, omega, omega_w, current, delta
    x0 = ref_trajectory[0].copy()

    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)

    # Warm start
    for k in range(N + 1):
        solver.set(k, "x", ref_trajectory[k])

    for k in range(N):
        solver.set(k, "u", np.zeros(2))

    status = solver.solve()

    if status != 0:
        print(
            f"Solver returned status {status} "
            f"(nonzero => check solver.print_statistics())"
        )
        solver.print_statistics()
    else:
        print("Solve succeeded.")

    cost = solver.get_cost()
    print(f"Cost: {cost:.4f}")

    print(
        "\nPredicted trajectory "
        "(x, y, phi, vx, vy, omega, omega_w, current, delta):"
    )
    for k in range(N + 1):
        xk = solver.get(k, "x")
        print(
            f"  k={k:2d}: "
            f"{np.array2string(xk, precision=3, suppress_small=True)}"
        )

    cur_trajectory = solver.get_flat("x").reshape(N + 1, 9)
    cur_controls = solver.get_flat("u").reshape(N, 2)

    print(
        f"First control [slew_rate, steer_vel]: "
        f"{cur_controls[0]}"
    )
    print("First 5 nominal controls:")
    print(cur_controls[:5])

    return cur_trajectory, cur_controls

def trajectory_rmse(ref_trajectory, trajectory):
    error = ref_trajectory - trajectory
    position_error = np.linalg.norm(error[:, 0:2], axis=1)
    position_rmse = np.sqrt(np.mean(position_error**2))
    vx_rmse = np.sqrt(np.mean(error[:, 3]**2))

    return position_rmse, vx_rmse


# Independent experiment copied from ilc_f1tenth_ilqr_nmpc.py.
# State ordering is exactly the original nine states; no hidden response states.
STATE_NAMES = ['x', 'y', 'heading', 'vx', 'vy', 'yaw_rate',
               'wheel_speed', 'current', 'steering_command']


def trajectory_cost(states, controls, reference, Q, Q_f, R):
    error = state_difference(states, reference)
    return float(0.5*np.einsum('ti,ij,tj->', error[:-1], Q, error[:-1])
                 + 0.5*error[-1]@Q_f@error[-1]
                 + 0.5*np.einsum('ti,ij,tj->', controls, R, controls))


def nominal_transition(car, p, dt):
    x = ca.MX.sym('defect_experiment_x', 9)
    u = ca.MX.sym('defect_experiment_u', 2)
    expression = discrete_blend_dynamics_model(x, u, ca.DM(p), car, False, dt)
    return ca.Function('defect_experiment_nominal', [x, u], [expression])


def construct_defects(measured, controls, transition):
    # Heading is the only periodic coordinate. d has the same ordering as X.
    predicted = np.asarray([np.asarray(transition(x, u)).reshape(9)
                            for x, u in zip(measured[:-1], controls)])
    return state_difference(measured[1:], predicted)


def defect_step(transition, state, control, defect):
    result = np.asarray(transition(state, control)).reshape(9) + defect
    result[2] = np.arctan2(np.sin(result[2]), np.cos(result[2]))
    if not np.isfinite(result).all():
        raise RuntimeError('Nonfinite defect-aware transition')
    return result


def model_rollout(transition, initial, controls, car, defects=None):
    states = [np.array(initial, dtype=float, copy=True)]
    applied = []
    for i, control in enumerate(controls):
        clipped = np.clip(control, [-300., -car['max_steer_vel']],
                          [300., car['max_steer_vel']])
        state = defect_step(transition, states[-1], clipped,
                            np.zeros(9) if defects is None else defects[i])
        states.append(state); applied.append(clipped)
    return np.asarray(states), np.asarray(applied)


def defect_aware_forward(k, K, measured, old_controls, transition, defects, car, alpha):
    state = measured[0].copy(); states = [state.copy()]; controls = []
    for i, old in enumerate(old_controls):
        dx = state_difference(state, measured[i])
        control = np.clip(old + alpha*k[i] + K[i]@dx,
                          [-300., -car['max_steer_vel']],
                          [300., car['max_steer_vel']])
        state = defect_step(transition, state, control, defects[i])
        states.append(state.copy()); controls.append(control)
    return np.asarray(states), np.asarray(controls)


def magnitude_diagnostic(values):
    absolute = np.abs(values)
    step, dimension = np.unravel_index(np.argmax(absolute), absolute.shape)
    return dict(mean_absolute=float(absolute.mean()), max_absolute=float(absolute.max()),
                max_timestep=int(step), max_state_dimension=int(dimension),
                max_state_name=STATE_NAMES[dimension],
                per_state_max_absolute=absolute.max(axis=0).tolist())


def prepare_update(measured, old_controls, reference, dt, car, p,
                   Q, Q_f, R, alphas, output):
    if measured.shape != reference.shape or measured.shape != (len(old_controls)+1, 9):
        raise ValueError('Expected N+1 nine-state observations and N two-rate controls')
    if old_controls.shape[1] != 2 or not all(np.isfinite(v).all() for v in [measured, old_controls, reference]):
        raise ValueError('Invalid stored rollout')
    transition = nominal_transition(car, p, dt)
    defects = construct_defects(measured, old_controls, transition)
    reconstruction, applied_old = model_rollout(transition, measured[0], old_controls, car, defects)
    if np.max(np.abs(applied_old-old_controls)) > 1e-10:
        raise RuntimeError('Old controls violate physical rate limits; clipping breaks reconstruction')
    reconstruction_error = state_difference(reconstruction, measured)
    # Fixed d has zero derivatives: reuse original blend backward/Jacobian forms.
    baseline_ilqr.Q = Q.copy(); baseline_ilqr.Q_f = Q_f.copy(); baseline_ilqr.R = R.copy()
    k, K, _ = baseline_ilqr.backward(reference, measured, old_controls, p, car, False, dt)
    if len(k) != len(old_controls):
        raise RuntimeError('Backward pass incomplete')
    zero_states, zero_controls = defect_aware_forward(k, K, measured, old_controls,
                                                     transition, defects, car, 0.)
    zero_error = float(np.max(np.abs(zero_controls-old_controls)))
    if np.max(np.abs(reconstruction_error)) > 1e-7 or zero_error > 1e-7:
        idx = np.unravel_index(np.argmax(np.abs(zero_controls-old_controls)), old_controls.shape)
        raise RuntimeError(f'Consistency failed: alpha=0 control max error {zero_error}, timestep/control {idx}; reconstruction {magnitude_diagnostic(reconstruction_error)}')
    def evaluate(states, controls):
        return dict(cost=trajectory_cost(states, controls, reference, Q, Q_f, R),
                    position_rmse=float(trajectory_rmse(reference, states)[0]))
    nominal_old, nominal_old_u = model_rollout(transition, measured[0], old_controls, car)
    report = dict(state_names=STATE_NAMES, dt=dt, Q=Q.tolist(), Q_f=Q_f.tolist(), R=R.tolist(),
                  measured_previous=evaluate(measured, old_controls),
                  defect_construction=magnitude_diagnostic(defects),
                  reconstruction=magnitude_diagnostic(reconstruction_error),
                  alpha_zero_control_max_error=zero_error,
                  alpha_zero_state_max_error=float(np.max(np.abs(state_difference(zero_states, measured)))),
                  unchanged_defect=evaluate(reconstruction, applied_old),
                  unchanged_nominal=evaluate(nominal_old, nominal_old_u),
                  acceptance_model='defect-aware', original_nominal_role='diagnostic only', candidates=[])
    print('Measured previous rollout', json.dumps(report['measured_previous']))
    print('Defect construction', json.dumps(report['defect_construction']))
    print('Defect reconstruction', json.dumps(report['reconstruction']))
    print('alpha=0 max |U_forward - U_old|:', zero_error)
    print('Unchanged control baseline', json.dumps({key: report[key] for key in ['unchanged_defect', 'unchanged_nominal']}))
    selected_controls=old_controls.copy(); selected_states=reconstruction.copy(); best_cost=report['unchanged_defect']['cost']; selected_alpha=0.
    for alpha in alphas:
        if not np.isfinite(alpha) or alpha <= 0 or alpha > 1:
            raise ValueError('Candidate alpha must be in (0,1]')
        candidate_states, candidate_controls = defect_aware_forward(k, K, measured, old_controls, transition, defects, car, alpha)
        # Re-evaluate both candidates open-loop with the same evaluator as U_old.
        defect_prediction, evaluated_controls = model_rollout(transition, measured[0], candidate_controls, car, defects)
        if np.max(np.abs(state_difference(candidate_states, defect_prediction))) > 1e-7:
            raise RuntimeError('Forward/evaluation model mismatch')
        nominal_prediction, nominal_controls = model_rollout(transition, measured[0], candidate_controls, car)
        d=evaluate(defect_prediction,evaluated_controls); n=evaluate(nominal_prediction,nominal_controls)
        item=dict(alpha=float(alpha), defect=d, nominal=n,
                  defect_improvement=report['unchanged_defect']['cost']-d['cost'],
                  nominal_improvement=report['unchanged_nominal']['cost']-n['cost'])
        report['candidates'].append(item);print('Alpha candidate', json.dumps(item))
        if d['cost'] < best_cost:
            best_cost=d['cost'];selected_alpha=float(alpha);selected_controls=candidate_controls.copy();selected_states=defect_prediction.copy()
    accepted=selected_alpha>0
    reason='Strictly lower defect-aware cost than unchanged-control prediction' if accepted else 'No candidate improves unchanged-control model prediction; keeping previous controls.'
    report['selection']=dict(status='ACCEPT' if accepted else 'REJECT',alpha=selected_alpha,cost=best_cost,reason=reason)
    print('Final selection', json.dumps(report['selection']))
    if not accepted: print(reason)
    output.mkdir(parents=True,exist_ok=True)
    (output/'diagnostics.json').write_text(json.dumps(report,indent=2))
    np.savez(output/'prepared_candidate.npz', measured_previous=measured, old_controls=old_controls,
             reference=reference, defects=defects, controls=selected_controls,
             defect_prediction=selected_states,dt=dt,Q=Q,Q_f=Q_f,R=R,
             selected_alpha=selected_alpha,accepted=accepted)
    print('STOP: candidate prepared; no actual rollout executed.')
    return report


def execute_prepared(path, output):
    # Explicit opt-in: one rollout only, no automatic ILC loop.
    prepared=np.load(path)
    if not bool(prepared['accepted']):
        print('REJECT: keeping previous controls; no candidate rollout executed.');return
    car=F110();reference=prepared['reference'];controls=prepared['controls'];dt=float(prepared['dt'])
    from fixed_deadline_rollout import rollout as timed_rollout
    class PublicStampedNode(F1tenth_ILC):
        def __init__(self):
            self.audit_lock=threading.Lock();super().__init__()
        def pose_callback(self,msg):
            with self.audit_lock:
                super().pose_callback(msg)
                self.audit_odom_stamp=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
                self.audit_receive_time=time.monotonic()
    rclpy.init();node=PublicStampedNode()
    try:
        if not reset_and_wait(node,reference[0]):raise RuntimeError('Reset failed')
        measured=timed_rollout(node,controls,reference[0],dt,car['gear_ratio'],car['pole_pairs'],car['lambda'],car['mass'],car['rw'],car['max_steer'])
        old=prepared['measured_previous'];old_u=prepared['old_controls'];Q,Qf,R=prepared['Q'],prepared['Q_f'],prepared['R']
        old_cost=trajectory_cost(old,old_u,reference,Q,Qf,R);new_cost=trajectory_cost(measured,controls,reference,Q,Qf,R)
        old_rmse=trajectory_rmse(reference,old)[0];new_rmse=trajectory_rmse(reference,measured)[0]
        report=dict(previous_measured_cost=old_cost,new_measured_cost=new_cost,actual_measured_cost_improvement=old_cost-new_cost,previous_measured_position_rmse=float(old_rmse),new_measured_position_rmse=float(new_rmse),actual_measured_rmse_improvement=float(old_rmse-new_rmse))
        output.mkdir(parents=True,exist_ok=True);(output/'actual_metrics.json').write_text(json.dumps(report,indent=2))
        np.savez(output/'actual_history.npz',reference=reference,states=np.stack([old,measured]),controls=np.stack([old_u,controls]),dt=dt,config=json.dumps(dict(Q=Q.tolist(),Q_f=Qf.tolist(),R=R.tolist())))
        print('After actual rollout',json.dumps(report))
    finally:
        for _ in range(10):node.publish_control(0.,0.);rclpy.spin_once(node,timeout_sec=.01)
        node.destroy_node();rclpy.shutdown()


def main():
    parser=argparse.ArgumentParser(description='Independent nine-state defect-aware experiment; default is OFFLINE preparation only.')
    parser.add_argument('--history',type=Path,default=Path('references/s_curve_epoch0.npz'))
    parser.add_argument('--epoch',type=int,default=0)
    parser.add_argument('--output',type=Path,default=Path('results/preflight'))
    parser.add_argument('--alphas',type=float,nargs='+',default=[.03,.015,.0075,.00375,.001875])
    parser.add_argument('--execute-prepared',type=Path,help='Explicitly execute ONE previously prepared candidate; not used during preflight.')
    args=parser.parse_args()
    if args.execute_prepared:
        execute_prepared(args.execute_prepared,args.output);return
    h=np.load(args.history);config=json.loads(str(h['config']))
    # Preserve stored effective weights, including the pre-existing Q_f integer truncation.
    if all(key in config for key in ['Q','Q_f','R']):
        Q=np.asarray(config['Q']);Qf=np.asarray(config['Q_f']);R=np.asarray(config['R'])
    else:
        Q=np.diag([config.get('Q_x',10),config.get('Q_y',10),5,2,.5,.5,0,config['Q_I'],config['Q_delta']])
        Qf=np.diag([config['Qf_position'],config['Qf_position'],20,5,1,1,0,int(config['Qf_I']),config['Qf_delta']]);R=np.diag(config['R'])
    car=F110()
    prepare_update(h['states'][args.epoch],h['controls'][args.epoch],h['reference'],float(h['dt']),car,_get_tire_params(car),Q,Qf,R,args.alphas,args.output)


if __name__ == '__main__':
    main()
