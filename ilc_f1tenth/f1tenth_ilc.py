import rclpy
import time
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
import casadi as ca
from params import F110
from tf_transformations import euler_from_quaternion
from trajectory_generator import generate_s_curve
from dynamics import discrete_dynamics_model
from iLQR import backward, forward
from trajectory_analysis import compare_trajectories, plot_trials, save_trials
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

        self.first_6_state = initial_state[0:6].copy()


    def publish_control(self, steer, v):
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steer
        msg.drive.speed = v
        self.pub.publish(msg)



def _get_tire_params(p_car):
    """
    Pull [Cf, Cr, muf, mur, Cro] to match the unpacking order used in
    dynamics: `Cf, Cr, muf, mur, Cro = [p[i] for i in range(5)]`.
    """
    def _get(key, default):
        try:
            return p_car[key]
        except (KeyError, TypeError):
            return default

    Cf  = _get('Cf', 5.0)
    Cr  = _get('Cr', 5.0)
    muf = _get('muf', 1.0)
    mur = _get('mur', 1.0)
    Cro = _get('Cro', 0.01)

    return np.array([Cf, Cr, muf, mur, Cro])

def rollout(node, new_controls, initial_state, dt, gear_ratio, pole_pairs, lam, mass, rw):
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
        steer = steer + steer_rate * dt

        old_count = node.odom_count
        node.publish_control(steer, velocity)
        step_start = time.time()

        # keep ROS spinning during dt
        while time.time() - step_start < dt:
            remaining = dt - (time.time() - step_start)

            rclpy.spin_once(
                node,
                timeout_sec=min(0.005, remaining)
            )

        # make sure we actually got a new odom
        if node.odom_count == old_count:
            print(f"WARNING: no new odom at step {t}")

    state = np.zeros(9)
    state[0:6] = node.first_6_state
    state[6] = node.first_6_state[3]/rw
    state[7] = I
    state[8] = steer
    new_trajectory.append(state)

    return np.array(new_trajectory)



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

    #simulation parameters
    dt = 0.05
    exact = False

    #ILC parameters
    epochs = 1

    #ILQR parameters
    alpha = 0.03

    rclpy.init()
    node = F1tenth_ILC()

    #reference trajectory
    ref_trajectory = generate_s_curve(
        params_car=params_car,
        path_length=8.0,
        vx_ref=3.0,
        dt=dt,
        kappa_max=0.20,
        hard = True
    )

    num_state = ref_trajectory.shape[0]

    #initial controls and trajectory
    cur_controls = np.zeros((num_state - 1, 2))

    # longitudinal initial guess
    accel_time = 1.0
    a_ref = 3.0 / accel_time

    I_accel = (
        a_ref * mass * rw
        / (gear_ratio * 1.5 * pole_pairs * lam)
    )

    accel_steps = int(accel_time / dt)

    cur_controls[0, 0] = I_accel / dt

    if accel_steps < cur_controls.shape[0]:
        cur_controls[accel_steps, 0] = -I_accel / dt

    # steering initial guess
    cur_controls[:, 1] = (
        ref_trajectory[1:, 8]
        - ref_trajectory[:-1, 8]
    ) / dt

    cur_trajectory = np.zeros([num_state, 9])
    initial_state = ref_trajectory[0].copy()
    node.reset_car(initial_state)

    #initial rollout
    cur_trajectory = rollout(node, cur_controls, initial_state, dt, gear_ratio, pole_pairs, lam, mass, rw)
    trials = [cur_trajectory.copy()]           # trial 0: the initial rollout
    trial_controls = [cur_controls.copy()]

    for i in tqdm(range(epochs), desc="ILC epochs"):
        #reset initial pose and control
        node.publish_control(0.0, 0.0)
        node.reset_car(initial_state)

        new_trajectory = []

        #backward
        k, K, cur_cost = backward(
            ref_trajectory,
            cur_trajectory,
            cur_controls,
            tire_params,
            params_car,
            exact,
            dt
        )

        #forward
        new_controls, _, _ = forward(
            k,
            K,
            ref_trajectory,
            cur_trajectory,
            cur_controls,
            tire_params,
            params_car,
            exact,
            dt,
            alpha
        )

        #apply the new controls in f1tenth and get a new trajectory
        new_trajectory = rollout(node, new_controls, initial_state, dt, gear_ratio, pole_pairs, lam, mass, rw)

        #then update cur_trajectory and cur_control
        cur_controls = new_controls
        cur_trajectory = new_trajectory
        trials.append(cur_trajectory.copy())
        trial_controls.append(cur_controls.copy())

    run = save_trials("f1tenth_ilc", ref_trajectory, trials, trial_controls, dt)   # ~/ilc_ws/log/f1tenth
    plot_trials(ref_trajectory, trials, dt, out=run.replace(".npz", ".png"))
    compare_trajectories(ref_trajectory, cur_trajectory, dt)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
