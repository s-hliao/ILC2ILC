import rclpy
from rclpy.node import Node
import casadi as ca
from params import F110
from tf_transformations import euler_from_quaternion
from trajectory_generator import generate_s_curve
from dynamics import discrete_dynamics_model
from iLQR import backward, alpha_search
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from trajectory_generator import generate_s_curve


class F1tenth_ILC():
    
    def __init__(Node):
        super.__init__("F1tenth_ILC_Node")

        self.first_6_state = None
        
        self.pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
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
        # current, delta(steer rate)

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        orientation = msg.pose.pose.orientation
        quaternion = [
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ]
        _, _, phi = euler_from_quaternion(quaternions)

        v_x = msg.twist.twist.linear.x
        v_y = msg.twist.twist.linear.y
        omega = msg.twist.twist.angular.z

        self.first_6_state = np.array([x, y , phi, v_x, v_y, omega])

    def publish_control(steer, v):
        msg = AckermannDriveStamped()
        msg.drive.steer_angle = steer
        msg.drive.speed = v


def main():
     #simulation parameters
    dt = 0.05
    exact = False

    #ILC parameters
    epochs = 100

    #car parameters
    params_car = F110()
    tire_params = _get_tire_params(params_car)

    rclpy.init()
    node = F1tenth_ILC_Node()

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
    cur_trajectory = np.zeros([state_num, 9])


    for i in range(epochs):

        new_trajectory = []

        #backward pass
        k, K, cur_cost = backward(
            ref_trajectory,
            cur_trajectory,
            cur_controls,
            tire_params,
            params_car,
            exact,
            dt
        )

        #forward pass with alpha search
        alpha, new_controls, predicted_trajectory, predicted_cost = alpha_search(
            k,
            K,
            ref_trajectory,
            cur_trajectory,
            cur_controls,
            cur_cost,
            tire_params,
            params_car,
            exact,
            dt
        )

        if alpha == 0.0:
            print("No alpha reduces the cost.")
            break

        #apply the new controls in f1tenth and get a new trajectory
        #temporary: use the nonlinear dynamics model as the plant

        '''publish drive and get rollout here'''
        '''need to rethink this and adjust'''
        steer = 0
        speed = 0
        state = np.zeros(9)

        delta_current = new_controls[0] * dt
        tau_drive = gear_ratio * 1.5 * pole_pairs * lam * current
        acceleration = tau_drive / (mass * rw)
        delta_speed = vx + acceleration * dt
        delta_steer = new_controls[1] * dt

        rclpy.spin(node)
        # publish control and record state
        for i in range(new_controls):
            steer = steer + delta_steer[i][1]
            speed = speed + delta_speed[i][0]
            node.publish_control(steer, speed)
            state[0:7] = node.first_6_state
            #need to add the last 3 state
            new_trajectory.append(state)

        #then update cur_trajectory and cur_control
        cur_controls = new_controls
        cur_trajectory = np.array(new_trajectory)


    





        
