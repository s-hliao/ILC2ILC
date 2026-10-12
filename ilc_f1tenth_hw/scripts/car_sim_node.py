#!/usr/bin/env python3
"""
car_sim_node: a desk stand-in for the real car + its EKF, to test net_runner.py over ROS 2 without hardware. Steps a
simulated multi-body car (ilc2real mb_car: mc.car_config(<car>), mc.step at DT = 0.025 s in real time) with the latest
/mpc/drive command and publishes what the car's stack would:
    subscribes  /mpc/drive          AckermannDriveStamped: jerk 2.0 -> motor current = drive.acceleration (A) and
                                    drive.steering_angle; jerk 1.0 (duty cycle 0, net_runner's brake) -> a braking current
                                    against the wheel speed; the car's own command delays / steering offset apply
    publishes   /odometry/filtered  nav_msgs/Odometry: pose in `map` (X, Y, psi), twist in the body frame (vx, vy, r)
                /sensors/core       vesc_msgs/VescStateStamped: speed = ERPM (omega_w x gear x pole pairs), avg_iq = I
The car starts rolling at 0.6 m/s at the plan's start (the multi-body car is kinematic below 0.5 m/s: no driveline force,
so a true standstill cannot launch -- real_car.run's slow-start launch does the same).
    ros2 run ilc_f1tenth_hw car_sim_node.py --ros-args -p car:=real_nom -p plan:=mocap_square2fast_b25
"""
import math
import os
import sys

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

ILC2REAL = os.environ.get('ILC2REAL_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                                                          'ilc_f1tenth', 'ilc2real')
sys.path.insert(0, ILC2REAL)
import bank                                      # noqa: E402
import car_model as cm                           # noqa: E402
import mb_car as mc                              # noqa: E402
from vesc_msgs.msg import VescStateStamped       # noqa: E402

DT = 0.025


class CarSim(Node):
    def __init__(self):
        super().__init__('car_sim')
        p = self.declare_parameter
        p('car', 'real_nom')
        p('plan', 'mocap_square2fast_b25')
        p('start_offset', 0.0)                  # m, lateral offset of the start from the plan (envelope tests)
        p('pole_pairs', 2.0)
        p('gear_ratio', 11.82)
        p('brake_gain', 2.0)
        g = lambda n: self.get_parameter(n).value
        self.c = mc.car_config(g('car'))
        plan = bank.load_plan(g('plan'))
        xs = np.asarray(plan['x'][0], float).copy()
        off = float(g('start_offset'))
        xs[0] += -math.sin(xs[2]) * off
        xs[1] += math.cos(xs[2]) * off
        xs[3:9] = 0.0
        xs[3] = 0.6
        xs[6] = xs[3] / cm.NOMINAL['rw']
        self.xm = mc.mb_from_planar(xs, self.c['p'])
        mc.step(np.array(self.xm, copy=True), 0.0, 0.0, self.c, DT)   # compile the plant (numba) before the clock runs
        self.pp, self.gear, self.kb = float(g('pole_pairs')), float(g('gear_ratio')), float(g('brake_gain'))
        self.cmd = np.zeros(2)                   # (steering, current) as last commanded
        self.brake = False
        self.moving = False                      # held at the start (the operator's placement) until the first command
        self.hist = [np.zeros(2)] * 8
        self.create_subscription(AckermannDriveStamped, '/mpc/drive', self.on_drive, 10)
        self.pub_odom = self.create_publisher(Odometry, '/odometry/filtered', 10)
        self.pub_vesc = self.create_publisher(VescStateStamped, '/sensors/core', 10)
        self.create_timer(DT, self.tick)
        self.get_logger().info(f"car_sim: {g('car')} on {g('plan')}, start offset {off:.2f} m")

    def on_drive(self, m):
        self.moving = True
        if m.drive.jerk == 2.0:
            self.cmd = np.array([m.drive.steering_angle, m.drive.acceleration])
            self.brake = False
        elif m.drive.jerk == 1.0 and m.drive.acceleration == 0.0:
            self.cmd = np.array([m.drive.steering_angle, 0.0])
            self.brake = True

    def tick(self):
        if not self.moving:
            self.publish(mc.planar(self.xm))
            return
        xt = mc.planar(self.xm)
        cmd = self.cmd.copy()
        if self.brake:                           # duty 0: the VESC brakes -- a current against the wheel speed
            lo = float(cm.A_OFF[1] - cm.A_SCALE[1])
            cmd[1] = max(lo, -self.kb * xt[6]) if xt[3] > 0.05 else 0.0
        self.hist.append(cmd)
        self.hist.pop(0)
        d_cmd = self.hist[-1 - int(self.c['d_delay'])][0] + self.c['d_off']
        i_cmd = self.hist[-1 - int(self.c['i_delay'])][1]
        try:
            self.xm = mc.step(self.xm, d_cmd, i_cmd, self.c, DT)
        except (ZeroDivisionError, FloatingPointError):
            self.get_logger().error('car_sim: numerics failure')
            return
        self.publish(mc.planar(self.xm))

    def publish(self, x):
        now = self.get_clock().now().to_msg()
        o = Odometry()
        o.header.stamp, o.header.frame_id, o.child_frame_id = now, 'map', 'cg'
        o.pose.pose.position.x, o.pose.pose.position.y = float(x[0]), float(x[1])
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(x[2] / 2.0), math.cos(x[2] / 2.0)
        o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.angular.z = float(x[3]), float(x[4]), float(x[5])
        self.pub_odom.publish(o)
        v = VescStateStamped()
        v.header.stamp = now
        v.state.speed = float(x[6] * self.gear * self.pp * 60.0 / (2.0 * math.pi))
        v.state.avg_iq = float(x[7])
        v.state.current_motor = float(x[7])
        self.pub_vesc.publish(v)


def main():
    rclpy.init()
    node = CarSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
