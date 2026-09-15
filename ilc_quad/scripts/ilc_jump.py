#!/usr/bin/env python3
import os

import mujoco
import numpy as np

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger


class MujocoSimNode(Node):
    def __init__(self):
        super().__init__("ilc_jump")


        self.declare_parameter("control_rate_hz", 250.0)
        self.declare_parameter("seed_trajectory", None)

        self.sequential = True
        self.control_rate = self.get_parameter("control_rate_hz").value
        self.dt = 1.0/self.control_rate


        self.create_timer(self.control_callback, self.dt)
        self.initialize_seed()



        self.started = False

    def initialize_seed(self):
        self.seed_traj = self.get_parameter("seed_trajectory").value
        if(not self.seed_traj is None):
            self.seed_traj =  np.load(self.seed_traj)
            self.last_traj_states = self.seed_traj["states"]
            self.last_traj_controls = self.seed_traj["controls"]
            


    def control_callback(self):
        if(not self.started):
            return



    



