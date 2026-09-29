"""ILC jumping on a real Go1: the unitree_legged_sdk bridge and the ILC node.

    ros2 launch ilc_quad ilc_jump_go1.launch.py pose_topic:=/optitrack/go1/pose pose_type:=pose
    ros2 service call /start_trial std_srvs/srv/Trigger      # once per trial
    ros2 service call /damp std_srvs/srv/Trigger             # any time: go limp

Before launching:
  * unitree_legged_sdk v3.8 built with its Python wrapper, `sdk_path` pointing at
    the directory holding robot_interface*.so (lib/python/amd64 on a PC), and the
    workstation on the robot's 192.168.123.x network.
  * The robot in low-level mode: L2+A (lie down), L2+B (damp), L1+L2+Start.
  * Your OptiTrack node publishing the trunk pose on `pose_topic` (PoseStamped or
    Odometry, `pose_type`) in a z-up world frame, the pose of the base frame -- or
    set `mocap_offset` to where the tracked point sits in the base frame.
  * The robot standing clear, the box (if any) `box_x_front` m ahead of the standing
    CoM along the robot's heading, and a hand on the damp command.

Nothing moves until /start_trial: the bridge damps the joints until the ILC node
commands, and the ILC node stands the robot up at the start of each trial.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ARGS = [
    ("pose_topic", "", "OptiTrack trunk pose topic (required)"),
    ("sdk_path", "", "directory holding unitree_legged_sdk's robot_interface*.so"),
    ("power_level", "10", "the SDK's PowerProtect level, 1..10 (10 = full power, needed to jump)"),
    ("pose_type", "pose", "pose (geometry_msgs/PoseStamped) or odometry (nav_msgs/Odometry)"),
    ("menagerie_root", "/mujoco_menagerie", "mujoco_menagerie checkout (model parameters)"),
    ("box_x_front", "0.25", "box front face, m ahead of the standing CoM"),
    ("box_height", "0.10", "box height, m; 0 for no box"),
    ("jump_dx", "0.50", "CoM displacement forward, m"),
    ("jump_dz", "0.10", "CoM displacement up, m"),
    ("max_trials", "25", "stop after this many trials"),
    ("reference_file", "", "npz to load the TO from, or save it to"),
    ("log_dir", "", "directory for one npz per trial"),
    ("run_name", "", "run folder under log_dir (default: a timestamp)"),
    ("resume_from", "", "trial npz or run folder to continue learning from"),
    ("transfer_from", "", "another task's trial npz or run folder to start from"),
    ("transfer_mode", "retarget", "retarget (this task's plan + learned correction) or paper"),
    ("margin", "0.85", "share of each hard limit the TO may use (Go1 validation: 0.85 for flat and 10 cm boxes, 0.9-1.0 for taller)"),
]


def _float(name):
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def generate_launch_description():
    root = LaunchConfiguration("menagerie_root")
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGS]
        + [
            Node(package="ilc_quad", executable="go1_bridge.py", name="go1_bridge",
                 output="screen", emulate_tty=True,
                 parameters=[{"menagerie_root": root, "rate_hz": 500.0,
                              "sdk_path": LaunchConfiguration("sdk_path"),
                              "power_level": ParameterValue(LaunchConfiguration("power_level"),
                                                            value_type=int)}]),
            Node(package="ilc_quad", executable="ilc_jump_sim.py", name="ilc_jump",
                 output="screen", emulate_tty=True,
                 parameters=[{
                     "backend": "go1",
                     "robot": "go1",
                     "menagerie_root": root,
                     "control_rate_hz": 500.0,
                     "pose_topic": LaunchConfiguration("pose_topic"),
                     "pose_type": LaunchConfiguration("pose_type"),
                     "jump_dx": _float("jump_dx"),
                     "jump_dz": _float("jump_dz"),
                     "box_x_front": _float("box_x_front"),
                     "box_height": _float("box_height"),
                     "max_trials": ParameterValue(LaunchConfiguration("max_trials"), value_type=int),
                     "reference_file": LaunchConfiguration("reference_file"),
                     "log_dir": LaunchConfiguration("log_dir"),
                     "run_name": LaunchConfiguration("run_name"),
                     "resume_from": LaunchConfiguration("resume_from"),
                     "transfer_from": LaunchConfiguration("transfer_from"),
                     "transfer_mode": LaunchConfiguration("transfer_mode"),
                     "margin": _float("margin"),
                     "auto_start": False,
                     "use_sim_time": False,
                 }]),
        ]
    )
