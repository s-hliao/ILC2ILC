"""The learned force policy jumping on a real Go1 (deep ILC's hardware stage): the unitree_legged_sdk bridge and
policy_jump_node (IlcJumpNode flying the policy, no learning; each jump saved as an episode for deploy.py).

    python3 policy_jump_node.py prepare --policy DIR --goal 0.5 0 --out /tmp/ref.npz
    ros2 launch ilc_quad policy_jump_go1.launch.py pose_topic:=/optitrack/go1/pose policy_dir:=DIR \
        reference_file:=/tmp/ref.npz jump_dx:=0.5 jump_dz:=0.0 episode_dir:=DIR/../real episode_tag:=go1
    ros2 service call /start_trial std_srvs/srv/Trigger      # one jump
    ros2 service call /damp std_srvs/srv/Trigger             # any time: go limp

Flat ground only (box_height 0). The same safety layers as ilc_jump_go1.launch.py; see its notes before launching
(low-level mode, OptiTrack, a hand on /damp). One goal per launch: relaunch for the next goal.
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
    ("box_height", "0.0", "flat ground (the policy jumps have no box)"),
    ("jump_dx", "0.50", "CoM displacement forward, m"),
    ("jump_dz", "0.0", "CoM displacement up, m"),
    ("max_trials", "1000", "jumps per launch (the operator decides)"),
    ("reference_file", "", "npz to load the TO from, or save it to"),
    ("log_dir", "", "directory for one npz per trial"),
    ("run_name", "", "run folder under log_dir (default: a timestamp)"),
    ("resume_from", "", "trial npz or run folder to continue learning from"),
    ("transfer_from", "", "another task's trial npz or run folder to start from"),
    ("transfer_mode", "retarget", "retarget (this task's plan + learned correction) or paper"),
    ("margin", "0.85", "share of each hard limit the TO may use (Go1 validation: 0.85 for flat and 10 cm boxes, 0.9-1.0 for taller)"),
    ("pose_latency", "0.006", "mocap latency, s (MEASURE it: the policy's state estimator dates every frame by it)"),
    ("mocap_offset", "[0.0, 0.0, 0.0]", "the tracked point in the base frame, m"),
    ("policy_dir", "", "the policy folder deploy.py exported (required)"),
    ("policy_member", "policy", "its member"),
    ("episode_dir", "", "where each jump's episode is saved (required)"),
    ("episode_tag", "go1", "episode file tag (the robot)"),
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
            Node(package="ilc_quad", executable="policy_jump_node.py", name="ilc_jump",
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
                     "pose_latency": _float("pose_latency"),
                     "mocap_offset": ParameterValue(LaunchConfiguration("mocap_offset"), value_type=None),
                     "policy_dir": LaunchConfiguration("policy_dir"),
                     "policy_member": LaunchConfiguration("policy_member"),
                     "episode_dir": LaunchConfiguration("episode_dir"),
                     "episode_tag": LaunchConfiguration("episode_tag"),
                     "auto_start": False,
                     "use_sim_time": False,
                 }]),
        ]
    )
