"""policy_jump_go1.launch.py's twin in MuJoCo (sim_node in place of the Go1 bridge, the same topics): the hardware
stage's executor tested end to end without the robot. Trials start on their own; the launch ends after max_trials.

    ros2 launch ilc_quad policy_jump_sim.launch.py policy_dir:=DIR reference_file:=REF jump_dx:=0.5 \
        episode_dir:=OUT max_trials:=3
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

ARGS = [
    ("robot", "go1", "robot model (the Go2 hardware path is go2 only)"),
    ("viewer", "false", "open the MuJoCo viewer"),
    ("menagerie_root", "/mujoco_menagerie", "path to a mujoco_menagerie checkout"),
    ("realtime_factor", "1.0", "sim pacing; the ILC node answers every state, so keep ~1"),
    ("box", "false", "put a box in the scene and plan the jump onto it"),
    ("box_x_front", "0.25", "box front face, m ahead of the standing CoM"),
    ("box_height", "0.10", "box height, m"),
    ("jump_dx", "0.50", "CoM displacement forward, m"),
    ("jump_dz", "0.0", "CoM displacement up, m"),
    ("max_trials", "25", "stop after this many trials"),
    ("reference_file", "", "npz to load the TO from, or save it to"),
    ("log_dir", "", "root folder for run folders (one npz per trial)"),
    ("run_name", "", "run folder under log_dir (default: a timestamp)"),
    ("resume_from", "", "trial npz or run folder to continue learning from"),
    ("transfer_from", "", "another task's trial npz or run folder to start from"),
    ("transfer_mode", "retarget", "retarget (this task's plan + learned correction) or paper"),
    ("feedforward", "to_torque", "to_torque (TO torque + J^T dU) or force (J^T U only)"),
    ("margin", "0.85", "share of each hard limit the TO may use (Go1 validation: 0.85 for flat and 10 cm boxes, 0.9-1.0 for taller)"),
    ("qu", "1e-4", "ILC step penalty Qu, Stages I-II"),
    ("qu_stage3", "1e-5", "ILC step penalty Qu, Stage III (safeguarded)"),
    ("ground_kp", "0.0", "floor stiffness per foot, N/m (0: the MJCF's contact); unknown to the ILC"),
    ("ground_kd", "0.0", "floor damping per foot, N s/m"),
    ("payload_mass", "0.0", "kg welded on the trunk, unknown to the ILC"),
    ("policy_dir", "", "the policy folder deploy.py exported (required)"),
    ("policy_member", "policy", "its member"),
    ("episode_dir", "", "where each jump's episode is saved (required)"),
    ("episode_tag", "sim", "episode file tag (the robot)"),
]


def _float(name):
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def generate_launch_description():
    robot = LaunchConfiguration("robot")
    sim_params = PathJoinSubstitution([FindPackageShare("ilc_quad"), "config", [robot, ".yaml"]])
    common_sim = {
        "robot": robot,
        "menagerie_root": LaunchConfiguration("menagerie_root"),
        "viewer": ParameterValue(LaunchConfiguration("viewer"), value_type=bool),
        "realtime_factor": _float("realtime_factor"),
        "control_rate_hz": 500.0,
        "box_x_front": _float("box_x_front"),
        "ground_kp": _float("ground_kp"),
        "ground_kd": _float("ground_kd"),
        "payload_mass": _float("payload_mass"),
        "use_sim_time": False,
    }
    ilc = {
        "backend": "sim",
        "robot": robot,
        "menagerie_root": LaunchConfiguration("menagerie_root"),
        "control_rate_hz": 500.0,
        "jump_dx": _float("jump_dx"),
        "jump_dz": _float("jump_dz"),
        "box_x_front": _float("box_x_front"),
        "max_trials": ParameterValue(LaunchConfiguration("max_trials"), value_type=int),
        "reference_file": LaunchConfiguration("reference_file"),
        "log_dir": LaunchConfiguration("log_dir"),
        "run_name": LaunchConfiguration("run_name"),
        "resume_from": LaunchConfiguration("resume_from"),
        "transfer_from": LaunchConfiguration("transfer_from"),
        "transfer_mode": LaunchConfiguration("transfer_mode"),
        "feedforward": LaunchConfiguration("feedforward"),
        "margin": _float("margin"),
        "qu": _float("qu"),
        "qu_stage3": _float("qu_stage3"),
        "exit_when_done": True,
        "policy_dir": LaunchConfiguration("policy_dir"),
        "policy_member": LaunchConfiguration("policy_member"),
        "episode_dir": LaunchConfiguration("episode_dir"),
        "episode_tag": LaunchConfiguration("episode_tag"),
        "use_sim_time": True,
    }
    with_box = IfCondition(LaunchConfiguration("box"))
    no_box = UnlessCondition(LaunchConfiguration("box"))

    def sim(condition, height):
        return Node(package="ilc_quad", executable="sim_node.py", name="mujoco_sim",
                    output="screen", emulate_tty=True, condition=condition,
                    parameters=[sim_params, dict(common_sim, box_height=height)])

    def controller(condition, height):
        # the launch ends with the ILC run (target reached or max_trials)
        return Node(package="ilc_quad", executable="policy_jump_node.py", name="ilc_jump",
                    output="screen", emulate_tty=True, condition=condition,
                    parameters=[dict(ilc, box_height=height)], on_exit=Shutdown())

    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGS]
        + [
            sim(with_box, _float("box_height")), controller(with_box, _float("box_height")),
            sim(no_box, 0.0), controller(no_box, 0.0),
        ]
    )
