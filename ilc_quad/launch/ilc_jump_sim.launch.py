"""ILC jumping in MuJoCo: the simulator and the ILC node together.

    ros2 launch ilc_quad ilc_jump_sim.launch.py viewer:=true
    ros2 launch ilc_quad ilc_jump_sim.launch.py box:=false jump_dx:=0.3 jump_dz:=0.0
    ros2 launch ilc_quad ilc_jump_sim.launch.py box:=false jump_dx:=0.6 jump_dz:=0.0 \
        ground_kp:=2e3 ground_kd:=5e2          # the paper's soft ground (Fig. 4)

The box is given once and goes to both nodes: sim_node puts it in the scene and the
ILC node plans the jump over it. Its x_front is measured from the robot's standing
CoM, which the home keyframe puts ~2 mm behind world x = 0 -- close enough that the
one number serves both.

The sim runs at 500 Hz so the 10 ms TO grid is a whole number of control ticks.
The TO takes tens of seconds on the first run; with reference_file set it is saved
there and reused (a changed task re-solves only if you point at a new file).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

ARGS = [
    ("robot", "go2", "robot model (the Go2 hardware path is go2 only)"),
    ("viewer", "false", "open the MuJoCo viewer"),
    ("menagerie_root", "/mujoco_menagerie", "path to a mujoco_menagerie checkout"),
    ("realtime_factor", "1.0", "sim pacing; the ILC node answers every state, so keep ~1"),
    ("box", "true", "put a box in the scene and plan the jump onto it"),
    ("box_x_front", "0.25", "box front face, m ahead of the standing CoM"),
    ("box_height", "0.10", "box height, m"),
    ("jump_dx", "0.50", "CoM displacement forward, m"),
    ("jump_dz", "0.10", "CoM displacement up, m (the box height to land on it)"),
    ("max_trials", "25", "stop after this many trials"),
    ("reference_file", "", "npz to load the TO from, or save it to"),
    ("log_dir", "", "root folder for run folders (one npz per trial)"),
    ("run_name", "", "run folder under log_dir (default: a timestamp)"),
    ("resume_from", "", "trial npz or run folder to continue learning from"),
    ("transfer_from", "", "another task's trial npz or run folder to start from"),
    ("transfer_mode", "retarget", "retarget (this task's plan + learned correction) or paper"),
    ("feedforward", "to_torque", "to_torque (TO torque + J^T dU) or force (J^T U only)"),
    ("margin", "0.8", "share of each hard limit the TO may use"),
    ("qu", "3e-5", "ILC step penalty Qu, Stages I-II"),
    ("qu_stage3", "3e-4", "ILC step penalty Qu, Stage III"),
    ("ground_kp", "0.0", "floor stiffness per foot, N/m (0: the MJCF's contact); unknown to the ILC"),
    ("ground_kd", "0.0", "floor damping per foot, N s/m"),
    ("payload_mass", "0.0", "kg welded on the trunk, unknown to the ILC"),
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
        return Node(package="ilc_quad", executable="ilc_jump_sim.py", name="ilc_jump",
                    output="screen", emulate_tty=True, condition=condition,
                    parameters=[dict(ilc, box_height=height)], on_exit=Shutdown())

    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGS]
        + [
            sim(with_box, _float("box_height")), controller(with_box, _float("box_height")),
            sim(no_box, 0.0), controller(no_box, 0.0),
        ]
    )
