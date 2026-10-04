"""The Go1 bridge alone (unitree_legged_sdk), kept up for a whole hardware session while hw_session.py launches the
policy node goal by goal (policy_jump_go1.launch.py with_bridge:=false). The bridge damps whenever no node commands.

    ros2 launch ilc_quad go1_bridge.launch.py sdk_path:=...
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ARGS = [
    ("sdk_path", "", "directory holding unitree_legged_sdk's robot_interface*.so"),
    ("power_level", "10", "the SDK's PowerProtect level, 1..10 (10 = full power, needed to jump)"),
    ("menagerie_root", "/mujoco_menagerie", "mujoco_menagerie checkout (model parameters)"),
]


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
        ]
    )
