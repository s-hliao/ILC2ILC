"""Bring up the MuJoCo simulation.

    ros2 launch ilc_quad sim.launch.py robot:=go2 viewer:=true

This starts the simulator alone. It waits on `joint_torque_cmd` and holds zero
torque until something publishes, so the robot will sit on the floor until your
controller connects -- that is the expected idle state, not a fault.

`use_sim_time` is forced False here: this node publishes /clock, and a timer
waiting on a clock that only that timer advances never fires. Your controller
should run with use_sim_time:=True so it reads the simulator's clock.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot = LaunchConfiguration("robot")
    menagerie_root = LaunchConfiguration("menagerie_root")

    # A LaunchConfiguration substitutes to a string and the node declares these as
    # bool/float, so without an explicit value_type the parameter arrives as "1.0"
    # and the declaration rejects it.
    viewer = ParameterValue(LaunchConfiguration("viewer"), value_type=bool)
    realtime_factor = ParameterValue(
        LaunchConfiguration("realtime_factor"), value_type=float
    )
    control_rate_hz = ParameterValue(
        LaunchConfiguration("control_rate_hz"), value_type=float
    )

    params = PathJoinSubstitution(
        [FindPackageShare("ilc_quad"), "config", [robot, ".yaml"]]
    )

    return LaunchDescription([
        DeclareLaunchArgument("robot", default_value="go2",
                              choices=["go2", "go1"],
                              description="which Unitree to simulate"),
        DeclareLaunchArgument("viewer", default_value="false",
                              description="open the MuJoCo viewer"),
        DeclareLaunchArgument("menagerie_root", default_value="/mujoco_menagerie",
                              description="path to a mujoco_menagerie checkout"),
        DeclareLaunchArgument("realtime_factor", default_value="1.0",
                              description="sim pacing; 0 runs as fast as it can"),
        DeclareLaunchArgument("control_rate_hz", default_value="250.0",
                              description="state publish rate; must divide the "
                                          "2 ms physics timestep evenly"),

        Node(
            package="ilc_quad",
            executable="sim_node",
            name="mujoco_sim",
            output="screen",
            emulate_tty=True,
            parameters=[params, {
                "robot": robot,
                "menagerie_root": menagerie_root,
                "viewer": viewer,
                "realtime_factor": realtime_factor,
                "control_rate_hz": control_rate_hz,
                "use_sim_time": False,
            }],
        ),
    ])
