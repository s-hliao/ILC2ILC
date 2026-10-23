"""Operator-gated Go1 stand test using the simulation's stance perturbations."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("kp", default_value="60.0", description="final stand position gain"),
        DeclareLaunchArgument("kd", default_value="4.0", description="stand damping gain; simulation used 5"),
        DeclareLaunchArgument("capture_time", default_value="1.0"),
        DeclareLaunchArgument("preload_time", default_value="2.5"),
        DeclareLaunchArgument("rise_time", default_value="4.0"),
        DeclareLaunchArgument("perturb_ramp_time", default_value="2.0"),
        DeclareLaunchArgument("state_timeout", default_value="0.30",
                              description="joint-state bridge jitter allowance"),
        DeclareLaunchArgument("hold_state_timeout", default_value="1.0",
                              description="joint-state outage tolerated while ramping/holding before damping"),
        DeclareLaunchArgument("pose_timeout", default_value="0.20"),
        DeclareLaunchArgument("pose_topic", default_value="/mocap/pose"),
        Node(
            package="ilc_quad", executable="go1_stand_perturb.py", name="go1_stand_perturb",
            output="screen", emulate_tty=True,
            parameters=[{
                "kp": ParameterValue(LaunchConfiguration("kp"), value_type=float),
                "kd": ParameterValue(LaunchConfiguration("kd"), value_type=float),
                "capture_time": ParameterValue(LaunchConfiguration("capture_time"), value_type=float),
                "preload_time": ParameterValue(LaunchConfiguration("preload_time"), value_type=float),
                "rise_time": ParameterValue(LaunchConfiguration("rise_time"), value_type=float),
                "perturb_ramp_time": ParameterValue(LaunchConfiguration("perturb_ramp_time"), value_type=float),
                "state_timeout": ParameterValue(LaunchConfiguration("state_timeout"), value_type=float),
                "hold_state_timeout": ParameterValue(LaunchConfiguration("hold_state_timeout"), value_type=float),
                "pose_timeout": ParameterValue(LaunchConfiguration("pose_timeout"), value_type=float),
                "pose_topic": LaunchConfiguration("pose_topic"),
            }],
        ),
    ])
