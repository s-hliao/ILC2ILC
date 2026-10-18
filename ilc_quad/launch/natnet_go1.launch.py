"""Run the Go1's NatNet receiver and 500 Hz state estimator on the workstation."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("ilc_quad")
    natnet_launch = os.path.join(
        get_package_share_directory("natnet_ros2"), "launch", "natnet_ros2.launch.py"
    )
    return LaunchDescription([
        DeclareLaunchArgument("server_ip", default_value="172.26.30.22",
                              description="Motive/OptiTrack computer"),
        DeclareLaunchArgument("client_ip", description="this workstation's mocap-network address"),
        DeclareLaunchArgument("mocap_topic", default_value="/dl_go1/pose",
                              description="NatNet PoseStamped topic for the Go1 rigid body"),
        DeclareLaunchArgument("up_axis", default_value="z"),
        DeclareLaunchArgument("yaw_offset", default_value="0.0"),
        DeclareLaunchArgument("mocap_offset", default_value="[0.0, 0.0, 0.0]",
                              description="rigid-body marker origin expressed in the trunk frame"),
        DeclareLaunchArgument("pose_latency", default_value="0.006"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(natnet_launch),
            launch_arguments={
                "serverIP": LaunchConfiguration("server_ip"),
                "clientIP": LaunchConfiguration("client_ip"),
                "serverType": "unicast",
                "pub_rigid_body": "true",
            }.items(),
        ),
        Node(
            package="ilc_quad", executable="mocap_odom_ros2.py", name="mocap_odom",
            output="screen",
            parameters=[{
                "mocap_topic": LaunchConfiguration("mocap_topic"),
                "up_axis": LaunchConfiguration("up_axis"),
                "yaw_offset": ParameterValue(LaunchConfiguration("yaw_offset"), value_type=float),
                # Leave array conversion to launch_ros' YAML parameter parser.
                "mocap_offset": LaunchConfiguration("mocap_offset"),
                "pose_latency": ParameterValue(LaunchConfiguration("pose_latency"), value_type=float),
            }],
        ),
        Node(
            package="ilc_quad", executable="go1_sensor_restamp.py",
            name="go1_sensor_restamp", output="screen",
        ),
        Node(
            package="robot_localization", executable="ekf_node", name="ekf_trunk",
            output="screen", parameters=[os.path.join(share, "config", "go1_ekf.yaml")],
            remappings=[("odometry/filtered", "/trunk/odometry")],
        ),
    ])
