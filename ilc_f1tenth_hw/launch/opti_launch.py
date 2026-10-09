# The car's state estimation with OptiTrack (ported from LLA-Control/LLA-MPC-onboard llampc/launch/opti_launch.py):
# the f1tenth_system stack (patched: scripts/apply_hw_patches.sh), the natnet_ros2 client, the VESC IMU cleanup + ZUPT,
# optitrack_node (NatNet pose -> /optitrack/odom) and the robot_localization EKF (config/mocap.yaml) -> /odometry/filtered
# in the cg frame.
#   ros2 launch ilc_f1tenth_hw opti_launch.py server_ip:=<Motive PC> client_ip:=<this car> mocap_topic:=/f1tenth/pose
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    f1tenth_launch = os.path.join(get_package_share_directory('f1tenth_stack'), 'launch', 'bringup_launch.py')
    natnet_launch = os.path.join(get_package_share_directory('natnet_ros2'), 'launch', 'natnet_ros2.launch.py')
    ekf_config = os.path.join(get_package_share_directory('ilc_f1tenth_hw'), 'config', 'mocap.yaml')
    args = [
        DeclareLaunchArgument('server_ip', default_value='172.26.119.139', description='the Motive (OptiTrack) PC'),
        DeclareLaunchArgument('client_ip', default_value='172.26.112.71', description='this car'),
        DeclareLaunchArgument('mocap_topic', default_value='/f1tenth/pose',
                              description='NatNet rigid-body pose topic: /<rigid body name in Motive>/pose'),
        DeclareLaunchArgument('yaw_offset', default_value='-1.5707963267948966',
                              description='mocap -> ROS yaw offset (rad), found per mocap space'),
    ]
    natnet = IncludeLaunchDescription(PythonLaunchDescriptionSource(natnet_launch), launch_arguments={
        'serverIP': LaunchConfiguration('server_ip'), 'clientIP': LaunchConfiguration('client_ip'),
        'serverType': 'unicast', 'pub_rigid_body': 'true'}.items())
    f1tenth_stack = IncludeLaunchDescription(PythonLaunchDescriptionSource(f1tenth_launch))
    zupt = Node(package='ilc_f1tenth_hw', executable='imu_zupt_prep.py', name='zupt_node', output='screen')
    ekf = Node(package='robot_localization', executable='ekf_node', name='ekf_filter_node', output='screen',
               parameters=[ekf_config], remappings=[('/set_pose', '/initialpose')])
    optitrack = Node(package='ilc_f1tenth_hw', executable='optitrack_node.py', name='optitrack_subscriber',
                     output='screen', parameters=[{'mocap_topic': LaunchConfiguration('mocap_topic'),
                                                   'yaw_offset': LaunchConfiguration('yaw_offset')}])
    return LaunchDescription(args + [natnet, f1tenth_stack, zupt, ekf, optitrack])
