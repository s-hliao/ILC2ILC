cmake_minimum_required(VERSION 3.8)
project(ilc_f1tenth_hw)

find_package(ament_cmake REQUIRED)

# the nodes (run with: ros2 run ilc_f1tenth_hw <script>.py); a new script must be added here and rebuilt
install(PROGRAMS
  scripts/optitrack_node.py
  scripts/imu_zupt_prep.py
  scripts/drive_relay.py
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
