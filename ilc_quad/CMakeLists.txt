cmake_minimum_required(VERSION 3.5)
project(ilc_quad)

# Python-only today. A C++ node needs rclcpp (and whatever msgs it uses) added
# here, plus add_executable / ament_target_dependencies / install(TARGETS ...).
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# The ilc_quad/ module: quad model, check_model, ILC (ilc_gen), Go2 LowCmd helpers.
# Imported by the scripts below.
ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

# RENAME drops the .py so `ros2 run ilc_quad sim_node` and the launch file's
# executable="sim_node" both work. RENAME takes one file, hence one call each.
install(PROGRAMS
  scripts/sim_node.py
  scripts/ilc_jump_sim.py
  scripts/go2_bridge.py
  scripts/go1_bridge.py
  scripts/ilc_jump_lockstep.py
  scripts/replay_trials.py
  scripts/dilc_execute.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
