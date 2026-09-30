# The container the ILC experiments run in: ROS 2 Humble (Ubuntu 22.04, Python 3.10) with
# MuJoCo and CasADi. Versions are the ones the 2026-09 Go1 experiments ran with.
#
#   docker build -t ilc_quad:humble docker/
FROM ros:humble-ros-base-jammy

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-colcon-common-extensions git \
        libgl1 libglx-mesa0 libglfw3 libosmesa6 libegl1 \
    && rm -rf /var/lib/apt/lists/*

# numpy stays at the 1.21.5 every ros-humble package is built against: mujoco declares
# numpy unpinned, and a numpy 2.x pulled in over it breaks rclpy (README, Setup)
RUN pip3 install --no-cache-dir "numpy==1.21.5" "mujoco==3.12.0" "casadi==3.8.0" "glfw==2.10.2"

# egl: offscreen rendering (replay_trials.py --video) on a GPU box; osmesa without a GPU,
# glx with a display. The experiments themselves never render.
ENV ROS_LOCALHOST_ONLY=1 \
    MUJOCO_MENAGERIE_PATH=/mujoco_menagerie \
    MUJOCO_GL=egl
WORKDIR /ilc_ws
CMD ["sleep", "infinity"]
