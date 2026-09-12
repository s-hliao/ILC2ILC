from glob import glob

from setuptools import find_packages, setup

package_name = "ilc_quad"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Henry Liao",
    maintainer_email="liao.henry2@gmail.com",
    description="MuJoCo simulation of the Unitree Go2/Go1 for quadruped jumping control.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sim_node = ilc_quad.sim_node:main",
            "check_model = ilc_quad.check_model:main",
        ],
    },
)
