from setuptools import find_packages, setup

package_name = "flyguard_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="flyguard",
    maintainer_email="azmozgame@gmail.com",
    description=("ROS2 nodes running the real FAFB v783 connectome escape "
                 "circuit as a live collision detector."),
    license="MIT",
    entry_points={
        "console_scripts": [
            "looming_node = flyguard_ros2.looming_node:main",
            "demo_stimulus_node = flyguard_ros2.demo_stimulus_node:main",
            "vision_node = flyguard_ros2.vision_node:main",
            "camera_node = flyguard_ros2.camera_node:main",
        ],
    },
)
