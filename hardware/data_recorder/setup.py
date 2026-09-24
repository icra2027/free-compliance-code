from setuptools import find_packages, setup

package_name = "data_recorder"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="User",
    maintainer_email="user@example.com",
    description="ROS 2 node to record synchronized Franka and camera data in LeRobot format.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "record_lerobot = data_recorder.lerobot_recorder_node:main",
            "replay_lerobot_episode = data_recorder.replay_episode:main",
        ],
    },
)
