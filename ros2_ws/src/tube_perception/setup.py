from setuptools import find_packages, setup

package_name = "tube_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Tube Positioning Development",
    maintainer_email="dev@example.com",
    description="Static-image tube candidate detector ROS 2 wrapper",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "static_image_node = tube_perception.static_image_node:main",
            "detect_image = tube_perception.cli:main",
        ]
    },
)
