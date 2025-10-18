from setuptools import find_packages, setup

package_name = 'grasp_pose_detection'

setup(
    name=package_name,
    version='0.0.2',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/hggd_launch.py']),  # install launch file
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='ROS2 nodes for online grasp detection with HGGD (camera topics, per-object top-K).',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'hggd_grasp_service = grasp_pose_detection.grasp_planner:main',
        ],
    },
)
