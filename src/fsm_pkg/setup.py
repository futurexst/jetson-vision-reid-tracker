from setuptools import find_packages, setup
from glob import glob

package_name = 'fsm_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    include_package_data=True,
    package_data={'fsm_pkg': ['*.pt', '*.engine']},
    zip_safe=True,
    maintainer='Wansik Kim',
    maintainer_email='nextsmile@naver.com',
    description='Mode-based FSM controller and distributed ROS2 communication monitor',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'fsm_node = fsm_pkg.fsm_node:main',
            'comm_monitor_node = fsm_pkg.comm_monitor_node:main',
            'rear_cam_node = fsm_pkg.rear_cam_node:main',
        ],
    },
)
