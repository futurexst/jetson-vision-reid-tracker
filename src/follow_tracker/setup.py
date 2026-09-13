from setuptools import find_packages, setup

package_name = 'follow_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    include_package_data=True,
    package_data={'follow_tracker': ['*.pt', '*.yaml']},
    maintainer='Wansik Kim',
    maintainer_email='nextsmile@naver.com',
    description='YOLO11n TensorRT detection with CLIP-ReID target re-identification (Python)',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'follow_node = follow_tracker.follow_node:main',
        ],
    },
)
