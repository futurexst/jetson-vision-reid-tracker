from setuptools import find_packages, setup

package_name = 'pickup_pkg'

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
    maintainer='Wansik Kim',
    maintainer_email='nextsmile@naver.com',
    description='Camera-to-base coordinate transform and docking/grasp target calculation',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pickup_node = pickup_pkg.pickup_node:main',
        ],
    },
)
