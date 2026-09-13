from setuptools import setup
import os
from glob import glob

package_name = 'tts_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('lib', package_name), glob('tts_pkg/textmsg.txt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Wansik Kim',
    maintainer_email='nextsmile@naver.com',
    description='TTS node for robot guidance',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'tts_node = tts_pkg.tts_node:main',
        ],
    },
)
