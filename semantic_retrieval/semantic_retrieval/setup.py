import os
from glob import glob
from setuptools import setup

package_name = 'semantic_retrieval'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Team Anveshan',
    maintainer_email='team@anveshan.local',
    description='DINOv2 semantic video-frame retrieval nodes for the Anveshan ASCEND mission.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'indexer_node = semantic_retrieval.indexer_node:main',
            'retriever_node = semantic_retrieval.retriever_node:main',
            'hd_frame_server = semantic_retrieval.hd_frame_server:main',
        ],
    },
)
