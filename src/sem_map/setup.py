from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'sem_map'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    # packages=[package_name, submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'launch'), glob('launch/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zhang Shengce',
    maintainer_email='shengcezhang@163.com',
    description='build semantic map',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'update_map = sem_map.update_map:main',
                'image_socket_recv = sem_map.image_socket_recv:main',
                'query_object = sem_map.query_object:main',
                'test_transform_point = sem_map.test_transform_point:main',
                'sem_map_service_yolo_lseg = sem_map.sem_map_service_yolo_lseg:main',
                'sem_map_service_yw_lseg = sem_map.sem_map_service_yw_lseg:main',
                'sem_map_service_cyw1_lseg = sem_map.sem_map_service_cyw1_lseg:main',
                'sem_map_service_lseg_feat = sem_map.sem_map_service_lseg_feat:main',
                'sem_map_service_gd_lseg = sem_map.sem_map_service_gd_lseg:main',
                'query_socket_handler = sem_map.query_socket_handler:main',
                'image_transform_client = sem_map.image_transform_client:main',
                'gdino_query_client = sem_map.gdino_query_client:main',
        ],
    },
)
