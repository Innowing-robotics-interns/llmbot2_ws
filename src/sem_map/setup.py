from setuptools import find_packages, setup

package_name = 'sem_map'
# submodules=package_name+"/submodules"

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    # packages=[package_name, submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
                'image_socket_send = sem_map.image_socket_send:main',
        ],
    },
)
