from setuptools import find_packages, setup

package_name = 'object_tracker'
sobmodules=package_name+"/submodules"

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, sobmodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fyp',
    maintainer_email='shengcezhang@163.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 'owl_vit = object_tracker.owl_vit:main',
            # 'gdino = object_tracker.gdino:main',
            # 'sam = object_tracker.sam:main',
            'query = object_tracker.query:main',
            'deTrack = object_tracker.deTrack:main',
        ],
    },
)
