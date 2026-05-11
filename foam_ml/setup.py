import glob
import os
from setuptools import find_packages, setup

package_name = 'foam_ml'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models',
            [f for f in glob.glob('models/*') if os.path.isfile(f)]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='daniel',
    maintainer_email='danielaugustin2027@u.northwestern.edu',
    description='ML models and interactive interfaces for the foam robot',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            # TODO: train_model, option1_dome, option2_coordinate, option3_path_draw
        ],
    },
)
