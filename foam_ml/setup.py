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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='daniel',
    maintainer_email='danielaugustin2027@u.northwestern.edu',
    description='ML models and interactive interfaces for the foam robot',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'train_model         = foam_ml.train_model:main',
            'option1_dome        = foam_ml.option1_dome:main',
            'option2_coordinate  = foam_ml.option2_coordinate:main',
            'option3_path_draw   = foam_ml.option3_path_draw:main',
        ],
    },
)
