from setuptools import setup

package_name = 'rccar_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Minsoo Kim',
    maintainer_email='minsoo.kim@rllab.snu.ac.kr',
    description='Bridge for rccar gym',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rccar_bringup = rccar_bringup.run_env:main',
            'keyboard_control = rccar_bringup.keyboard_control:main'
        ],
    },
)
