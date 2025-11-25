from setuptools import setup, find_packages

setup(
    name='rccar_gym',
    version='25.0.0',
    packages=[package for package in find_packages() if package.startswith("rccar_gym")],
    install_requires=[
        'gymnasium',
        'numpy',
        'ruamel.yaml',
        'easydict',
        'numba',
        'pyyaml',
        'scipy',
        'requests',
        'pillow',
        'yamldataclassconfig==1.5.0',
        'opencv-python',
        'pygame',
        'scikit-learn',
        'gitpython',
        'lark',
        'empy==3.3.4',
        'matplotlib',
        'shapely',
        'wandb'
    ]
)
