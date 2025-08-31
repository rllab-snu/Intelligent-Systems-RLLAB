from setuptools import setup, find_packages

setup(name='rccar_gym',
      version='24.0.0',
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
          'yamldataclassconfig',
          'opencv-python',
          'pygame',
          'scikit-learn',
          'gitpython',
      ]
)