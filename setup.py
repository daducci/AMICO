from setuptools import setup, find_packages

description = 'Implementation of the linear framework for Accelerated Microstructure Imaging via Convex Optimization (AMICO) from diffusion MRI data'

setup(name='dmri-amico',
      version='1.2.10',
      description='Accelerated Microstructure Imaging via Convex Optimization (AMICO)',
      long_description=description,
      author='Alessandro Daducci',
      author_email='alessandro.daducci@univr.it',
      url='https://github.com/daducci/AMICO',
      license='BSD license',
      packages=find_packages(),
      setup_requires=['numpy>=1.12'],
      install_requires=['numpy>=1.12', 'dipy>=1.0', 'scipy>=1.0', 'python-spams>=2.6.1'],
      package_data={'': ['*.bin', 'directions/*.bin']})
