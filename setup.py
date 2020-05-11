from setuptools import setup, find_packages

description = 'Implementation of the linear framework for Accelerated Microstructure Imaging via Convex Optimization (AMICO) from diffusion MRI data'

setup(name='amico-daducci',
      version='1.2.2',
      description='Accelerated Microstructure Imaging via Convex Optimization (AMICO)',
      long_description=description,
      author='Alessandro Daducci',
      author_email='alessandro.daducci@univr.it',
      url='https://github.com/daducci/AMICO',
      packages=find_packages(),
      setup_requires=['Cython==0.29.17', 'numpy==1.18.4'],
      package_data={'': ['*.bin', 'directions/*.bin']},
      install_requires=['dipy==1.1.1', 'scipy==1.4.1', 'spams==2.6.1'])
