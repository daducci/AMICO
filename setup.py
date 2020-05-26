from setuptools import setup, find_packages

description = 'Implementation of the linear framework for Accelerated Microstructure Imaging via Convex Optimization (AMICO) from diffusion MRI data'

setup(name='dmri-amico',
      version='1.2.3.3',
      description='Accelerated Microstructure Imaging via Convex Optimization (AMICO)',
      long_description=description,
      author='Alessandro Daducci',
      author_email='alessandro.daducci@univr.it',
      url='https://github.com/daducci/AMICO',
      packages=find_packages(),
      setup_requires=['numpy==1.18.*'],
      install_requires=['numpy==1.18.*', 'dipy==1.1.*', 'scipy==1.4.*', 'python-spams==2.6.1.*'],
      package_data={'': ['*.bin', 'directions/*.bin']})
