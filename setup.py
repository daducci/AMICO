from setuptools import setup, find_packages

# import details from amico/info.py
import sys
sys.path.insert(0, './amico/')
import info

setup(name=info.NAME,
      version=info.VERSION,
      description=info.DESCRIPTION,
      long_description=info.LONG_DESCRIPTION,
      author=info.AUTHOR,
      author_email=info.AUTHOR_EMAIL,
      url=info.URL,
      license=info.LICENSE,
      packages=find_packages(),
      install_requires=['wheel', 'numpy>=1.12', 'scipy>=1.0', 'dipy>=1.0', 'spams>=2.6.5.2', 'tqdm>=4.56.0', 'joblib>=1.0.1'],
      package_data={'': ['*.bin', 'directions/*.bin']})
