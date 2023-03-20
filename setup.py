from setuptools import setup, Extension
from Cython.Build import cythonize
import cyspams
import sys
from os import cpu_count
from os.path import isfile
import configparser

libraries = []
library_dirs = []
include_dirs = []
extra_compile_args = []
# spams-cython headers
include_dirs.extend(cyspams.get_include())
# OpenBLAS headers
if not isfile('site.cfg'):
      print(f'\033[31mFileNotFoundError: cannot find the site.cfg file\033[0m')
config = configparser.ConfigParser()
config.read('site.cfg')
try:
      libraries.extend([config['openblas']['library']])
      library_dirs.extend([config['openblas']['library_dir']])
      include_dirs.extend([config['openblas']['include_dir']])
except KeyError as err:
      print(f'\033[31mKeyError: cannot find the {err} key in the site.cfg file. See the site.cfg.example file for documentation\033[0m')
      exit(1)

if sys.platform.startswith('win32'):
      extra_compile_args.extend(['/std:c++14', '/fp:fast'])
if sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
      libraries.extend(['stdc++'])
      extra_compile_args.extend(['-std=c++14', '-Ofast'])

extensions = [
      Extension(
            'amico.models',
            sources=['amico/models.pyx'],
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            extra_compile_args=extra_compile_args
      ),
      Extension(
            'amico.lut',
            sources=['amico/lut.pyx']
      )
]

setup(
      url='https://github.com/daducci/AMICO', # NOTE this is to show the 'Home-page' field when run 'pip show dmri-amico' (open bug https://github.com/pypa/pip/issues/11221)
      ext_modules=cythonize(extensions)
)
