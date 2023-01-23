from setuptools import setup, Extension
from Cython.Build import cythonize
import cyspams
import sys
from os import environ

include_dirs = []
libraries = []
library_dirs = []
extra_compile_args = []

# cyspams headers
include_dirs.extend(cyspams.get_include())

# OpenBLAS library (cyspams requirement)
try:
      openblas_dir = environ['OPENBLAS_DIR']
except KeyError as err:
      print(f"\033[31mKeyError: cannot find the {err} env variable\033[0m")
      exit(1)

include_dirs.extend([openblas_dir+'/include'])
library_dirs.extend([openblas_dir+'/lib'])

if sys.platform.startswith('win32'):
      libraries.extend(['libopenblas']) # .lib filenames
      extra_compile_args.extend(['/std:c++14'])
if sys.platform.startswith('linux'):
      libraries.extend(['stdc++', 'openblas']) # library names (not filenames)
      extra_compile_args.extend(['-std=c++14'])
if sys.platform.startswith('darwin'):
      libraries.extend(['stdc++', 'openblas']) # library names (not filenames)
      extra_compile_args.extend(['-std=c++14'])

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
