# How to install AMICO
## Install from PyPI
```bash
$ pip install dmri-amico
```

### PyPI available prebuilt wheels
| | __Windows__ | __MacOS__ | __Linux__ |
|:-:|:-:|:-:|:-:|
| __Python 3.6__ | cp36-win_amd64<br>cp36-win32 | cp36-macosx_x86_64 | cp36-manylinux_x86_64<br>cp36-manylinux_aarch64 |
| __Python 3.7__ | cp37-win_amd64<br>cp37-win32 | cp37-macosx_x86_64 | cp37-manylinux_x86_64<br>cp37-manylinux_aarch64 |
| __Python 3.8__ | cp38-win_amd64<br>cp38-win32 | cp38-macosx_x86_64<br>cp38-macosx_arm64 | cp38-manylinux_x86_64<br>cp38-manylinux_aarch64 |
| __Python 3.9__ | cp39-win_amd64<br>cp39-win32 | cp39-macosx_x86_64<br>cp39-macosx_arm64 | cp39-manylinux_x86_64<br>cp39-manylinux_aarch64 |
| __Python 3.10__ | cp310-win_amd64<br>cp310-win32 | cp310-macosx_x86_64<br>cp310-macosx_arm64 | cp310-manylinux_x86_64<br>cp310-manylinux_aarch64 |
| __Python 3.11__ | cp311-win_amd64<br>cp311-win32 | cp311-macosx_x86_64<br>cp311-macosx_arm64 | cp311-manylinux_x86_64<br>cp311-manylinux_aarch64 |

!!! note
    If any of the prebuilt wheels listed above are compatible with your system, please feel free to contact us or open an issue [here](https://github.com/daducci/AMICO/issues).

## Install from source
!!! note
    To build and install `dmri-amico` you need to have the OpenBLAS library on your system. Other BLAS/LAPACK libraries may be supported in the future (e.g. Intel MKL).

### 1. Clone the repository
```bash
$ git clone https://github.com/daducci/AMICO.git
$ cd AMICO
```

### 2. Set up OpenBLAS
The location of the OpenBLAS library must be specified in a `site.cfg` file located in the root of the `AMICO` repository. You can use the `setup_site.py` helper script, which will create a `site.cfg` file with the specified settings for you.
```bash
# Help
$ python setup_site.py --help
usage: setup_site.py [-h] section library library_dir include_dir

positional arguments:
  section      Section name
  library      Library name
  library_dir  Library directory path
  include_dir  Include directory path

optional arguments:
  -h, --help   show this help message and exit

# Windows
$ python setup_site.py openblas libopenblas C:\OpenBLAS\lib C:\OpenBLAS\include

# MacOS/Linux
$ python setup_site.py openblas openblas /home/OpenBLAS/lib /home/OpenBLAS/include
```
You can create a `site.cfg` file by yourself. See the `site.cfg.example` file included in the `AMICO` repository for documentation.
```ini
# Windows
[openblas]
libraries = libopenblas
library_dirs = C:\OpenBLAS\lib
include_dirs = C:\OpenBLAS\include

# MacOS/Linux
[openblas]
libraries = openblas
library_dirs = /home/OpenBLAS/lib
include_dirs = /home/OpenBLAS/include
```

### 3. Build the wheel
```bash
$ pip install build
$ python -m build -w
```

### 4. Repair the wheel (add runtime dependency on OpenBLAS)
```bash
# Windows
$ pip install delvewheel
$ delvewheel repair --add-path OpenBLAS\bin -w dest_dir -v wheel_to_repair.whl

# MacOS
$ pip install delocate
$ DYLD_LIBRARY_PATH=OpenBLAS/lib delocate-wheel -w dest_dir -v wheel_to_repair.whl

# Linux
$ pip install auditwheel patchelf
$ auditwheel show wheel_to_repair.whl # Show your target platform tag (e.g. manylinux2014_x86_64)
$ LD_LIBRARY_PATH=OpenBLAS/lib auditwheel repair --plat target_platform -w dest_dir wheel_to_repair.whl
```

### 5. Install the `dmri-amico` repaired wheel
```bash
$ pip install dest_dir/repaired_wheel.whl
```
