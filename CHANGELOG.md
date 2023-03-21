# Change Log
## All notable changes to AMICO will be documented in this file.

## `v2.0.0`
_2023-03-##_

### Changed 🛠️
- From multiprocessing to `multithreading` (huge speedup in models fit! 🚀)
    - `models` and `lut` modules in Cython
    - `fit()` methods in Cython
    - `dir_to_lut_idx()` method in Cython
- New `NODDI` maps name:
    - from `ICVF` to `NDI`
    - from `OD` to `ODI`
    - from `FISO` to `FWF`
- New output images name casing (e.g. from `FIT_NDI.nii.gz` to `fit_NDI.nii.gz`)
- Default Diffusion Tensor model fit method to compute the primary direction in each voxel from `WLS` to `OLS`
- Default number of threads used by BLAS libraries from `-1` (all threads) to `1`
- Parallel threads config name from `parallel_jobs` to `nthreads`
- Default `study_path='.'` and `subject='.'` in `core.Evaluation()`
- Default `ndirs=500` in `core.generate_kernels()`
- Expire the deprecated `ndirs` parameter in `core.setup()`

### Added ✨
- C++ interfaces for NNLS and LASSO solvers (provided by `spams-cython`)
- Loader-like context manager, `util.Loader()`
- `DTI_fit_method` config to choose the Diffusion Tensor model fit method (from DIPY)
    - `OLS` (Ordinary Least Squares)
    - `WLS` (Weighted Least Squares)
- `BLAS_nthreads` config to set the number of threads used in the threadpool-backend of common BLAS implementations (e.g. OpenBLAS)
- `doComputeRMSE` config to compute the Root Mean Square Error image
- Modulated maps for the NODDI model ([Tissue-Weighted Mean](https://csparker.github.io/research/2021/11/16/Tissue-weighted-mean.html)), e.g. `fit_NDI_modulated.nii.gz`

### Fixed 🐛
- DeprecationWarning: `dipy.reconst.shm.real_sym_sh_basis` (with dipy>=1.4.0)

---

## [1.5.4] - 2023-02-15

### Fixed
- `invalid command 'bdist_wheel'` error when install with pip <= v20.0.2
- [#155](https://github.com/daducci/AMICO/issues/155) Replaced the NiBabel `get_data()` method with `get_fdata()` (raise `ExpiredDeprecationError` with NiBabel >= v5.0.0)

## [1.5.3] - 2023-01-23

### Fixed
- Problem with the `set_solver()` method that raise a `multiple values for keyword argument` error
- Replaced `np.bool`, `np.float` and `np.object` deprecated aliases (raise `AttributeError` with NumPy >= v1.24.0)

## [1.5.2] - 2022-10-21

### Added
- 'b0_min_signal' parameter in 'load_data()' to crop to zero the signal in voxels where the b0 <= b0_min_signal * mean(b0[b0 > 0])
- 'replace_bad_voxels' parameter in 'load_data()' to replace NaN and Inf values in the signal

## [1.5.1] - 2022-09-21

### Fixed
- Check if DWI file is a 4D image
- Removed unused 'verbose' option in __init__ (now all is controleld by 'set_verbose')

## [1.5.0] - 2022-07-06

### Changed
- Implemented the 'synthesis.py' module for kernels generation
- 'datasynth' tool (CAMINO) no more needed

## [1.4.4] - 2022-06-14

### Fixed
- Import error in 'util.py' on Windows systems
- Loading errors in 'lut.py' on Windows systems

## [1.4.3] - 2022-05-18

### Fixed
- Problems with joblib and pickle on some systems (issue #136)
- Added 'packaging' to requirements as it is not present by default on some systems

### Changed
- Removed config option 'parallel_backend' as only 'loky' works in current implementation

## [1.4.2] - 2022-03-26

### Fixed
- Bug when using 'doSaveCorrectedDWI' (issue #134)

## [1.4.1] - 2022-03-20

### Fixed
- Replace all 'print' with 'PRINT'

## [1.4.0] - 2022-03-04

### Added
- Function 'amico.set_verbose' to control what is printed during execution

### Changed
- Required dependency from 'python-spams' to 'spams'

## [1.3.2] - 2022-01-31

### Changed
- Required dependency numpy>=1.12,<1.22
- amico.core.setup: removed 'ndirs' parameter, now precomputes all directions

### Added
- Shortcut 'amico.core.setup' to 'amico.setup'

## [1.3.1] - 2021-12-03

### Fixed
- Removed unused hasISO parameter in experimental VolumeFractions model

### Changed
- Install information are stored (and taken from) amico/info.py

## [1.3.0] - 2021-10-27

### Added
- Possibility to fit different voxels in parallel
- Config options 'parallel_jobs' and 'parallel_backend' for better control of parallel execution
- Possibility to specify d_perp<>0  for the Stick in StickZeppelinBall

### Fixed
- Forcing SPAMS to use only one thread
- Added missing dependencies, e.g. wheel

### Changed
- Replaced in-house progress bar to tqdm

## [1.2.10] - 2021-05-20

### Fixed
- Warning message in util.py fsl2scheme and sandi2scheme

## [1.2.9] - 2021-02-23

### Added
- Controls if files exist before opening

### Fixed
- Wrong datatype when saving results

## [1.2.8] - 2021-01-29

### Added
- SANDI model

## [1.2.7] - 2020-10-28

### Added
- Possibility to set one single direction in the LUT resolution

## [1.2.6] - 2020-10-22

### Added
- Function get_params to get the specific parameter values of a model

## [1.2.5] - 2020-08-06

### Changed
- Moved the documentation to the Wiki

## [1.2.4] - 2020-06-10

### Added
- Added the parameter 'd_par_zep' in the StickZeppelinBall model

## [1.2.3] - 2020-05-25

### Fixed
- Modify setup.py and fix spams dependency

## [1.2.2] - 2020-05-05

### Fixed
- Checks if input mask is 3D.

## [1.2.1] - 2020-04-25

### Changed
- Use d_perps to set models instead of ICVFs.
- Canged case of some parameters for consistency

## [1.2.0] - 2020-04-01

### Added
- Functions to colorize the output messages.

## [1.1.0] - 2019-10-30

This version of AMICO *is not compatible* with [COMMIT](https://github.com/daducci/COMMIT) v1.2 of below. If you update AMICO to this version, please update COMMIT to version 1.3 or above.

### Added
- Changelog file to keep tracking of the AMICO versions.
- New feature that allows to decrease the LUT resolution. [Example here.](https://github.com/ErickHernandezGutierrez/AMICO/blob/lowresLUT/doc/demos/NODDI_lowres.md)
