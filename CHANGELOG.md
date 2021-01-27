
# Change Log
All notable changes to AMICO will be documented in this file.

## [1.2.8] - 2021-01-27

### Added

- MANIFEST.in
- Added direction-average and SANDI model 


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
