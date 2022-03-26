# -*- coding: UTF-8 -*-

# Format version as expected by setup.py (string of form "X.Y.Z")
_version_major = 1
_version_minor = 4
_version_micro = 2
_version_extra = '' #'.dev'
__version__    = "%s.%s.%s%s" % (_version_major,_version_minor,_version_micro,_version_extra)

NAME                = 'dmri-amico'
DESCRIPTION         = 'Accelerated Microstructure Imaging via Convex Optimization (AMICO)'
LONG_DESCRIPTION    = """
=======
 AMICO
=======

Implementation of the linear framework for Accelerated Microstructure Imaging via Convex Optimization (AMICO) described here:

| Accelerated Microstructure Imaging via Convex Optimization (AMICO) from diffusion MRI data
| Alessandro Daducci, Erick Canales-Rodriguez, Hui Zhang, Tim Dyrby, Daniel Alexander, Jean-Philippe Thiran
| NeuroImage 105, pp. 32-44 (2015)
"""
URL                 = 'https://github.com/daducci/AMICO'
DOWNLOAD_URL        = "N/A"
LICENSE             = 'BSD license'
AUTHOR              = 'Alessandro Daducci'
AUTHOR_EMAIL        = 'alessandro.daducci@univr.it'
PLATFORMS           = "OS independent"
MAJOR               = _version_major
MINOR               = _version_minor
MICRO               = _version_micro
VERSION             = __version__
