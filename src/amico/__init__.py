"""
Attributes
----------
__version__ : str
    The version of dmri-amico
"""

from __future__ import absolute_import, division, print_function

from amico.core import Evaluation, setup
from amico.util import get_version, set_verbose, get_verbose

__version__ = get_version()

__all__ = ['Evaluation', 'setup', 'set_verbose', 'get_verbose', '__version__']
