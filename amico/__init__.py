from __future__ import absolute_import, division, print_function

from .core import Evaluation, setup
from .util import set_verbose, get_verbose
from . import core
from . import scheme
from . import lut
from . import models
from . import util

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version
__version__ = version('dmri-amico')
