from amico.core import Evaluation, setup
from amico.util import set_verbose, get_verbose
# from amico import core
# from amico import scheme
# from amico import lut
# from amico import models
# from amico import util

__all__ = ['Evaluation', 'setup', 'set_verbose', 'get_verbose']

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version
__version__ = version('dmri-amico')
