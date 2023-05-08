from __future__ import absolute_import, division, print_function

import numpy as np
import os.path
import itertools
from threading import Thread
from time import time, sleep
from sys import exit

__VERBOSE_LEVEL__ = 3

def set_verbose( verbose: int ):
	"""Set the verbosity of all functions.

	Parameters
	----------
	verbose : int
        3 = show everything
		2 = show messages but no progress bars
        1 = show only warnings/errors
        0 = hide everything
	"""
	global __VERBOSE_LEVEL__
	if type(verbose) != int or verbose not in [0,1,2,3]:
		raise TypeError( '"verbose" must be either 0, 1, 2 or 3' )
	__VERBOSE_LEVEL__ = verbose

def get_verbose():
    return __VERBOSE_LEVEL__

def PRINT( *args, **kwargs ):
    if __VERBOSE_LEVEL__ >= 2:
        print( *args, **kwargs )

def LOG( msg, prefix='' ):
    if __VERBOSE_LEVEL__ >= 2:
        print( prefix+"\033[0;32m%s\033[0m" % msg )

def NOTE( msg, prefix='' ):
    if __VERBOSE_LEVEL__ == 2:
        print( prefix+"\033[0;30;44m[ NOTE ]\033[0;34m %s\033[0m" % msg )

def WARNING( msg, prefix='' ):
    if __VERBOSE_LEVEL__ >= 1:
        print( prefix+"\033[0;30;43m[ WARNING ]\033[0;33m %s\033[0m" % msg )

def ERROR( msg, prefix='' ):
    if __VERBOSE_LEVEL__ >= 1:
        print( prefix+"\033[0;30;41m[ ERROR ]\033[0;31m %s\033[0m\n" % msg )
    try:
        from os import EX_USAGE # Only available on UNIX systems
        exit(EX_USAGE)
    except ImportError:
        exit(1) # Exit with error code 1 if on Windows system


def fsl2scheme( bvalsFilename, bvecsFilename, schemeFilename = None, flipAxes = [False,False,False], bStep = 1.0, delimiter = None ):
    """Create a scheme file from bvals+bvecs and write to file.

    If required, b-values can be rounded up to a specific threshold (bStep parameter).

    Parameters
    ----------
    :param str bvalsFilename: The path to bval file.
    :param str bvecsFilename: The path to bvec file.
    :param str schemeFilename: The path to output scheme file (optional).
    :param list of three boolean flipAxes: Whether to flip or not each axis (optional).
    :param float or list or np.bStep: If bStep is a scalar, round b-values to nearest integer multiple of bStep. If bStep is a list, it is treated as an array of shells in increasing order. B-values will be forced to the nearest shell value.
    :param str delimiter: Change the delimiter used by np.loadtxt (optional). None means "all white spaces".
    """

    if not os.path.exists(bvalsFilename):
        ERROR( 'bvals file not exist:' + bvalsFilename )
    if not os.path.exists(bvecsFilename):
        ERROR( 'bvecs file not exist:' + bvecsFilename )

    if schemeFilename is None:
        schemeFilename = os.path.splitext(bvalsFilename)[0]+".scheme"

    # load files and check size
    bvecs = np.loadtxt( bvecsFilename, delimiter=delimiter)
    bvals = np.loadtxt( bvalsFilename, delimiter=delimiter )

    if bvecs.ndim !=2 or bvals.ndim != 1 or bvecs.shape[0] != 3 or bvecs.shape[1] != bvals.shape[0]:
        ERROR( 'incorrect/incompatible bval/bvecs files' )

    # if requested, flip the axes
    flipAxes = np.array(flipAxes, dtype = np.bool_)
    if flipAxes.ndim != 1 or flipAxes.size != 3 :
        ERROR( '"flipAxes" must contain 3 boolean values (one for each axis)' )
    if flipAxes[0] :
        bvecs[0,:] *= -1
    if flipAxes[1] :
        bvecs[1,:] *= -1
    if flipAxes[2] :
        bvecs[2,:] *= -1

    # if requested, round the b-values
    bStep = np.array(bStep, dtype = np.double)
    if bStep.size == 1 and bStep > 1.0:
        PRINT("-> Rounding b-values to nearest multiple of %s" % np.array_str(bStep))
        bvals = np.round(bvals/bStep) * bStep
    elif bStep.size > 1:
        PRINT("-> Setting b-values to the closest shell in %s" % np.array_str(bStep))
        for i in range(0, bvals.size):
            diff = min(abs(bvals[i] - bStep))
            ind = np.argmin(abs(bvals[i] - bStep))

            # warn if b > 99 is set to 0, possible error
            if (bStep[ind] == 0.0 and diff > 100) or (bStep[ind] > 0.0 and diff > bStep[ind] / 20.0):
                # For non-zero shells, warn if actual b-value is off by more than 5%. For zero shells, warn above 50. Assuming s / mm^2
                WARNING("Measurement %d has b-value %d, being forced to %d\n'" % (i, bvals[i], bStep[ind]))

            bvals[i] = bStep[ind]

    # write corresponding scheme file
    np.savetxt( schemeFilename, np.c_[bvecs.T, bvals], fmt="%.06f", delimiter="\t", header="VERSION: BVECTOR", comments='' )
    LOG("\n-> Writing scheme file to [ %s ]" % schemeFilename)
    return schemeFilename


def sandi2scheme( bvalsFilename, bvecsFilename, Delta_data, smalldel_data, TE_data = None, schemeFilename = None, flipAxes = [False,False,False], bStep = 1.0, delimiter = None ):
    """Create a scheme file from bvals+bvecs and write to file.

    If required, b-values can be rounded up to a specific threshold (bStep parameter).

    Parameters
    ----------
    :param str bvalsFilename: The path to bval file.
    :param str bvecsFilename: The path to bvec file.
    :param str Delta_data: string or numpy.ndarray. The path to Delta file or a value to use in all the scheme (seconds).
    :param str smalldel_data: string or numpy.ndarray. The path to (small) delta file or a value to use in all the scheme (seconds).
    :param str TE_data: string or numpy.ndarray. The path to echo time file or a value to use in all the scheme (seconds) (optional).
    :param str schemeFilename: The path to output scheme file (optional).
    :param list of three boolean flipAxes: Whether to flip or not each axis (optional).
    :param float or list or np.bStep: If bStep is a scalar, round b-values to nearest integer multiple of bStep. If bStep is a list, it is treated as an array of shells in increasing order. B-values will be forced to the nearest shell value.
    :param str delimiter: Change the delimiter used by np.loadtxt (optional). None means "all white spaces".
    """

    if not os.path.exists(bvalsFilename):
        ERROR( 'bvals file not exist:' + bvalsFilename )
    if not os.path.exists(bvecsFilename):
        ERROR( 'bvecs file not exist:' + bvecsFilename )
    if type(Delta_data) is str :
        if not os.path.exists(Delta_data):
            ERROR( 'delta file not exist:' + Delta_data )
    if type(smalldel_data) is str :
        if not os.path.exists(smalldel_data):
            ERROR( 'small delta file not exist:' + smalldel_data )

    if schemeFilename is None:
        schemeFilename = os.path.splitext(bvalsFilename)[0]+".scheme"

    # load files and check size
    bvecs    = np.loadtxt( bvecsFilename, delimiter=delimiter)
    bvals    = np.loadtxt( bvalsFilename, delimiter=delimiter )

    if bvecs.ndim !=2 or bvals.ndim != 1 or bvecs.shape[0] != 3 or bvecs.shape[1] != bvals.shape[0]:
        ERROR( 'incorrect/incompatible bval/bvecs files' )

    if type(Delta_data) is str :
        delta = np.loadtxt( Delta_data, delimiter=delimiter)
        if delta.ndim !=1  or  delta.shape[0] != bvals.shape[0]:
            ERROR('incorrect/incompatible delta files')
        if delta.mean( ) > 0.1:
            WARNING(f'The mean of the delta values is {delta.mean():.4f}, these values must be in seconds.')
    else:
        delta = np.ones_like(bvals) * Delta_data
        if Delta_data > 0.1:
            WARNING(f'The delta value is {delta.mean():.4f}, this value must be in seconds.')

    if type(smalldel_data) is str :
        smalldel = np.loadtxt( smalldel_data, delimiter=delimiter)
        if smalldel.ndim !=1 or  smalldel.shape[0] != bvals.shape[0]:
            ERROR('incorrect/incompatible small delta files')
        if smalldel.mean() > 0.1:
            WARNING(f'The mean of the small delta values is {smalldel.mean():.4f}, these values must be in seconds.')
    else:
        smalldel = np.ones_like(bvals) * smalldel_data
        if smalldel_data > 0.1:
            WARNING(f'The small delta value is {smalldel.mean():.4f}, this value must be in seconds.')



    if TE_data is None:
        TE = delta + smalldel
    else:

        if type(TE_data) is str :
            TE = np.loadtxt( TE_data, delimiter=delimiter)
            if TE.ndim !=1  or  TE.shape[0] != bvals.shape[0]:
                ERROR('incorrect/incompatible TE files')
        else:
            TE = np.ones_like(bvals) * TE_data
        if not (TE >= (delta + smalldel)).all():
                ERROR('The value TE < (Delta + delta) ')

    # if requested, flip the axes
    flipAxes = np.array(flipAxes, dtype = np.bool_)
    if flipAxes.ndim != 1 or flipAxes.size != 3 :
        ERROR( '"flipAxes" must contain 3 boolean values (one for each axis)' )
    if flipAxes[0] :
        bvecs[0,:] *= -1
    if flipAxes[1] :
        bvecs[1,:] *= -1
    if flipAxes[2] :
        bvecs[2,:] *= -1

    # if requested, round the b-values
    bStep = np.array(bStep, dtype = np.double)
    if bStep.size == 1 and bStep > 1.0:
        PRINT("-> Rounding b-values to nearest multiple of %s" % np.array_str(bStep))
        bvals = np.round(bvals/bStep) * bStep
    elif bStep.size > 1:
        PRINT("-> Setting b-values to the closest shell in %s" % np.array_str(bStep))
        for i in range(0, bvals.size):
            diff = min(abs(bvals[i] - bStep))
            ind = np.argmin(abs(bvals[i] - bStep))

            # warn if b > 99 is set to 0, possible error
            if (bStep[ind] == 0.0 and diff > 100) or (bStep[ind] > 0.0 and diff > bStep[ind] / 20.0):
                # For non-zero shells, warn if actual b-value is off by more than 5%. For zero shells, warn above 50. Assuming s / mm^2
                WARNING("Measurement %d has b-value %d, being forced to %d\n'" % (i, bvals[i], bStep[ind]))

            bvals[i] = bStep[ind]

    from amico.synthesis import _GAMMA
    G = np.sqrt( bvals*1e6 / (_GAMMA**2 *  smalldel**2 * (delta - smalldel/3.0) ) )

    # write corresponding scheme file
    np.savetxt( schemeFilename, np.c_[bvecs.T, G, delta, smalldel, TE], fmt="%.06f", delimiter="\t", header="VERSION: 1", comments='' )
    LOG("\n-> Writing scheme file to [ %s ]" % schemeFilename)
    return schemeFilename

class ProgressBar:
    """Class that provides a progress bar during long-running processes.
    
    It can be used either as a indeterminate or determinate progress bar.
    Determinate progress bar supports multithreaded progress tracking.
    It can be used as a context manager.

    Parameters
    ----------
    total : int
        Total number of steps (default: None)
    ncols : int
        Number of columns of the progress bar in the terminal (default: 58)
    refresh : float
        Refresh rate of the progress bar in seconds (default: 0.05)
    eta_refresh : float
        Refresh rate of the estimated time of arrival (eta) in seconds (default: 1)
    multithread_progress : np.ndarray
        Array of shape (nthreads,) that contains the progress of each thread (default: None)
    disable : bool
        Whether to disable the progress bar (default: False)
    """
    def __init__(self, total=None, ncols=58, refresh=0.05, eta_refresh=1, multithread_progress=None, disable=False):
        self.total = total
        self.ncols = ncols
        self.refresh = refresh
        self.eta_refresh = eta_refresh
        self.multithread_progress = multithread_progress
        self.disable = disable

        self._graphics = {
            'clear_line': '\x1b[2K',
            'reset': '\x1b[0m',
            'black': '\x1b[30m',
            'green': '\x1b[32m',
            'magenta': '\x1b[35m',
            'cyan': '\x1b[36m',
            'bright_black': '\x1b[90m'
        }

        self._done = False

        if self.total is None:
            bar_length = int(self.ncols // 2)
            self._steps = []
            for i in range(self.ncols - bar_length + 1):
                self._steps.append(f"{self._graphics['bright_black']}{'━' * i}{self._graphics['magenta']}{'━' * bar_length}{self._graphics['bright_black']}{'━' * (self.ncols - bar_length - i)}")
            for i in range(bar_length - 1):
                self._steps.append(f"{self._graphics['magenta']}{'━' * (i + 1)}{self._graphics['bright_black']}{'━' * (self.ncols - bar_length)}{self._graphics['magenta']}{'━' * (bar_length - i - 1)}")
        else:
            self._eta = '<eta --m --s>'
            self._start_time = 0
            self._last_time = 0
            self._progress = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def _update_eta(self):
        self._last_time = time()
        if self.multithread_progress is not None:
            self._progress = np.sum(self.multithread_progress)
        eta = (time() - self._start_time) * (self.total - self._progress) / self._progress if self._progress > 0 else 0
        self._eta = f'<eta {int(eta // 60):02d}m {int(eta % 60):02d}s>'

    def _animate(self):
        if self.total is None:
            for step in itertools.cycle(self._steps):
                if self._done:
                    break
                print(f"\r   {step}{self._graphics['reset']}", end='', flush=True)
                sleep(self.refresh)
        else:
            while True:
                if self._done:
                    break
                if time() - self._last_time > self.eta_refresh:
                    self._update_eta()
                if self.multithread_progress is not None:
                    self._progress = np.sum(self.multithread_progress)
                print(f"\r   {self._graphics['magenta']}{'━' * int(self.ncols * self._progress / self.total)}{self._graphics['bright_black']}{'━' * (self.ncols - int(self.ncols * self._progress / self.total))} {self._graphics['green']}{100 * self._progress / self.total:.1f}% {self._graphics['cyan']}{self._eta}{self._graphics['reset']}", end='', flush=True)
                sleep(self.refresh)

    def start(self):
        if not self.disable:
            if self.total is not None:
                self._start_time = time()
                self._last_time = self._start_time
            Thread(target=self._animate, daemon=True).start()

    def stop(self):
        self._done = True
        if not self.disable:
            print(self._graphics['clear_line'], end='\r', flush=True)
            if self.total is None:
                print(f"\r   {self._graphics['green']}{'━' * self.ncols} 100.0%{self._graphics['reset']}")
            else:
                if self.multithread_progress is not None:
                    self._progress = np.sum(self.multithread_progress)
                print(f"\r   {self._graphics['green']}{'━' * int(self.ncols * self._progress / self.total)}{'━' * (self.ncols - int(self.ncols * self._progress / self.total))} {100 * self._progress / self.total:.1f}%{self._graphics['reset']}")

    def update(self):
        self._progress += 1
