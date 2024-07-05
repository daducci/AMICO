# distutils: language = c++
# cython: language_level = 3

import numpy as np
from os import environ
from os.path import join as pjoin
import sys
import amico.lut
from abc import ABC, abstractmethod
from amico.util import PRINT, ERROR, get_verbose
from dicelib.ui import ProgressBar
from amico.synthesis import Stick, Zeppelin, Ball, CylinderGPD, SphereGPD, Astrosticks, NODDIIntraCellular, NODDIExtraCellular, NODDIIsotropic
from concurrent.futures import ThreadPoolExecutor

cimport cython
from libc.math cimport pi, atan2, sqrt, pow as cpow
from amico.lut cimport dir_to_lut_idx
from cyspams.interfaces cimport nnls, lasso

try:
    sys.path.append(environ['AMICO_WIP_MODELS'])
    from amicowipmodels import *
except KeyError:
    pass
except ImportError:
    pass

_MULTITHREAD_PROGRESS = np.zeros(1, dtype=np.intc)
cdef int [::1]_MULTITHREAD_PROGRESS_VIEW = _MULTITHREAD_PROGRESS

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _init_multithread_progress(int nthreads):
    global _MULTITHREAD_PROGRESS
    global _MULTITHREAD_PROGRESS_VIEW
    _MULTITHREAD_PROGRESS = np.zeros(nthreads, dtype=np.intc)
    _MULTITHREAD_PROGRESS_VIEW = _MULTITHREAD_PROGRESS

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _update_multithread_progress(int thread_id) noexcept nogil:
    global _MULTITHREAD_PROGRESS_VIEW
    _MULTITHREAD_PROGRESS_VIEW[thread_id] += 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_rmse(double [::1, :]A_view, double [::1]y_view, double [::1]x_view, double *y_est_view, double *rmse_view) noexcept nogil:
    cdef Py_ssize_t i, j
    for i in range(A_view.shape[0]):
        y_est_view[i] = 0.0
        for j in range(A_view.shape[1]):
            y_est_view[i] += A_view[i, j] * x_view[j]
        rmse_view[0] += cpow(y_view[i] - y_est_view[i], 2.0) / y_view.shape[0]
    rmse_view[0] = sqrt(rmse_view[0])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_nrmse(double [::1, :]A_view, double [::1]y_view, double [::1]x_view, double *y_est_view, double *nrmse_view) noexcept nogil:
    cdef double den = 0.0
    cdef Py_ssize_t i, j
    for i in range(A_view.shape[0]):
        y_est_view[i] = 0.0
        den += cpow(y_view[i], 2.0)
        for j in range(A_view.shape[1]):
            y_est_view[i] += A_view[i, j] * x_view[j]
    if den > 1e-16:
        for i in range(A_view.shape[0]):
            nrmse_view[0] += cpow(y_view[i] - y_est_view[i], 2.0) / den
        nrmse_view[0] = sqrt(nrmse_view[0])
    else:
        nrmse_view[0] = 0.0



class BaseModel(ABC) :
    """Basic class to build a model; new models should inherit from this class.
    All the methods need to be overloaded to account for the specific needs of the model.
    Each method will then be called by a dispatcher when needed.

    NB: this model also serves the purpose of illustrating the creation of new models.

    Attributes
    ----------
    id : string
        Identification code for the model
    name : string
        A more human-readable description for the model (can be equal to id)
    scheme: Scheme class
        Acquisition scheme to be used for resampling
    maps_name : list of strings
        Names of the maps computed/returned by the model (suffix to saved filenames)
    maps_descr : list of strings
        Description of each map (will be saved in the description of the NIFTI header)
    """
    @abstractmethod
    def __init__( self ) :
        """To define the parameters of the model, e.g. id and name, returned maps,
        model-specific parameters etc.
        """
        self.id         = 'BaseModel'
        self.name       = 'Base Model'
        self.maps_name  = []
        self.maps_descr = []
        self.scheme = None
        return


    @abstractmethod
    def set( self, *args, **kwargs ) :
        """For setting all the parameters specific to the model.
        NB: the parameters are model-dependent.
        """
        return


    @abstractmethod
    def get_params( self ) :
        """For getting the actual values of all the parameters specific to the model.
        NB: the parameters are model-dependent.
        """
        return


    @abstractmethod
    def set_solver( self ) :
        """For setting the parameters required by the solver to fit the model.
        NB: the parameters are model-dependent.

        Returns
        -------
        params : dictionary
            All the parameters that the solver will need to fit the model
        """
        self.solver_params = {}


    @abstractmethod
    def generate( self, out_path, aux, idx_in, idx_out, ndirs ) :
        """For generating the signal response-functions and createing the LUT.
        NB: do not change the signature!

        Parameters
        ----------
        out_path : string
            Path where the response function have to be saved
        aux : structure
            Auxiliary structures to perform SH fitting and rotations
        idx_in : array
            Indices of the samples belonging to each shell
        idx_out : array
            Indices of the SH coefficients corresponding to each shell
        ndirs : int
            Number of directions on the half of the sphere representing the possible orientations of the response functions
        """
        return


    @abstractmethod
    def resample( self, in_path, idx_out, Ylm_out, doMergeB0, ndirs ) :
        """For projecting the LUT to the subject space.
        NB: do not change the signature!

        Parameters
        ----------
        in_path : Scheme class
            Acquisition scheme of the acquired signal
        idx_out : array
            Indices of the samples belonging to each shell
        Ylm_out : array
            SH bases to project back each shell to signal space
        doMergeB0: bool
            Merge b0-volumes into a single volume if True
        ndirs : int
            Number of directions on the half of the sphere representing the possible orientations of the response functions

        Returns
        -------
        KERNELS : dictionary
            Contains the LUT and all corresponding details. In particular, it is
            required to have a field 'model' set to "self.id".
        """
        return


    @abstractmethod
    def fit(self, evaluation):
        """For fitting the model to the data.
        NB: do not change the signature!

        Parameters
        ----------
        evaluation: amico.core.Evaluation
            AMICO Evaluaton instance

        Returns
        -------
        results: dict of {str: np.ndarray}
            'estimates': Scalar values eastimated in each voxel
            'rmse': Fitting error (Root Mean Square Error) (optional)
            'nrmse': Fitting error (Normalized Root Mean Square Error) (optional)
            'y_corrected': Corrected DWI (only FreeWater model) (optional)
            'estimates_mod': Modulated maps (only NODDI model) (optional)
        """
        # build chunks
        n = evaluation.y.shape[0]
        c = n // evaluation.nthreads
        self.chunks = []
        for i, j in zip(range(0, n, c), range(c, n+1, c)):
            self.chunks.append((i, j))
        if self.chunks[-1][1] != n:
            self.chunks[-1] = (self.chunks[-1][0], n)

        # common configs
        self.configs = {
            'compute_rmse': evaluation.get_config('doComputeRMSE'),
            'compute_nrmse': evaluation.get_config('doComputeNRMSE')
        }



class StickZeppelinBall( BaseModel ) :
    """Implements the Stick-Zeppelin-Ball model [1].

    The intra-cellular contributions from within the axons are modeled as "sticks", i.e.
    tensors with a given axial diffusivity (d_par) but null perpendicular diffusivity (d_perp=0);
    if d_perp>0, then a Zeppelin is used instead of a Stick.
    Extra-cellular contributions are modeled as "Zeppelins", i.e. tensors with a given axial
    diffusivity (d_par_zep) and, possibily, a series of perpendicular diffusivities (d_perps_zep).
    If the axial diffusivity of the Zeppelins is not specified, then it is assumed equal to that
    of the Stick. Isotropic contributions are modeled as "Balls", i.e. tensors with isotropic
    diffusivities (d_isos).

    References
    ----------
    .. [1] Panagiotaki et al. (2012) Compartment models of the diffusion MR signal
           in brain white matter: A taxonomy and comparison. NeuroImage, 59: 2241-54
    """
    def __init__( self ) :
        self.id         = 'StickZeppelinBall'
        self.name       = 'Stick-Zeppelin-Ball'
        self.maps_name  = [ ]
        self.maps_descr = [ ]
        self.set()


    def set(
        self,
        d_par=1.7E-3,
        d_perps_zep=np.array([1.19E-3, 0.85E-3, 0.51E-3, 0.17E-3]),
        d_isos=np.array([3.0E-3]),
        d_par_zep=1.7E-3,
        d_perp=0
        ):
        '''
        Set the parameters of the Stick-Zeppelin-Ball model.

        Parameters
        ----------
        d_par : float
            Parallel diffusivity for the Stick [mm^2/s]
        d_perp : float
            Perpendicular diffusivity for the Stick [mm^2/s]
        d_par_zep : float
            Parallel diffusivity for the Zeppelins [mm^2/s]
        d_perps_zep : list of floats
            Perpendicular diffusivitie(s) [mm^2/s]
        d_isos : list of floats
            Isotropic diffusivitie(s) [mm^2/s]
        '''
        self.d_par = d_par
        self.d_perp = d_perp
        if d_par_zep is None:
            self.d_par_zep = d_par
        else:
            self.d_par_zep = d_par_zep
        self.d_perps_zep = np.array( d_perps_zep )
        self.d_isos  = np.array( d_isos )


    def get_params( self ) :
        params = {}
        params['id'] = self.id
        params['name'] = self.name
        params['d_par'] = self.d_par
        params['d_perp'] = self.d_perp
        params['d_par_zep'] = self.d_par_zep
        params['d_perps_zep'] = self.d_perps_zep
        params['d_isos'] = self.d_isos
        return params


    def set_solver( self ) :
        ERROR( 'Not implemented' )


    def generate( self, out_path, aux, idx_in, idx_out, ndirs ) :
        scheme_high = amico.lut.create_high_resolution_scheme( self.scheme )
        stick = Stick(scheme_high)
        zeppelin = Zeppelin(scheme_high)
        ball = Ball(scheme_high)

        nATOMS = 1 + len(self.d_perps_zep) + len(self.d_isos)
        idx = 0
        with ProgressBar(total=nATOMS, disable=get_verbose()<3) as pbar:
            # Stick
            signal = stick.get_signal(self.d_par)
            lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, False, ndirs )
            np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
            idx += 1
            pbar.update()
            # Zeppelin(s)
            for d in self.d_perps_zep :
                signal = zeppelin.get_signal(self.d_par_zep, d)
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, False, ndirs )
                np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
                idx += 1
                pbar.update()
            # Ball(s)
            for d in self.d_isos :
                signal = ball.get_signal(d)
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, True, ndirs )
                np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
                idx += 1
                pbar.update()


    def resample( self, in_path, idx_out, Ylm_out, doMergeB0, ndirs ) :
        KERNELS = {}
        KERNELS['model'] = self.id
        if doMergeB0:
            nS = 1+self.scheme.dwi_count
            merge_idx = np.hstack((self.scheme.b0_idx[0],self.scheme.dwi_idx))
        else:
            nS = self.scheme.nS
            merge_idx = np.arange(nS)
        KERNELS['wmr']   = np.zeros( (1,ndirs,nS), dtype=np.float32 )
        KERNELS['wmh']   = np.zeros( (len(self.d_perps_zep),ndirs,nS), dtype=np.float32 )
        KERNELS['iso']   = np.zeros( (len(self.d_isos),nS), dtype=np.float32 )

        nATOMS = 1 + len(self.d_perps_zep) + len(self.d_isos)
        idx = 0
        with ProgressBar(total=nATOMS, disable=get_verbose()<3) as pbar:
            # Stick
            lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
            if lm.shape[0] != ndirs:
                ERROR( 'Outdated LUT. Call "generate_kernels( regenerate=True )" to update the LUT' )
            KERNELS['wmr'][0,...] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, False, ndirs )[:,merge_idx]
            idx += 1
            pbar.update()

            # Zeppelin(s)
            for i in range(len(self.d_perps_zep)) :
                lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                if lm.shape[0] != ndirs:
                    ERROR( 'Outdated LUT. Call "generate_kernels( regenerate=True )" to update the LUT' )
                KERNELS['wmh'][i,...] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, False, ndirs )[:,merge_idx]
                idx += 1
                pbar.update()

            # Ball(s)
            for i in range(len(self.d_isos)) :
                lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                KERNELS['iso'][i,...] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, True, ndirs )[merge_idx]
                idx += 1
                pbar.update()

        return KERNELS


    def fit(self, evaluation):
        ERROR( 'Not implemented' )



class CylinderZeppelinBall( BaseModel ) :
    """Implements the Cylinder-Zeppelin-Ball model [1].

    The intra-cellular contributions from within the axons are modeled as "cylinders"
    with specific radii (Rs) and a given axial diffusivity (d_par).
    Extra-cellular contributions are modeled as tensors with the same axial diffusivity
    as the cylinders (d_par) and, possibily, a series of perpendicular diffusivities (d_perps).
    Isotropic contributions are modeled as tensors with isotropic diffusivities (d_isos).

    NB: this model works only with schemes containing the full specification of
        the diffusion gradients (eg gradient strength, small delta etc).

    References
    ----------
    .. [1] Panagiotaki et al. (2012) Compartment models of the diffusion MR signal
           in brain white matter: A taxonomy and comparison. NeuroImage, 59: 2241-54
    """
    def __init__( self ) :
        self.id         = 'CylinderZeppelinBall'
        self.name       = 'Cylinder-Zeppelin-Ball'
        self.maps_name  = [ 'v', 'a', 'd' ]
        self.maps_descr = [ 'Intra-cellular volume fraction', 'Mean axonal diameter', 'Axonal density' ]
        self.set()


    def set(
        self,
        d_par=0.6E-3,
        Rs=np.concatenate(([0.01], np.linspace(0.5, 8.0, 20))) * 1E-6,
        d_perps=np.array([1.19E-3, 0.85E-3, 0.51E-3, 0.17E-3]),
        d_isos=np.array([2.0E-3])
        ):
        '''
        Set the parameters of the Cylinder-Zeppelin-Ball model.

        Parameters
        ----------
        d_par : float
            Parallel diffusivity [mm^2/s]
        Rs : list of floats
            Radii of the axons [meters]
        d_perps : list of floats
            Perpendicular diffusivitie(s) [mm^2/s]
        d_isos : list of floats
            Isotropic diffusivitie(s) [mm^2/s]
        '''
        self.d_par   = d_par
        self.Rs      = np.array(Rs)
        self.d_perps = np.array(d_perps)
        self.d_isos  = np.array(d_isos)


    def get_params( self ) :
        params = {}
        params['id'] = self.id
        params['name'] = self.name
        params['d_par'] = self.d_par
        params['Rs'] = self.Rs
        params['d_perps'] = self.d_perps
        params['d_isos'] = self.d_isos
        params['isExvivo'] = self.isExvivo
        return params


    def set_solver( self, lambda1 = 0.0, lambda2 = 4.0 ) :
        super().set_solver()
        self.solver_params['lambda1'] = lambda1
        self.solver_params['lambda2'] = lambda2


    def generate( self, out_path, aux, idx_in, idx_out, ndirs ) :
        if self.scheme.version != 1 :
            ERROR( 'This model requires a "VERSION: STEJSKALTANNER" scheme' )

        scheme_high = amico.lut.create_high_resolution_scheme( self.scheme )
        cylinder = CylinderGPD(scheme_high)
        zeppelin = Zeppelin(scheme_high)
        ball = Ball(scheme_high)

        nATOMS = len(self.Rs) + len(self.d_perps) + len(self.d_isos)
        idx = 0
        with ProgressBar(total=nATOMS, disable=get_verbose()<3) as pbar:
            # Cylinder(s)
            for R in self.Rs :
                signal = cylinder.get_signal(self.d_par, R)
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, False, ndirs )
                np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
                idx += 1
                pbar.update()

            # Zeppelin(s)
            for d in self.d_perps :
                signal = zeppelin.get_signal(self.d_par, d)
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, False, ndirs )
                np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
                idx += 1
                pbar.update()

            # Ball(s)
            for d in self.d_isos :
                signal = ball.get_signal(d)
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, True, ndirs )
                np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
                idx += 1
                pbar.update()


    def resample( self, in_path, idx_out, Ylm_out, doMergeB0, ndirs ) :
        if doMergeB0:
            nS = 1+self.scheme.dwi_count
            merge_idx = np.hstack((self.scheme.b0_idx[0],self.scheme.dwi_idx))
        else:
            nS = self.scheme.nS
            merge_idx = np.arange(nS)
        KERNELS = {}
        KERNELS['model'] = self.id
        KERNELS['wmr'] = np.zeros( (len(self.Rs),ndirs,nS,), dtype=np.float32 )
        KERNELS['wmh'] = np.zeros( (len(self.d_perps),ndirs,nS,), dtype=np.float32 )
        KERNELS['iso'] = np.zeros( (len(self.d_isos),nS,), dtype=np.float32 )

        nATOMS = len(self.Rs) + len(self.d_perps) + len(self.d_isos)
        idx = 0
        with ProgressBar(total=nATOMS, disable=get_verbose()<3) as pbar:
            # Cylinder(s)
            for i in range(len(self.Rs)) :
                lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                if lm.shape[0] != ndirs:
                    ERROR( 'Outdated LUT. Call "generate_kernels( regenerate=True )" to update the LUT' )
                KERNELS['wmr'][i,:,:] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, False, ndirs )[:,merge_idx]
                idx += 1
                pbar.update()

            # Zeppelin(s)
            for i in range(len(self.d_perps)) :
                lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                if lm.shape[0] != ndirs:
                    ERROR( 'Outdated LUT. Call "generate_kernels( regenerate=True )" to update the LUT' )
                KERNELS['wmh'][i,:,:] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, False, ndirs )[:,merge_idx]
                idx += 1
                pbar.update()

            # Ball(s)
            for i in range(len(self.d_isos)) :
                lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                KERNELS['iso'][i,:] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, True, ndirs )[merge_idx]
                idx += 1
                pbar.update()

        return KERNELS


    def fit(self, evaluation):
        super().fit(evaluation)

        # fit chunks in parallel
        global _MULTITHREAD_PROGRESS
        _init_multithread_progress(evaluation.nthreads)
        with ProgressBar(total=evaluation.y.shape[0], multithread_progress=_MULTITHREAD_PROGRESS, disable=get_verbose()<3):
            with ThreadPoolExecutor(max_workers=evaluation.nthreads) as executor:
                futures = [executor.submit(self._fit, thread_id, evaluation.y[i:j, :], evaluation.DIRs[i:j, :], evaluation.htable, evaluation.KERNELS) for thread_id, (i, j) in enumerate(self.chunks)]
                chunked_results = [f.result() for f in futures]
        
        # concatenate results and return
        results = {}
        for k in chunked_results[0]:
            results[k] = np.concatenate([cr[k] for cr in chunked_results])
        return results


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit(self, thread_id, y, dirs, hash_table, kernels):
        # configs
        cdef int thread_id_c = thread_id
        cdef bint is_exvivo = 1 if self.isExvivo else 0
        cdef bint compute_rmse = 1 if self.configs['compute_rmse'] else 0
        cdef bint compute_nrmse = 1 if self.configs['compute_nrmse'] else 0
        cdef int n_rs = len(self.Rs)
        cdef int n_perp = len(self.d_perps)
        cdef int n_iso = len(self.d_isos)
        cdef int n_atoms = n_rs + n_perp + n_iso
        # NOTE not implemented
        if is_exvivo:
            n_atoms += 1

        # solver params
        cdef double lambda1 = self.solver_params['lambda1']
        cdef double lambda2 = self.solver_params['lambda2']

        # directions
        cdef double [:, ::1]directions_view = np.ascontiguousarray(dirs, dtype=np.double)
        cdef short [::1]hash_table_view = hash_table
        cdef int lut_index

        # kernels
        cdef double [::1, :, :]kernels_wmr_view = np.asfortranarray(np.swapaxes(kernels['wmr'].T, 1, 2), dtype=np.double)
        cdef double [::1, :, :]kernels_wmh_view = np.asfortranarray(np.swapaxes(kernels['wmh'].T, 1, 2), dtype=np.double)
        cdef double [::1, :]kernels_iso_view = np.asfortranarray(kernels['iso'].T, dtype=np.double)

        # y, A, x
        cdef double [:, ::1]y_view = np.ascontiguousarray(y, dtype=np.double)
        cdef double [::1, :]A_view = np.zeros((kernels_wmr_view.shape[0], n_atoms), dtype=np.double, order='F')
        cdef double [::1]x_view = np.zeros(n_atoms, dtype=np.double)

        # return
        cdef double v = 0.0
        cdef double a = 0.0
        cdef double d = 0.0
        estimates = np.zeros((y_view.shape[0], len(self.maps_name)), dtype=np.double, order='C')
        cdef double [:, ::1]estimates_view = estimates

        # support variables
        cdef double f1 = 0.0
        cdef double f2 = 0.0
        cdef double ic_sum = 0.0
        cdef double [::1]rs_view = self.Rs

        # fitting error
        cdef double [::1]y_est_view
        cdef double [::1]rmse_view
        cdef double [::1]nrmse_view
        if compute_rmse or compute_nrmse:
            y_est_view = np.zeros(y_view.shape[0], dtype=np.double)
        if compute_rmse:
            rmse = np.zeros(y_view.shape[0], dtype=np.double)
            rmse_view = rmse
        if compute_nrmse:
            nrmse = np.zeros(y_view.shape[0], dtype=np.double)
            nrmse_view = nrmse
        
        cdef Py_ssize_t i, j
        with nogil:
            for i in range(y_view.shape[0]):
                # prepare dictionary
                lut_index = dir_to_lut_idx(directions_view[i, :], hash_table_view)
                A_view[:, :n_rs] = kernels_wmr_view[:, :, lut_index]
                A_view[:, n_rs:n_rs + n_perp] = kernels_wmh_view[:, :, lut_index]
                A_view[:, n_rs + n_perp:] = kernels_iso_view[:, :]

                # fit
                lasso(&A_view[0, 0], &y_view[i, 0], A_view.shape[0], A_view.shape[1], 1, &x_view[0], lambda1, lambda2)

                # estimates
                f1 = 0.0
                f2 = 0.0
                ic_sum = 0.0
                a = 0.0
                for j in range(n_rs + n_perp):
                    if j < n_rs:
                        f1 += x_view[j]
                    if j >= n_rs and j < n_rs + n_perp:
                        f2 += x_view[j]
                f2 += 1e-16
                v = f1 / (f1 + f2 + 1e-16)
                f1 += 1e-16
                for j in range(rs_view.shape[0]):
                    a += rs_view[j] * x_view[j]
                a = 1e6 * 2.0 * a / f1
                d = (4.0 * v) / (pi * cpow(a, 2.0) + 1e-16)
                estimates_view[i, 0] = v
                estimates_view[i, 1] = a
                estimates_view[i, 2] = d

                # fitting error
                if compute_rmse:
                    _compute_rmse(A_view, y_view[i, :], x_view, &y_est_view[0], &rmse_view[i])
                if compute_nrmse:
                    _compute_nrmse(A_view, y_view[i, :], x_view, &y_est_view[0], &nrmse_view[i])

                _update_multithread_progress(thread_id_c)

        results = {}
        results['estimates'] = estimates
        if compute_rmse:
            results['rmse'] = rmse
        if compute_nrmse:
            results['nrmse'] = nrmse
        return results



class NODDI( BaseModel ) :
    """Implements the NODDI model [2].

    NB: this model does not require to have the "NODDI MATLAB toolbox" installed;
        all the necessary functions have been ported to Python.

    References
    ----------
    .. [2] Zhang et al. (2012) NODDI: Practical in vivo neurite orientation
           dispersion and density imaging of the human brain. NeuroImage, 61: 1000-16
    """
    def __init__( self ):
        self.id         = "NODDI"
        self.name       = "NODDI"
        self.maps_name  = [ 'NDI', 'ODI', 'FWF' ]
        self.maps_descr = [ 'Neurite Density Index', 'Orientation Dispersion Index', 'Free Water Fraction' ]
        self.set()


    def set(
        self,
        dPar=1.7E-3,
        dIso=3.0E-3,
        IC_VFs=np.linspace(0.1, 0.99, 12),
        IC_ODs=np.hstack((np.array([0.03, 0.06]), np.linspace(0.09, 0.99, 10))),
        isExvivo=False
        ):
        '''
        Set the parameters of the NODDI model.

        Parameters
        ----------
        dPar : float
            Parallel diffusivity [mm^2/s]
        dIso : float
            Isotropic diffusivity [mm^2/s]
        IC_VFs : list of floats
            Intra-cellular volume fractions
        IC_ODs : list of floats
            Intra-cellular orientation dispersions
        isExvivo : bool
            Is ex-vivo data
        '''
        self.dPar      = dPar
        self.dIso      = dIso
        self.IC_VFs    = np.array( IC_VFs ) if isinstance(IC_VFs, list) else IC_VFs
        self.IC_ODs    = np.array( IC_ODs ) if isinstance(IC_ODs, list) else IC_ODs
        self.isExvivo  = isExvivo
        if isExvivo:
            self.maps_name.append('dot')
            self.maps_descr.append('Dot volume fraction')


    def get_params( self ) :
        params = {}
        params['id'] = self.id
        params['name'] = self.name
        params['dPar'] = self.dPar
        params['dIso'] = self.dIso
        params['IC_VFs'] = self.IC_VFs
        params['IC_ODs'] = self.IC_ODs
        params['isExvivo'] = self.isExvivo
        return params


    def set_solver( self, lambda1 = 5e-1, lambda2 = 1e-3 ):
        super().set_solver()
        self.solver_params['lambda1'] = lambda1
        self.solver_params['lambda2'] = lambda2


    def generate( self, out_path, aux, idx_in, idx_out, ndirs ):
        scheme_high = amico.lut.create_high_resolution_scheme( self.scheme )
        noddi_ic = NODDIIntraCellular(scheme_high)
        noddi_ec = NODDIExtraCellular(scheme_high)
        noddi_iso = NODDIIsotropic(scheme_high)

        nATOMS = len(self.IC_ODs)*len(self.IC_VFs) + 1
        idx = 0
        with ProgressBar(total=nATOMS, disable=get_verbose()<3) as pbar:
            # Coupled contributions
            IC_KAPPAs = 1 / np.tan(self.IC_ODs*np.pi/2)
            for kappa in IC_KAPPAs:
                signal_ic = noddi_ic.get_signal(self.dPar, kappa)
                for v_ic in self.IC_VFs:
                    signal_ec = noddi_ec.get_signal(self.dPar, kappa, v_ic)
                    signal = v_ic*signal_ic + (1-v_ic)*signal_ec
                    lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, False, ndirs )
                    np.save( pjoin( out_path, f'A_{idx+1:03d}.npy') , lm )
                    idx += 1
                    pbar.update()
            # Isotropic
            signal = noddi_iso.get_signal(self.dIso)
            lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, True, ndirs )
            np.save( pjoin( out_path, f'A_{nATOMS:03d}.npy') , lm )
            pbar.update()


    def resample( self, in_path, idx_out, Ylm_out, doMergeB0, ndirs ):
        nATOMS = len(self.IC_ODs)*len(self.IC_VFs) + 1
        if doMergeB0:
            nS = 1+self.scheme.dwi_count
            merge_idx = np.hstack((self.scheme.b0_idx[0],self.scheme.dwi_idx))
        else:
            nS = self.scheme.nS
            merge_idx = np.arange(nS)
        KERNELS = {}
        KERNELS['model'] = self.id
        KERNELS['wm']    = np.zeros( (nATOMS-1,ndirs,nS), dtype=np.float32 )
        KERNELS['iso']   = np.zeros( nS, dtype=np.float32 )
        KERNELS['kappa'] = np.zeros( nATOMS-1, dtype=np.float32 )
        KERNELS['icvf']  = np.zeros( nATOMS-1, dtype=np.float32 )
        KERNELS['norms'] = np.zeros( (self.scheme.dwi_count, nATOMS-1) )

        idx = 0
        with ProgressBar(total=nATOMS, disable=get_verbose()<3) as pbar:
            # Coupled contributions
            for i in range( len(self.IC_ODs) ):
                for j in range( len(self.IC_VFs) ):
                    lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                    if lm.shape[0] != ndirs:
                        ERROR( 'Outdated LUT. Call "generate_kernels( regenerate=True )" to update the LUT' )
                    KERNELS['wm'][idx,:,:] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, False, ndirs )[:,merge_idx]
                    KERNELS['kappa'][idx] = 1.0 / np.tan( self.IC_ODs[i]*np.pi/2.0 )
                    KERNELS['icvf'][idx]  = self.IC_VFs[j]
                    if doMergeB0:
                        KERNELS['norms'][:,idx] = 1 / np.linalg.norm( KERNELS['wm'][idx,0,1:] ) # norm of coupled atoms (for l1 minimization)
                    else:
                        KERNELS['norms'][:,idx] = 1 / np.linalg.norm( KERNELS['wm'][idx,0,self.scheme.dwi_idx] ) # norm of coupled atoms (for l1 minimization)
                    idx += 1
                    pbar.update()
            # Isotropic
            lm = np.load( pjoin( in_path, f'A_{nATOMS:03d}.npy' ) )
            KERNELS['iso'] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, True, ndirs )[merge_idx]
            pbar.update()

        return KERNELS


    def fit(self, evaluation):
        super().fit(evaluation)
        self.configs['compute_modulated_maps'] = evaluation.get_config('doSaveModulatedMaps')

        # fit chunks in parallel
        global _MULTITHREAD_PROGRESS
        _init_multithread_progress(evaluation.nthreads)
        with ProgressBar(total=evaluation.y.shape[0], multithread_progress=_MULTITHREAD_PROGRESS, disable=get_verbose()<3):
            with ThreadPoolExecutor(max_workers=evaluation.nthreads) as executor:
                futures = [executor.submit(self._fit, thread_id, evaluation.y[i:j, :], evaluation.DIRs[i:j, :], evaluation.htable, evaluation.KERNELS) for thread_id, (i, j) in enumerate(self.chunks)]
                chunked_results = [f.result() for f in futures]
        
        # concatenate results and return
        results = {}
        for k in chunked_results[0]:
            results[k] = np.concatenate([cr[k] for cr in chunked_results])
        return results


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit(self, thread_id, y, dirs, hash_table, kernels):
        # configs
        cdef int thread_id_c = thread_id
        cdef bint is_exvivo = 1 if self.isExvivo else 0
        cdef bint single_b0 = 1 if y.shape[1] == (1 + self.scheme.dwi_count) else 0
        cdef bint compute_rmse = 1 if self.configs['compute_rmse'] else 0
        cdef bint compute_nrmse = 1 if self.configs['compute_nrmse'] else 0
        cdef bint compute_modulated_maps = 1 if self.configs['compute_modulated_maps'] else 0
        cdef long long [::1]dwi_idx_view = self.scheme.dwi_idx
        cdef int n_wm = len(self.IC_ODs) * len(self.IC_VFs)
        cdef int n_atoms = n_wm + 1
        if is_exvivo:
            n_atoms += 1

        # solver params
        cdef double lambda1 = self.solver_params['lambda1']
        cdef double lambda2 = self.solver_params['lambda2']

        # directions
        cdef double [:, ::1]directions_view = np.ascontiguousarray(dirs, dtype=np.double)
        cdef short [::1]hash_table_view = hash_table
        cdef int lut_index

        # kernels
        cdef double [::1, :, :]kernels_wm_view = np.asfortranarray(np.swapaxes(kernels['wm'].T, 1, 2), dtype=np.double)
        cdef double [::1]kernels_iso_view = kernels['iso'].astype(np.double)
        cdef double [::1]kernels_exvivo_view
        if is_exvivo:
            kernels_exvivo_view = np.ones(kernels_wm_view.shape[0], dtype=np.double)
        cdef double [:, ::1]kernels_norms_view = np.ascontiguousarray(kernels['norms'], dtype=np.double)
        cdef float [::1]kernels_icvf_view = kernels['icvf']
        cdef float [::1]kernels_kappa_view = kernels['kappa']

        # y, A, x
        cdef double [:, ::1]y_view = np.ascontiguousarray(y, dtype=np.double)
        cdef double [::1, :]A_view = np.zeros((kernels_wm_view.shape[0], n_atoms), dtype=np.double, order='F')
        cdef double [::1]x_view = np.zeros(n_atoms, dtype=np.double)
        cdef double r_norm = 0.0

        # y_2, A_2
        cdef double [::1]y2_view = np.zeros(kernels_norms_view.shape[0], dtype=np.double)
        cdef double [::1, :]A2_view = np.zeros((kernels_norms_view.shape[0], kernels_norms_view.shape[1]), dtype=np.double, order='F')
        
        # A_3, x_3
        cdef double [::1, :]A3_view = np.zeros((kernels_wm_view.shape[0], n_atoms), dtype=np.double, order='F')
        cdef double [::1]x3_view = np.zeros(n_atoms, dtype=np.double)

        cdef int positive_count = 0
        cdef int [::1]positive_indices_view = np.zeros(n_atoms, dtype=np.intc)

        # return
        cdef double ndi = 0.0
        cdef double odi = 0.0
        cdef double fwf = 0.0
        estimates = np.zeros((y_view.shape[0], len(self.maps_name)), dtype=np.double, order='C')
        cdef double [:, ::1]estimates_view = estimates

        # support variables
        cdef double f1 = 0.0
        cdef double f2 = 0.0
        cdef double k1 = 0.0
        cdef double sum_n_atoms = 0.0
        cdef double sum_n_wm = 0.0

        # fitting error
        cdef double [::1]y_est_view
        cdef double [::1]rmse_view
        cdef double [::1]nrmse_view
        if compute_rmse or compute_nrmse:
            y_est_view = np.zeros(y_view.shape[0], dtype=np.double)
        if compute_rmse:
            rmse = np.zeros(y_view.shape[0], dtype=np.double)
            rmse_view = rmse
        if compute_nrmse:
            nrmse = np.zeros(y_view.shape[0], dtype=np.double)
            nrmse_view = nrmse

        # modulated maps
        cdef double [:, ::1]estimates_mod_view
        if compute_modulated_maps:
            estimates_mod = np.zeros((y_view.shape[0], 2), dtype=np.double, order='C')
            estimates_mod_view = estimates_mod
        cdef double tf = 0.0

        cdef Py_ssize_t i, j, k
        with nogil:
            for i in range(y_view.shape[0]):
                # prepare dictionary
                lut_index = dir_to_lut_idx(directions_view[i, :], hash_table_view)
                A_view[:, :n_wm] = kernels_wm_view[:, :, lut_index]
                if is_exvivo:
                    A_view[:, n_atoms-2] = kernels_exvivo_view
                A_view[:, n_atoms-1] = kernels_iso_view

                # fit_1 (CSF)
                nnls(&A_view[0, 0], &y_view[i, 0], A_view.shape[0], A_view.shape[1], &x_view[0], r_norm)

                # fit_2 (IC + EC)
                for j in range(A2_view.shape[0]):
                    for k in range(A2_view.shape[1]):
                        if single_b0:
                            A2_view[j, k] = A_view[j+1, k] * kernels_norms_view[j, k]
                            y2_view[j] = y_view[i, j+1] - x_view[n_atoms-1] * kernels_iso_view[j+1]
                        else:
                            A2_view[j, k] = A_view[dwi_idx_view[j], k] * kernels_norms_view[j, k]
                            y2_view[j] = y_view[i, dwi_idx_view[j]] - x_view[n_atoms-1] * kernels_iso_view[dwi_idx_view[j]]
                    if is_exvivo:
                        y2_view[j] = y2_view[j] - x_view[n_atoms-2] * 1.0
                    if y2_view[j] < 0.0:
                        y2_view[j] = 0.0
                lasso(&A2_view[0, 0], &y2_view[0], A2_view.shape[0], A2_view.shape[1], 1, &x_view[0], lambda1, lambda2)

                # fit_3 (debias coefficients)
                positive_count = 0
                if is_exvivo:
                    x_view[n_atoms-2] = 1.0
                x_view[n_atoms-1] = 1.0
                for j in range(n_atoms):
                    if x_view[j] > 0.0:
                        positive_indices_view[positive_count] = j
                        positive_count += 1
                for j in range(A3_view.shape[0]):
                    for k in range(positive_count):
                        A3_view[j, k] = A_view[j, positive_indices_view[k]]
                nnls(&A3_view[0, 0], &y_view[i, 0], A3_view.shape[0], positive_count, &x3_view[0], r_norm)
                for j in range(positive_count):
                    x_view[positive_indices_view[j]] = x3_view[j]

                # estimates
                f1 = 0.0
                f2 = 0.0
                k1 = 0.0
                sum_n_atoms = 0.0
                sum_n_wm = 0.0
                for j in range(n_atoms):
                    sum_n_atoms += x_view[j]
                sum_n_atoms += 1e-16
                for j in range(n_wm):
                    sum_n_wm += x_view[j] / sum_n_atoms
                sum_n_wm += 1e-16
                for j in range(n_wm):
                    f1 += kernels_icvf_view[j] * x_view[j] / sum_n_atoms / sum_n_wm
                    f2 += (<float> (1.0 - kernels_icvf_view[j])) * x_view[j] / sum_n_atoms / sum_n_wm
                    k1 += kernels_kappa_view[j] * x_view[j] / sum_n_atoms / sum_n_wm
                ndi = f1 / (f1 + f2 + 1e-16)
                odi = 2.0 / pi * atan2(1.0, k1)
                fwf = x_view[n_atoms-1] / sum_n_atoms
                estimates_view[i, 0] = ndi
                estimates_view[i, 1] = odi
                estimates_view[i, 2] = fwf
                if is_exvivo:
                    estimates_view[i, 3] = x_view[n_atoms-2] / sum_n_atoms

                # fitting error
                if compute_rmse:
                    _compute_rmse(A_view, y_view[i, :], x_view, &y_est_view[0], &rmse_view[i])
                if compute_nrmse:
                    _compute_nrmse(A_view, y_view[i, :], x_view, &y_est_view[0], &nrmse_view[i])

                # modulated maps
                if compute_modulated_maps:
                    tf = 1.0 - fwf
                    estimates_mod_view[i, 0] = ndi * tf
                    estimates_mod_view[i, 1] = odi * tf

                _update_multithread_progress(thread_id_c)

        results = {}
        results['estimates'] = estimates
        if compute_rmse:
            results['rmse'] = rmse
        if compute_nrmse:
            results['nrmse'] = nrmse
        if compute_modulated_maps:
            results['estimates_mod'] = estimates_mod
        return results



class FreeWater( BaseModel ) :
    """Implements the Free-Water model.
    """
    def __init__( self ) :
        self.id         = 'FreeWater'
        self.name       = 'Free-Water'
        self.set()


    def set(
        self,
        d_par=None,
        d_perps=None,
        d_isos=None,
        type='Human'
        ):
        '''
        Set the parameters of the Free-Water model.

        Parameters
        ----------
        d_par : float
            Parallel diffusivity [mm^2/s]
        d_perps : list of floats
            Perpendicular diffusivities [mm^2/s]
        d_isos : list of floats
            Isotropic diffusivities [mm^2/s]
        type : str
            Type of data ('Human' or 'Mouse')
        '''
        self.type = type
        if self.type == 'Mouse' :
            self.maps_name  = [ 'FiberVolume', 'FW', 'FW_blood', 'FW_csf' ]
            self.maps_descr = [ 'fiber volume fraction',
                                'Isotropic free-water volume fraction',
                                'FW blood', 'FW csf' ]
            if d_par is None:
                self.d_par = 1.0E-3
            else:
                self.d_par = d_par
            if d_perps is None:
                self.d_perps = np.linspace(0.15, 0.55, 10) * 1E-3
            else:
                self.d_perps = d_perps
            if d_isos is None:
                self.d_isos = [1.5E-3, 3E-3]
            else:
                self.d_isos = d_isos
        else :
            self.maps_name  = [ 'FiberVolume', 'FW' ]
            self.maps_descr = [ 'fiber volume fraction',
                                'Isotropic free-water volume fraction']
            if d_par is None:
                self.d_par = 1.0E-3
            else:
                self.d_par = d_par
            if d_perps is None:
                self.d_perps = np.linspace(0.1, 1.0, 10) * 1E-3
            else:
                self.d_perps = d_perps
            if d_isos is None:
                self.d_isos = [2.5E-3]
            else:
                self.d_isos = d_isos

        PRINT('      %s settings for Freewater elimination... ' % self.type)
        PRINT('             -iso  compartments: ', self.d_isos)
        PRINT('             -perp compartments: ', self.d_perps)
        PRINT('             -para compartments: ', self.d_par)


    def get_params( self ) :
        params = {}
        params['id'] = self.id
        params['name'] = self.name
        params['d_par'] = self.d_par
        params['d_perps'] = self.d_perps
        params['d_isos'] = self.d_isos
        params['type'] = self.type
        return params


    def set_solver( self, lambda1 = 0.0, lambda2 = 1e-3 ):
        super().set_solver()
        self.solver_params['lambda1'] = lambda1
        self.solver_params['lambda2'] = lambda2

        # TODO check this
        # need more regul for mouse data
        if self.type == 'Mouse' :
            lambda2 = 0.25


    def generate( self, out_path, aux, idx_in, idx_out, ndirs ) :
        scheme_high = amico.lut.create_high_resolution_scheme( self.scheme )
        zeppelin = Zeppelin(scheme_high)
        ball = Ball(scheme_high)

        nATOMS = len(self.d_perps) + len(self.d_isos)
        idx = 0
        with ProgressBar(total=nATOMS, disable=get_verbose()<3) as pbar:
            # Tensor compartment(s)
            for d in self.d_perps :
                signal = zeppelin.get_signal(self.d_par, d)
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, False, ndirs )
                np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
                idx += 1
                pbar.update()

            # Isotropic compartment(s)
            for d in self.d_isos :
                signal = ball.get_signal(d)
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, True, ndirs )
                np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
                idx += 1
                pbar.update()


    def resample( self, in_path, idx_out, Ylm_out, doMergeB0, ndirs ) :
        if doMergeB0:
            nS = 1+self.scheme.dwi_count
            merge_idx = np.hstack((self.scheme.b0_idx[0],self.scheme.dwi_idx))
        else:
            nS = self.scheme.nS
            merge_idx = np.arange(nS)
        KERNELS = {}
        KERNELS['model'] = self.id
        KERNELS['D']     = np.zeros( (len(self.d_perps),ndirs,nS), dtype=np.float32 )
        KERNELS['CSF']   = np.zeros( (len(self.d_isos),nS), dtype=np.float32 )

        nATOMS = len(self.d_perps) + len(self.d_isos)
        idx = 0
        with ProgressBar(total=nATOMS, disable=get_verbose()<3) as pbar:
            # Tensor compartment(s)
            for i in range(len(self.d_perps)) :
                lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                if lm.shape[0] != ndirs:
                    ERROR( 'Outdated LUT. Call "generate_kernels( regenerate=True )" to update the LUT' )
                KERNELS['D'][i,...] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, False, ndirs )[:,merge_idx]
                idx += 1
                pbar.update()

            # Isotropic compartment(s)
            for i in range(len(self.d_isos)) :
                lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                KERNELS['CSF'][i,...] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, True, ndirs )[merge_idx]
                idx += 1
                pbar.update()

        return KERNELS


    def fit(self, evaluation):
        super().fit(evaluation)
        self.configs['save_corrected_DWI'] = evaluation.get_config('doSaveCorrectedDWI')

        # fit chunks in parallel
        global _MULTITHREAD_PROGRESS
        _init_multithread_progress(evaluation.nthreads)
        with ProgressBar(total=evaluation.y.shape[0], multithread_progress=_MULTITHREAD_PROGRESS, disable=get_verbose()<3):
            with ThreadPoolExecutor(max_workers=evaluation.nthreads) as executor:
                futures = [executor.submit(self._fit, thread_id, evaluation.y[i:j, :], evaluation.DIRs[i:j, :], evaluation.htable, evaluation.KERNELS) for thread_id, (i, j) in enumerate(self.chunks)]
                chunked_results = [f.result() for f in futures]
        
        # concatenate results and return
        results = {}
        for k in chunked_results[0]:
            results[k] = np.concatenate([cr[k] for cr in chunked_results])
        return results


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit(self, thread_id, y, dirs, hash_table, kernels):
        # configs
        cdef int thread_id_c = thread_id
        cdef bint is_mouse = 1 if self.type == 'Mouse' else 0
        cdef bint compute_rmse = 1 if self.configs['compute_rmse'] else 0
        cdef bint compute_nrmse = 1 if self.configs['compute_nrmse'] else 0
        cdef bint save_corrected_DWI = 1 if self.configs['save_corrected_DWI'] else 0
        cdef int n_perp = len(self.d_perps)
        cdef int n_iso = len(self.d_isos)
        cdef int n_atoms = n_perp + n_iso

        # solver params
        cdef double lambda1 = self.solver_params['lambda1']
        cdef double lambda2 = self.solver_params['lambda2']

        # directions
        cdef double [:, ::1]directions_view = np.ascontiguousarray(dirs, dtype=np.double)
        cdef short [::1]hash_table_view = hash_table
        cdef int lut_index

        # kernels
        cdef double [::1, :, :]kernels_D_view = np.asfortranarray(np.swapaxes(kernels['D'].T, 1, 2), dtype=np.double)
        cdef double [::1, :]kernels_CSF_view = np.asfortranarray(kernels['CSF'].T, dtype=np.double)

        # y, A, x
        cdef double [:, ::1]y_view = np.ascontiguousarray(y, dtype=np.double)
        cdef double [::1, :]A_view = np.zeros((kernels_D_view.shape[0], n_atoms), dtype=np.double, order='F')
        cdef double [::1]x_view = np.zeros(n_atoms, dtype=np.double)

        # return
        cdef double v = 0.0
        cdef double v_blood = 0.0
        cdef double v_csf = 0.0
        estimates = np.zeros((y_view.shape[0], len(self.maps_name)), dtype=np.double, order='C')
        cdef double [:, ::1]estimates_view = estimates

        # support variables
        cdef double x_sum = 0.0
        cdef double x_n_perp_sum = 0.0

        # fitting error
        cdef double [::1]y_est_view
        cdef double [::1]rmse_view
        cdef double [::1]nrmse_view
        if compute_rmse or compute_nrmse:
            y_est_view = np.zeros(y_view.shape[0], dtype=np.double)
        if compute_rmse:
            rmse = np.zeros(y_view.shape[0], dtype=np.double)
            rmse_view = rmse
        if compute_nrmse:
            nrmse = np.zeros(y_view.shape[0], dtype=np.double)
            nrmse_view = nrmse

        # y_corrected
        cdef double [::1]y_fw_part
        cdef double [:, ::1]y_corrected_view
        if save_corrected_DWI:
            y_fw_part = np.zeros(y_view.shape[1], dtype=np.double)
            y_corrected = np.zeros((y_view.shape[0], y_view.shape[1]), dtype=np.double, order='C')
            y_corrected_view = y_corrected

        cdef Py_ssize_t i, j, k
        with nogil:
            for i in range(y_view.shape[0]):
                # prepare dictionary
                lut_index = dir_to_lut_idx(directions_view[i, :], hash_table_view)
                A_view[:, :n_perp] = kernels_D_view[:, :, lut_index]
                A_view[:, n_perp:] = kernels_CSF_view[:, :]

                # fit
                lasso(&A_view[0, 0], &y_view[i, 0], A_view.shape[0], A_view.shape[1], 1, &x_view[0], lambda1, lambda2)

                # estimates
                x_sum = 0.0
                x_n_perp_sum = 0.0
                for j in range(x_view.shape[0]):
                    x_sum += x_view[j]
                    if j < n_perp:
                        x_n_perp_sum += x_view[j]
                x_sum += 1e-16   
                v = x_n_perp_sum / x_sum
                estimates_view[i, 0] = v
                estimates_view[i, 1] = 1.0 - v
                if is_mouse:
                    v_blood = x_view[n_perp] / x_sum
                    v_csf = x_view[n_perp + 1] / x_sum
                    estimates_view[i, 2] = v_blood
                    estimates_view[i, 3] = v_csf

                # fitting error
                if compute_rmse:
                    _compute_rmse(A_view, y_view[i, :], x_view, &y_est_view[0], &rmse_view[i])
                if compute_nrmse:
                    _compute_nrmse(A_view, y_view[i, :], x_view, &y_est_view[0], &nrmse_view[i])

                # y_corrected
                if save_corrected_DWI:
                    for j in range(x_view.shape[0] - n_iso):
                        x_view[j] = 0.0
                    for j in range(A_view.shape[0]):
                        y_fw_part[j] = 0.0
                        for k in range(A_view.shape[1]):
                            y_fw_part[j] += A_view[j, k] * x_view[k]
                    for j in range(y_fw_part.shape[0]):
                        y_corrected_view[i, j] = y_view[i, j] - y_fw_part[j]
                        if y_corrected_view[i, j] < 0.0:
                            y_corrected_view[i, j] = 0.0

                _update_multithread_progress(thread_id_c)

        results = {}
        results['estimates'] = estimates
        if compute_rmse:
            results['rmse'] = rmse
        if compute_nrmse:
            results['nrmse'] = nrmse
        if save_corrected_DWI:
            results['y_corrected'] = y_corrected
        return results



class VolumeFractions( BaseModel ) :
    """Implements a simple model where each compartment contributes only with
       its own volume fraction. This model has been created to test there
       ability to remove false positive fibers with COMMIT.
    """
    def __init__( self ) :
        self.id         = 'VolumeFractions'
        self.name       = 'Volume fractions'
        self.maps_name  = [ ]
        self.maps_descr = [ ]
        self.set()


    def set( self ) :
        return


    def get_params( self ) :
        params = {}
        params['id'] = self.id
        params['name'] = self.name
        return params


    def set_solver( self ) :
        ERROR( 'Not implemented' )


    def generate( self, out_path, aux, idx_in, idx_out, ndirs ) :
        return


    def resample( self, in_path, idx_out, Ylm_out, doMergeB0, ndirs ) :
        if doMergeB0:
            nS = 1+self.scheme.dwi_count
            merge_idx = np.hstack((self.scheme.b0_idx[0],self.scheme.dwi_idx))
        else:
            nS = self.scheme.nS
            merge_idx = np.arange(nS)

        KERNELS = {}
        KERNELS['model'] = self.id
        KERNELS['wmr']   = np.ones( (1,ndirs,nS), dtype=np.float32 )
        KERNELS['wmh']   = np.ones( (0,ndirs,nS), dtype=np.float32 )
        KERNELS['iso']   = np.ones( (0,nS), dtype=np.float32 )

        return KERNELS


    def fit(self, evaluation):
        ERROR( 'Not implemented' )



class SANDI( BaseModel ) :
    """Implements the SANDI model [1].

    The intra-cellular contributions from within the neural cells are modeled as intra-soma + intra-neurite,
    with the soma modelled as "sphere" of radius (Rs) and fixed intra-soma diffusivity (d_is) to 3 micron^2/ms;
    the neurites are modelled as randomly oriented sticks with axial intra-neurite diffusivity (d_in).
    Extra-cellular contributions are modeled as isotropic gaussian diffusion, i.e. "ball", with the mean diffusivity (d_iso)

    NB: this model works only with direction-averaged signal and schemes containing the full specification of
        the diffusion gradients (eg gradient strength, small delta etc).

    References
    ----------
    .. [1] Palombo, Marco, et al. "SANDI: a compartment-based model for non-invasive apparent soma and neurite imaging by diffusion MRI." Neuroimage 215 (2020): 116835.
    """
    def __init__( self ) :
        self.id         = 'SANDI'
        self.name       = 'SANDI'
        self.maps_name  = [ 'fsoma', 'fneurite', 'fextra', 'Rsoma', 'Din', 'De' ]
        self.maps_descr = [ 'Intra-soma volume fraction', 'Intra-neurite volume fraction', 'Extra-cellular volume fraction', 'Apparent soma radius', 'Neurite axial diffusivity', 'Extra-cellular mean diffusivity' ]
        self.set()


    def set(
        self,
        d_is=3.0E-3,
        Rs=np.linspace(1.0, 12.0, 5) * 1E-6,
        d_in=np.linspace(0.25, 3.0, 5) * 1E-3,
        d_isos=np.linspace(0.25, 3.0, 5) * 1E-3
        ):
        '''
        Set the parameters of the SANDI model.

        Parameters
        ----------
        d_is : float
            Intra-soma diffusivity [mm^2/s]
        Rs : list of floats
            Radii of the soma [meters]
        d_in : list of floats
            Intra-neurite diffusivities [mm^2/s]
        d_isos : list of floats
            Extra-cellular isotropic mean diffusivities [mm^2/s]
        '''
        self.d_is   = d_is
        self.Rs     = np.array(Rs)
        self.d_in   = np.array(d_in)
        self.d_isos = np.array(d_isos)


    def get_params( self ) :
        params = {}
        params['id']     = self.id
        params['name']   = self.name
        params['d_is']   = self.d_is
        params['Rs']     = self.Rs
        params['d_in']   = self.d_in
        params['d_isos'] = self.d_isos
        return params


    def set_solver( self, lambda1 = 0.0, lambda2 = 5.0E-3 ) :
        super().set_solver()
        self.solver_params['lambda1'] = lambda1
        self.solver_params['lambda2'] = lambda2


    def generate( self, out_path, aux, idx_in, idx_out, ndirs ) :
        if self.scheme.version != 1 :
            ERROR( 'This model requires a "VERSION: STEJSKALTANNER" scheme' )

        scheme_high = amico.lut.create_high_resolution_scheme( self.scheme )
        sphere = SphereGPD(scheme_high)
        astrosticks = Astrosticks(scheme_high)
        ball = Ball(scheme_high)

        nATOMS = len(self.Rs) + len(self.d_in) + len(self.d_isos)
        idx = 0
        with ProgressBar(total=nATOMS, disable=get_verbose()<3) as pbar:
            # Soma = SPHERE
            for R in self.Rs :
                signal = sphere.get_signal(self.d_is, R)
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, True, ndirs )
                np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
                idx += 1
                pbar.update()
            # Neurites = ASTRO STICKS
            for d in self.d_in :
                signal = astrosticks.get_signal(d)
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, True, ndirs )
                np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
                idx += 1
                pbar.update()
            # Extra-cellular = BALL
            for d in self.d_isos :
                signal = ball.get_signal(d)
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, True, ndirs )
                np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
                idx += 1
                pbar.update()


    def resample( self, in_path, idx_out, Ylm_out, doMergeB0, ndirs ) :
        nATOMS = len(self.Rs) + len(self.d_in) + len(self.d_isos)
        if doMergeB0:
            nS = 1+self.scheme.dwi_count
            merge_idx = np.hstack((self.scheme.b0_idx[0],self.scheme.dwi_idx))
        else:
            nS = self.scheme.nS
            merge_idx = np.arange(nS)
        KERNELS = {}
        KERNELS['model']  = self.id
        KERNELS['signal'] = np.zeros( (nS,nATOMS), dtype=np.float64, order='F' )
        KERNELS['norms']  = np.zeros( nATOMS, dtype=np.float64 )

        idx = 0
        with ProgressBar(total=nATOMS, disable=get_verbose()<3) as pbar:
            # Soma = SPHERE
            for i in range(len(self.Rs)) :
                lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                signal = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, True, ndirs )[merge_idx].T
                KERNELS['norms'][idx] = 1.0 / np.linalg.norm( signal )
                KERNELS['signal'][:,idx] = signal * KERNELS['norms'][idx]
                idx += 1
                pbar.update()
            # Neurites = STICKS
            for i in range(len(self.d_in)) :
                lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                signal = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, True, ndirs )[merge_idx].T
                KERNELS['norms'][idx] = 1.0 / np.linalg.norm( signal )
                KERNELS['signal'][:,idx] = signal * KERNELS['norms'][idx]
                idx += 1
                pbar.update()
            # Extra-cellular = BALL
            for i in range(len(self.d_isos)) :
                lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                signal = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, True, ndirs )[merge_idx].T
                KERNELS['norms'][idx] = 1.0 / np.linalg.norm( signal )
                KERNELS['signal'][:,idx] = signal * KERNELS['norms'][idx]
                idx += 1
                pbar.update()

        return KERNELS

    
    def fit(self, evaluation):
        super().fit(evaluation)

        # fit chunks in parallel
        global _MULTITHREAD_PROGRESS
        _init_multithread_progress(evaluation.nthreads)
        with ProgressBar(total=evaluation.y.shape[0], multithread_progress=_MULTITHREAD_PROGRESS, disable=get_verbose()<3):
            with ThreadPoolExecutor(max_workers=evaluation.nthreads) as executor:
                futures = [executor.submit(self._fit, thread_id, evaluation.y[i:j, :], evaluation.KERNELS) for thread_id, (i, j) in enumerate(self.chunks)]
                chunked_results = [f.result() for f in futures]
        
        # concatenate results and return
        results = {}
        for k in chunked_results[0]:
            results[k] = np.concatenate([cr[k] for cr in chunked_results])
        return results


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit(self, thread_id, y, kernels):
        # configs
        cdef int thread_id_c = thread_id
        cdef bint compute_rmse = 1 if self.configs['compute_rmse'] else 0
        cdef bint compute_nrmse = 1 if self.configs['compute_nrmse'] else 0
        cdef int n_rs = len(self.Rs)
        cdef int n_in = len(self.d_in)
        cdef int n_iso = len(self.d_isos)
        cdef int n_atoms = n_rs + n_in + n_iso

        # solver params
        cdef double lambda1 = self.solver_params['lambda1']
        cdef double lambda2 = self.solver_params['lambda2']

        # kernels
        cdef double [::1]kernels_norms_view = kernels['norms']

        # y, A, x
        cdef double [:, ::1]y_view = np.ascontiguousarray(y, dtype=np.double)
        cdef double [::1, :]A_view = np.asfortranarray(kernels['signal'], dtype=np.double)
        cdef double [::1]x_view = np.zeros(n_atoms, dtype=np.double)

        # return
        cdef double fsoma = 0.0
        cdef double fneurite = 0.0
        cdef double fextra = 0.0
        cdef double Rsoma = 0.0
        cdef double Din = 0.0
        cdef double De = 0.0
        estimates = np.zeros((y_view.shape[0], len(self.maps_name)), dtype=np.double, order='C')
        cdef double [:, ::1] estimates_view = estimates

        # support variables
        cdef double [::1]rs_view = self.Rs
        cdef double [::1]d_in_view = self.d_in
        cdef double [::1]d_isos_view = self.d_isos
        cdef int n1 = rs_view.shape[0]
        cdef int n2 = d_in_view.shape[0]
        cdef double x_sum = 0.0
        cdef double xsph_sum = 0.0
        cdef double xstk_sum = 0.0
        cdef double xiso_sum = 0.0

        # fitting error
        cdef double [::1]y_est_view
        cdef double [::1]rmse_view
        cdef double [::1]nrmse_view
        if compute_rmse or compute_nrmse:
            y_est_view = np.zeros(y_view.shape[0], dtype=np.double)
        if compute_rmse:
            rmse = np.zeros(y_view.shape[0], dtype=np.double)
            rmse_view = rmse
        if compute_nrmse:
            nrmse = np.zeros(y_view.shape[0], dtype=np.double)
            nrmse_view = nrmse

        cdef Py_ssize_t i, j
        with nogil:
            for i in range(y_view.shape[0]):
                # fit
                lasso(&A_view[0, 0], &y_view[i, 0], A_view.shape[0], A_view.shape[1], 1, &x_view[0], lambda1, lambda2)
                for j in range(kernels_norms_view.shape[0]):
                    x_view[j] = x_view[j] * kernels_norms_view[j]

                # return estimates
                x_sum = 0.0
                xsph_sum = 0.0
                xstk_sum = 0.0
                xiso_sum = 0.0
                Rsoma = 0.0
                Din = 0.0
                De = 0.0
                for j in range(A_view.shape[1]):
                    x_sum += x_view[j]
                    if j < n_rs:
                        xsph_sum += x_view[j]
                    if j >= n_rs and j < n_rs + n_in:
                        xstk_sum += x_view[j]
                    if j >= n_rs + n_in:
                         xiso_sum += x_view[j]
                x_sum += 1e-16
                fsoma = xsph_sum / x_sum
                fneurite = xstk_sum / x_sum
                fextra = xiso_sum / x_sum
                for j in range(A_view.shape[1]):
                    if j < n_rs:
                        Rsoma += rs_view[j] * x_view[j]
                    if j >= n_rs and j < n_rs + n_in:
                        Din += d_in_view[j - n_rs] * x_view[j]
                    if j >= n_rs + n_in:
                         De += d_isos_view[j - (n_rs + n_in)] * x_view[j]
                xsph_sum += 1e-16
                xstk_sum += 1e-16
                xiso_sum += 1e-16
                Rsoma = 1e6 * Rsoma / xsph_sum
                Din = 1e3 * Din / xstk_sum
                De = 1e3 * De / xiso_sum
                estimates_view[i, 0] = fsoma
                estimates_view[i, 1] = fneurite
                estimates_view[i, 2] = fextra
                estimates_view[i, 3] = Rsoma
                estimates_view[i, 4] = Din
                estimates_view[i, 5] = De

                # fitting error
                if compute_rmse:
                    _compute_rmse(A_view, y_view[i, :], x_view, &y_est_view[0], &rmse_view[i])
                if compute_nrmse:
                    _compute_nrmse(A_view, y_view[i, :], x_view, &y_est_view[0], &nrmse_view[i])

                _update_multithread_progress(thread_id_c)

        results = {}
        results['estimates'] = estimates
        if compute_rmse:
            results['rmse'] = rmse
        if compute_nrmse:
            results['nrmse'] = nrmse
        return results
