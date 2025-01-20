import numpy as np
import time
import glob
import sys
from os import makedirs, remove, cpu_count
from os.path import exists, join as pjoin, isfile
import inspect

import nibabel
import pickle
import amico.scheme
from amico.preproc import debiasRician
import amico.lut
import amico.models
from amico.lut import is_valid, valid_dirs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from amico.util import PRINT, LOG, WARNING, ERROR, get_verbose
from dicelib.ui import ProgressBar
from pkg_resources import get_distribution
from threadpoolctl import ThreadpoolController

def setup( lmax=12 ) :
    """General setup/initialization of the AMICO framework.

    Parameters
    ----------
    lmax : int
        Maximum SH order to use for the rotation phase (default : 12).
        NB: change only if you know what you are doing.
    """
    LOG( '\n-> Precomputing rotation matrices:' )
    dirs = valid_dirs()
    with ProgressBar(total=len(dirs), disable=get_verbose()<3) as pbar:
        for dir in dirs:
            amico.lut.precompute_rotation_matrices(lmax, dir)
            pbar.update()
    LOG('   [ DONE ]')



class Evaluation :
    """Class to hold all the information (data and parameters) when performing an
    evaluation with the AMICO framework.
    """

    def __init__( self, study_path='.', subject='.', output_path=None ) :
        """Setup the data structure with default values.

        Parameters
        ----------
        study_path : string
            The path to the folder containing all the subjects from one study
        subject : string
            The path (relative to previous folder) to the subject folder
        output_path : string
            Optionally sets a custom full path for the output. Leave as None
            for default behaviour - output in study_path/subject/AMICO/<MODEL>
        """
        self.niiDWI        = None    # set by "load_data" method
        self.niiDWI_img    = None
        self.scheme        = None
        self.niiMASK       = None
        self.niiMASK_img   = None
        self.model         = None    # set by "set_model" method
        self.KERNELS       = None    # set by "load_kernels" method
        self.y             = None    # set by "fit" method
        self.DIRs          = None    # set by "fit" method
        self.nthreads      = None    # set by "fit" method
        self.BLAS_nthreads = None    # set by "generate_kernel", "load_kernels" and "fit" methods
        self.RESULTS       = None    # set by "fit" method
        self.mean_b0s      = None    # set by "load_data" method
        self.htable        = None

        # store all the parameters of an evaluation with AMICO
        self.CONFIG = {}
        self.set_config('version', get_distribution('dmri-amico').version)
        self.set_config('study_path', study_path)
        self.set_config('subject', subject)
        self.set_config('DATA_path', pjoin( study_path, subject ))
        self.set_config('OUTPUT_path', output_path)

        self.set_config('peaks_filename', None)
        self.set_config('doNormalizeSignal', True)
        self.set_config('doKeepb0Intact', False)       # does change b0 images in the predicted signal
        self.set_config('doComputeRMSE', False)
        self.set_config('doComputeNRMSE', False)
        self.set_config('doSaveModulatedMaps', False)  # NODDI model specific config
        self.set_config('doSaveCorrectedDWI', False)   # FreeWater model specific config
        self.set_config('doMergeB0', False)            # Merge b0 volumes
        self.set_config('doDebiasSignal', False)       # Flag to remove Rician bias
        self.set_config('DWI-SNR', None)               # SNR of DWI image: SNR = b0/sigma
        self.set_config('doDirectionalAverage', False) # To perform the directional average on the signal of each shell
        self.set_config('nthreads', -1)                # Number of threads to be used in multithread-enabled parts of code (default: -1)
        self.set_config('DTI_fit_method', 'OLS')       # Fit method for the Diffusion Tensor model (dipy) (default: 'OLS')
        self.set_config('BLAS_nthreads', 1)            # Number of threads used in the threadpool-backend of common BLAS implementations (dafault: 1)

        self._controller = ThreadpoolController()

    def set_config( self, key, value ) :
        self.CONFIG[ key ] = value

    def get_config( self, key ) :
        return self.CONFIG.get( key )


    def load_data( self, dwi_filename='DWI.nii', scheme_filename='DWI.scheme', mask_filename=None, b0_thr=0, b0_min_signal=0, replace_bad_voxels=None ) :
        """Load the diffusion signal and its corresponding acquisition scheme.

        Parameters
        ----------
        dwi_filename : string
            The file name of the DWI data, relative to the subject folder (default : 'DWI.nii')
        scheme_filename : string
            The file name of the corresponding acquisition scheme (default : 'DWI.scheme')
        mask_filename : string
            The file name of the (optional) binary mask (default : None)
        b0_thr : float
            The threshold below which a b-value is considered a b0 (default : 0)
        b0_min_signal : float, optional
            Crop to zero the signal in voxels where the b0 <= b0_min_signal * mean(b0[b0>0]). (default : 0)
        replace_bad_voxels : float, optional
            Value to be used to fill NaN and Inf values in the signal. (default : do nothing)
        """
        # Loading data, acquisition scheme and mask (optional)
        LOG( '\n-> Loading data:' )
        tic = time.time()

        PRINT('\t* DWI signal')
        if not isfile( pjoin(self.get_config('DATA_path'), dwi_filename) ):
            ERROR( 'DWI file not found' )
        self.set_config('dwi_filename', dwi_filename)
        self.set_config('b0_min_signal', b0_min_signal)
        self.set_config('replace_bad_voxels', replace_bad_voxels)
        self.niiDWI = nibabel.load( pjoin(self.get_config('DATA_path'), dwi_filename) )
        self.niiDWI_img = self.niiDWI.get_fdata().astype(np.float32)
        hdr = self.niiDWI.header if nibabel.__version__ >= '2.0.0' else self.niiDWI.get_header()
        if self.niiDWI_img.ndim != 4 :
            ERROR( 'DWI file is not a 4D image' )
        self.set_config('dim', self.niiDWI_img.shape[:3])
        self.set_config('pixdim', tuple( hdr.get_zooms()[:3] ))
        PRINT('\t\t- dim    = %d x %d x %d x %d' % self.niiDWI_img.shape)
        PRINT('\t\t- pixdim = %.3f x %.3f x %.3f' % self.get_config('pixdim'))

        # Scale signal intensities (if necessary)
        if ( np.isfinite(hdr['scl_slope']) and np.isfinite(hdr['scl_inter']) and hdr['scl_slope'] != 0 and
            ( hdr['scl_slope'] != 1 or hdr['scl_inter'] != 0 ) ):
            PRINT('\t\t- rescaling data ', end='')
            self.niiDWI_img = self.niiDWI_img * hdr['scl_slope'] + hdr['scl_inter']
            PRINT('[OK]')

        # Check for Nan or Inf values in raw data
        if np.isnan(self.niiDWI_img).any() or np.isinf(self.niiDWI_img).any():
            if replace_bad_voxels is not None:
                WARNING(f'Nan or Inf values in the raw signal. They will be replaced with: {replace_bad_voxels}')
                np.nan_to_num(self.niiDWI_img, copy=False, nan=replace_bad_voxels, posinf=replace_bad_voxels, neginf=replace_bad_voxels)
            else:
                ERROR('Nan or Inf values in the raw signal. Try using the "replace_bad_voxels" or "b0_min_signal" parameters when calling "load_data()"')

        PRINT('\t* Acquisition scheme')
        if not isfile( pjoin(self.get_config('DATA_path'), scheme_filename) ):
            ERROR( 'SCHEME file not found' )
        self.set_config('scheme_filename', scheme_filename)
        self.set_config('b0_thr', b0_thr)
        self.scheme = amico.scheme.Scheme( pjoin( self.get_config('DATA_path'), scheme_filename), b0_thr )
        PRINT(f'\t\t- {self.scheme.nS} samples, {len(self.scheme.shells)} shells')
        PRINT(f'\t\t- {self.scheme.b0_count} @ b=0', end=' ')
        for i in range(len(self.scheme.shells)) :
            PRINT(f', {len(self.scheme.shells[i]["idx"])} @ b={self.scheme.shells[i]["b"]:.1f}', end=' ')
        PRINT()

        if self.scheme.nS != self.niiDWI_img.shape[3] :
            ERROR( 'Scheme does not match with DWI data' )

        PRINT('\t* Binary mask')
        if mask_filename is not None :
            if not isfile( pjoin(self.get_config('DATA_path'), mask_filename) ):
                ERROR( 'MASK file not found' )
            self.niiMASK = nibabel.load( pjoin( self.get_config('DATA_path'), mask_filename) )
            self.niiMASK_img = self.niiMASK.get_fdata().astype(np.uint8)
            niiMASK_hdr = self.niiMASK.header if nibabel.__version__ >= '2.0.0' else self.niiMASK.get_header()
            PRINT('\t\t- dim    = %d x %d x %d' % self.niiMASK_img.shape[:3])
            PRINT('\t\t- pixdim = %.3f x %.3f x %.3f' % niiMASK_hdr.get_zooms()[:3])
            if self.niiMASK.ndim != 3 :
                ERROR( 'MASK file is not a 3D image' )
            if self.get_config('dim') != self.niiMASK_img.shape[:3] :
                ERROR( 'MASK geometry does not match with DWI data' )
        else :
            self.niiMASK = None
            self.niiMASK_img = np.ones( self.get_config('dim') )
            PRINT('\t\t- not specified')
        self.set_config('mask_filename', mask_filename)
        PRINT(f'\t\t- voxels = {np.count_nonzero(self.niiMASK_img)}')

        LOG( f'   [ {time.time() - tic:.1f} seconds ]' )

        # Preprocessing
        LOG( '\n-> Preprocessing:' )
        tic = time.time()

        if self.get_config('doDebiasSignal') :
            PRINT('\t* Debiasing signal... ', end='')
            sys.stdout.flush()
            if self.get_config('DWI-SNR') == None:
                ERROR( "Set noise variance for debiasing (eg. ae.set_config('RicianNoiseSigma', sigma))" )
            self.niiDWI_img = debiasRician(self.niiDWI_img,self.get_config('DWI-SNR'),self.niiMASK_img,self.scheme)
            PRINT(' [OK]')

        if self.get_config('doNormalizeSignal') :
            PRINT('\t* Normalizing to b0... ', end='')
            sys.stdout.flush()
            if self.scheme.b0_count > 0 :
                self.mean_b0s = np.mean( self.niiDWI_img[:,:,:,self.scheme.b0_idx], axis=3 )
            else:
                ERROR( 'No b0 volume to normalize signal with' )
            norm_factor = self.mean_b0s.copy()
            idx = norm_factor <= b0_min_signal * norm_factor[norm_factor > 0].mean()
            norm_factor[ idx ] = 1
            norm_factor = 1 / norm_factor
            norm_factor[ idx ] = 0
            for i in range(self.scheme.nS) :
                self.niiDWI_img[:,:,:,i] *= norm_factor
            PRINT(f'[ min={self.niiDWI_img.min():.2f},  mean={self.niiDWI_img.mean():.2f}, max={self.niiDWI_img.max():.2f} ]')

        if self.get_config('doMergeB0') :
            PRINT('\t* Merging multiple b0 volume(s)')
            mean = np.expand_dims( np.mean( self.niiDWI_img[:,:,:,self.scheme.b0_idx], axis=3 ), axis=3 )
            self.niiDWI_img = np.concatenate( (mean, self.niiDWI_img[:,:,:,self.scheme.dwi_idx]), axis=3 )
        else :
            PRINT('\t* Keeping all b0 volume(s)')

        if self.get_config('doDirectionalAverage') :
            PRINT('\t* Performing the directional average on the signal of each shell... ')
            numShells = len(self.scheme.shells)
            dir_avg_img = self.niiDWI_img[:,:,:,:(numShells + 1)]
            scheme_table = np.zeros([numShells + 1, 7])

            id_bval = 0
            dir_avg_img[:,:,:,id_bval] = np.mean( self.niiDWI_img[:,:,:,self.scheme.b0_idx], axis=3 )
            scheme_table[id_bval, : ] = np.array([1, 0, 0, 0, 0, 0, 0])

            bvals = []
            for shell in self.scheme.shells:
                bvals.append(shell['b'])

            sort_idx = np.argsort(bvals)

            for shell_idx in sort_idx:
                shell = self.scheme.shells[shell_idx]
                id_bval = id_bval + 1
                dir_avg_img[:,:,:,id_bval] = np.mean( self.niiDWI_img[:,:,:,shell['idx']], axis=3 )
                scheme_table[id_bval, : ] = np.array([1, 0, 0, shell['G'], shell['Delta'], shell['delta'], shell['TE']])

            self.niiDWI_img = dir_avg_img.astype(np.float32)
            self.set_config('dim', self.niiDWI_img.shape[:3])
            PRINT('\t\t- dim    = %d x %d x %d x %d' % self.niiDWI_img.shape)
            PRINT('\t\t- pixdim = %.3f x %.3f x %.3f' % self.get_config('pixdim'))

            PRINT('\t* Acquisition scheme')
            self.scheme = amico.scheme.Scheme( scheme_table, b0_thr )
            PRINT(f'\t\t- {self.scheme.nS} samples, {len(self.scheme.shells)} shells')
            PRINT(f'\t\t- {self.scheme.b0_count} @ b=0', end=' ')
            for i in range(len(self.scheme.shells)) :
                PRINT(f', {len(self.scheme.shells[i]["idx"])} @ b={self.scheme.shells[i]["b"]:.1f}', end=' ')
            PRINT()

            if self.scheme.nS != self.niiDWI_img.shape[3] :
                ERROR( 'Scheme does not match with DWI data' )

        # Check for Nan or Inf values in pre-processed data
        if np.isnan(self.niiDWI_img).any() or np.isinf(self.niiDWI_img).any():
            if replace_bad_voxels is not None:
                WARNING(f'Nan or Inf values in the signal after the pre-processing. They will be replaced with: {replace_bad_voxels}')
                np.nan_to_num(self.niiDWI_img, copy=False, nan=replace_bad_voxels, posinf=replace_bad_voxels, neginf=replace_bad_voxels)
            else:
                ERROR('Nan or Inf values in the signal after the pre-processing. Try using the "replace_bad_voxels" or "b0_min_signal" parameters when calling "load_data()"')

        LOG( f'   [ {time.time() - tic:.1f} seconds ]' )


    def set_model( self, model_name ) :
        """Set the model to use to describe the signal contributions in each voxel.

        Parameters
        ----------
        model_name : string
            The name of the model (must match a class name in "amico.models" module)
        """
        # Call the specific model constructor
        if hasattr(amico.models, model_name ) :
            self.model = getattr(amico.models,model_name)()
        else :
            ERROR( f'Model "{model_name}" not recognized' )

        self.set_config('ATOMS_path', pjoin( self.get_config('study_path'), 'kernels', self.model.id ))

        # setup default parameters for fitting the model (can be changed later on)
        self.set_solver()


    def set_solver( self, **params ) :
        """Set up the specific parameters of the solver to fit the model.
        Dispatch to the proper function, depending on the model; a model should provide a "set_solver" function to set these parameters.
        Currently supported parameters are:
        StickZeppelinBall:      'set_solver()' not implemented
        CylinderZeppelinBall:   lambda1 = 0.0, lambda2 = 4.0
        NODDI:                  lambda1 = 5e-1, lambda2 = 1e-3
        FreeWater:              lambda1 = 0.0, lambda2 = 1e-3
        VolumeFractions:        'set_solver()' not implemented
        SANDI:                  lambda1 = 0.0, lambda2 = 5e-3
        NOTE: non-existing parameters will be ignored
        """
        if self.model is None :
            ERROR( 'Model not set; call "set_model()" method first' )
        
        solver_params = list(inspect.signature(self.model.set_solver).parameters)
        params_new = {}
        for key in params.keys():
            if key not in solver_params:
                WARNING(f"Cannot find the '{key}' solver-parameter for the {self.model.name} model. It will be ignored")
            else:
                params_new[key] = params[key]

        self.model.set_solver(**params_new)
        self.set_config('solver_params', params_new)


    def generate_kernels( self, regenerate = False, lmax = 12, ndirs = 500 ) :
        """Generate the high-resolution response functions for each compartment.
        Dispatch to the proper function, depending on the model.

        Parameters
        ----------
        regenerate : boolean
            Regenerate kernels if they already exist (default : False)
        lmax : int
            Maximum SH order to use for the rotation procedure (default : 12)
        ndirs : int
            Number of directions on the half of the sphere representing the possible orientations of the response functions (default : 500)
        """
        if self.scheme is None :
            ERROR( 'Scheme not loaded; call "load_data()" first' )
        if self.model is None :
            ERROR( 'Model not set; call "set_model()" method first' )
        if not is_valid(ndirs):
            ERROR( 'Unsupported value for ndirs.\nNote: Supported values for ndirs are [1, 500 (default), 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 32761]' )
        
        self.BLAS_nthreads = self.get_config('BLAS_nthreads') if self.get_config('BLAS_nthreads') > 0 else cpu_count() if self.get_config('BLAS_nthreads') == -1 else ERROR('Number of BLAS threads must be positive or -1')

        # store some values for later use
        self.set_config('lmax', lmax)
        self.set_config('ndirs', ndirs)
        self.model.scheme = self.scheme
        LOG( f'\n-> Creating LUT for "{self.model.name}" model:' )

        # check if kernels were already generated
        tmp = glob.glob( pjoin(self.get_config('ATOMS_path'),'A_*.npy') )
        if len(tmp)>0 and not regenerate :
            LOG( '   [ LUT already computed. Use option "regenerate=True" to force regeneration ]' )
            return

        # create folder or delete existing files (if any)
        if not exists( self.get_config('ATOMS_path') ) :
            makedirs( self.get_config('ATOMS_path') )
        else :
            for f in glob.glob( pjoin(self.get_config('ATOMS_path'),'*') ) :
                remove( f )

        # auxiliary data structures
        aux = amico.lut.load_precomputed_rotation_matrices( lmax, ndirs )
        idx_IN, idx_OUT = amico.lut.aux_structures_generate( self.scheme, lmax )

        # Dispatch to the right handler for each model
        tic = time.time()
        with self._controller.limit(limits=self.BLAS_nthreads, user_api='blas'):
            self.model.generate( self.get_config('ATOMS_path'), aux, idx_IN, idx_OUT, ndirs )
        LOG( f'   [ {time.time() - tic:.1f} seconds ]' )


    def load_kernels( self ) :
        """Load rotated kernels and project to the specific gradient scheme of this subject.
        Dispatch to the proper function, depending on the model.
        """
        if self.model is None :
            ERROR( 'Model not set; call "set_model()" method first' )
        if self.scheme is None :
            ERROR( 'Scheme not loaded; call "load_data()" first' )
        
        self.BLAS_nthreads = self.get_config('BLAS_nthreads') if self.get_config('BLAS_nthreads') > 0 else cpu_count() if self.get_config('BLAS_nthreads') == -1 else ERROR('Number of BLAS threads must be positive or -1')

        tic = time.time()
        LOG( f'\n-> Resampling LUT for subject "{self.get_config("subject")}":' )

        # auxiliary data structures
        idx_OUT, Ylm_OUT = amico.lut.aux_structures_resample( self.scheme, self.get_config('lmax') )

        # hash table
        self.htable = amico.lut.load_precomputed_hash_table( self.get_config('ndirs') )

        # Dispatch to the right handler for each model
        with self._controller.limit(limits=self.BLAS_nthreads, user_api='blas'):
            self.KERNELS = self.model.resample( self.get_config('ATOMS_path'), idx_OUT, Ylm_OUT, self.get_config('doMergeB0'), self.get_config('ndirs') )

        LOG( f'   [ {time.time() - tic:.1f} seconds ]')


    def fit( self ) :
        """Fit the model to the data.
        Call the appropriate fit() method of the actual model used.
        """
        if self.niiDWI is None :
            ERROR( 'Data not loaded; call "load_data()" first' )
        if self.model is None :
            ERROR( 'Model not set; call "set_model()" first' )
        if self.KERNELS is None :
            ERROR( 'Response functions not generated; call "generate_kernels()" and "load_kernels()" first' )
        if self.KERNELS['model'] != self.model.id :
            ERROR( 'Response functions were not created with the same model' )
        if self.get_config('DTI_fit_method') not in ['OLS', 'LS', 'WLS', 'NLLS', 'RT', 'RESTORE', 'restore']:
            ERROR("DTI fit method must be one of the following:\n'OLS'(default) or 'LS': ordinary least squares\n'WLS': weighted least squares\n'NLLS': non-linear least squares\n'RT' or 'RESTORE' or 'restore': robust tensor\nNOTE: more info at https://dipy.org/documentation/1.6.0./reference/dipy.reconst/#dipy.reconst.dti.TensorModel")
        
        self.nthreads = self.get_config('nthreads') if self.get_config('nthreads') > 0 else cpu_count() if self.get_config('nthreads') == -1 else ERROR('Number of parallel threads must be positive or -1')
        self.BLAS_nthreads = self.get_config('BLAS_nthreads') if self.get_config('BLAS_nthreads') > 0 else cpu_count() if self.get_config('BLAS_nthreads') == -1 else ERROR('Number of BLAS threads must be positive or -1')

        self.set_config('fit_time', None)
        totVoxels = np.count_nonzero(self.niiMASK_img)
        # LOG( f'\n-> Fitting "{self.model.name}" model to {totVoxels} voxels (using {self.nthreads} thread{"s" if self.nthreads > 1 else ""}):' )

        # setup fitting directions
        peaks_filename = self.get_config('peaks_filename')
        if peaks_filename is None :
            if self.get_config('doMergeB0'):
                gtab = gradient_table( np.hstack((0,self.scheme.b[self.scheme.dwi_idx])), np.vstack((np.zeros((1,3)),self.scheme.raw[self.scheme.dwi_idx,:3])) )
            else:
                gtab = gradient_table( bvals=self.scheme.b, bvecs=self.scheme.raw[:,:3] )
            DTI = dti.TensorModel( gtab, fit_method=self.get_config('DTI_fit_method'))
        else :
            if not isfile( pjoin(self.get_config('DATA_path'), peaks_filename) ):
                ERROR( 'PEAKS file not found' )
            niiPEAKS = nibabel.load( pjoin( self.get_config('DATA_path'), peaks_filename) )
            self.DIRs = niiPEAKS.get_fdata().astype(np.float32)
            PRINT('\t* peaks dim = %d x %d x %d x %d' % self.DIRs.shape[:4])
            if self.DIRs.shape[:3] != self.niiMASK_img.shape[:3]:
                ERROR( 'PEAKS geometry does not match with DWI data' )
            DTI = None

        # fit the model to the data
        # =========================
        t = time.time()
        # NOTE binary mask indexing
        self.y = self.niiDWI_img[self.niiMASK_img==1, :].astype(np.double)
        self.y[self.y < 0] = 0

        # precompute directions
        LOG(f"\n-> Estimating principal directions ({self.get_config('DTI_fit_method')}):")
        if not self.get_config('doDirectionalAverage') and DTI is not None:
            with ProgressBar(disable=get_verbose()<3):
                self.DIRs = np.squeeze(DTI.fit(self.y).directions)
        self.set_config('dirs_precomputing_time', time.time()-t)
        LOG( '   [ %s ]' % ( time.strftime("%Hh %Mm %Ss", time.gmtime(self.get_config('dirs_precomputing_time')) ) ) )

        t = time.time()
        LOG(f"\n-> Fitting '{self.model.name}' model to {totVoxels} voxels (using {self.nthreads} thread{'s' if self.nthreads > 1 else ''}):")
        # call the fit() method of the actual model
        with self._controller.limit(limits=self.BLAS_nthreads, user_api='blas'):
            results = self.model.fit(self)
        self.set_config('fit_time', time.time()-t)
        LOG( '   [ %s ]' % ( time.strftime("%Hh %Mm %Ss", time.gmtime(self.get_config('fit_time')) ) ) )
        # =========================

        # store results
        self.RESULTS = {}
        # estimates (maps)
        self.RESULTS['MAPs'] = np.zeros([self.get_config('dim')[0], self.get_config('dim')[1], self.get_config('dim')[2], len(self.model.maps_name)], dtype=np.float32)
        self.RESULTS['MAPs'][self.niiMASK_img==1, :] = results['estimates']
        # directions
        self.RESULTS['DIRs'] = np.zeros([self.get_config('dim')[0], self.get_config('dim')[1], self.get_config('dim')[2], 3], dtype=np.float32)
        self.RESULTS['DIRs'][self.niiMASK_img==1, :] = self.DIRs
        # fitting error
        if self.get_config('doComputeRMSE') :
            self.RESULTS['RMSE'] = np.zeros([self.get_config('dim')[0], self.get_config('dim')[1], self.get_config('dim')[2]], dtype=np.float32)
            self.RESULTS['RMSE'][self.niiMASK_img==1] = results['rmse']
        if self.get_config('doComputeNRMSE') :
            self.RESULTS['NRMSE'] = np.zeros([self.get_config('dim')[0], self.get_config('dim')[1], self.get_config('dim')[2]], dtype=np.float32)
            self.RESULTS['NRMSE'][self.niiMASK_img==1] = results['nrmse']
        # Modulated NDI and ODI maps (NODDI)
        if self.model.name == 'NODDI' and self.get_config('doSaveModulatedMaps'):
            self.RESULTS['MAPs_mod'] = np.zeros([self.get_config('dim')[0], self.get_config('dim')[1], self.get_config('dim')[2], 2], dtype=np.float32)
            self.RESULTS['MAPs_mod'][self.niiMASK_img==1, :] = results['estimates_mod']
        # corrected DWI (Free-Water)
        if self.model.name == 'Free-Water' and self.get_config('doSaveCorrectedDWI') :
            y_corrected = results['y_corrected']
            if self.get_config('doNormalizeSignal') and self.scheme.b0_count > 0:
                y_corrected = y_corrected * np.reshape(self.mean_b0s[self.niiMASK_img==1], (-1, 1))
            if self.get_config('doKeepb0Intact') and self.scheme.b0_count > 0:
                y_corrected[:, self.scheme.b0_idx] = self.y[:, self.scheme.b0_idx] * np.reshape(self.mean_b0s[self.niiMASK_img==1], (-1, 1))
            self.RESULTS['DWI_corrected'] = np.zeros(self.niiDWI.shape, dtype=np.float32)
            self.RESULTS['DWI_corrected'][self.niiMASK_img==1, :] = y_corrected


    def save_results( self, path_suffix = None, save_dir_avg = False ) :
        """Save the output (directions, maps etc).

        Parameters
        ----------
        path_suffix : string
            Text to be appended to the output path (default : None)
        save_dir_avg : boolean
            If true and the option doDirectionalAverage is true
            the directional average signal and the scheme
            will be saved in files (default : False)
        """
        if self.RESULTS is None :
            ERROR( 'Model not fitted to the data; call "fit()" first' )
        if self.get_config('OUTPUT_path') is None:
            RESULTS_path = pjoin( 'AMICO', self.model.id )
            if path_suffix :
                RESULTS_path = RESULTS_path +'_'+ path_suffix
            self.RESULTS['RESULTS_path'] = RESULTS_path
            LOG( f'\n-> Saving output to "{pjoin(RESULTS_path, "*")}":' )
            RESULTS_path = pjoin( self.get_config('DATA_path'), RESULTS_path )
        else:
            RESULTS_path = self.get_config('OUTPUT_path')
            if path_suffix :
                RESULTS_path = RESULTS_path +'_'+ path_suffix
            self.RESULTS['RESULTS_path'] = RESULTS_path
            LOG( f'\n-> Saving output to "{pjoin(RESULTS_path, "*")}":' )

        # delete previous output
        if not exists( RESULTS_path ) :
            makedirs( RESULTS_path )
        else :
            for f in glob.glob( pjoin(RESULTS_path,'*') ) :
                remove( f )

        # configuration file
        PRINT('\t- configuration', end=' ')
        with open( pjoin(RESULTS_path,'config.pickle'), 'wb+' ) as fid :
            pickle.dump( self.CONFIG, fid, protocol=2 )
        PRINT(' [OK]')

        affine  = self.niiDWI.affine if nibabel.__version__ >= '2.0.0' else self.niiDWI.get_affine()
        hdr     = self.niiDWI.header if nibabel.__version__ >= '2.0.0' else self.niiDWI.get_header()
        hdr['datatype'] = 16
        hdr['bitpix'] = 32

        # estimated orientations
        if not self.get_config('doDirectionalAverage'):
            PRINT('\t- fit_dir.nii.gz', end=' ')
            niiMAP_img = self.RESULTS['DIRs']
            niiMAP     = nibabel.Nifti1Image( niiMAP_img, affine, hdr )
            niiMAP_hdr = niiMAP.header if nibabel.__version__ >= '2.0.0' else niiMAP.get_header()
            niiMAP_hdr['cal_min'] = -1
            niiMAP_hdr['cal_max'] = 1
            niiMAP_hdr['scl_slope'] = 1
            niiMAP_hdr['scl_inter'] = 0
            nibabel.save( niiMAP, pjoin(RESULTS_path, 'fit_dir.nii.gz') )
            PRINT(' [OK]')

        # fitting error
        if self.get_config('doComputeRMSE') :
            PRINT('\t- fit_RMSE.nii.gz', end=' ')
            niiMAP_img = self.RESULTS['RMSE']
            niiMAP     = nibabel.Nifti1Image( niiMAP_img, affine, hdr )
            niiMAP_hdr = niiMAP.header if nibabel.__version__ >= '2.0.0' else niiMAP.get_header()
            niiMAP_hdr['cal_min'] = 0
            niiMAP_hdr['cal_max'] = 1
            niiMAP_hdr['scl_slope'] = 1
            niiMAP_hdr['scl_inter'] = 0
            nibabel.save( niiMAP, pjoin(RESULTS_path, 'fit_RMSE.nii.gz') )
            PRINT(' [OK]')
        if self.get_config('doComputeNRMSE') :
            PRINT('\t- fit_NRMSE.nii.gz', end=' ')
            niiMAP_img = self.RESULTS['NRMSE']
            niiMAP     = nibabel.Nifti1Image( niiMAP_img, affine, hdr )
            niiMAP_hdr = niiMAP.header if nibabel.__version__ >= '2.0.0' else niiMAP.get_header()
            niiMAP_hdr['cal_min'] = 0
            niiMAP_hdr['cal_max'] = 1
            niiMAP_hdr['scl_slope'] = 1
            niiMAP_hdr['scl_inter'] = 0
            nibabel.save( niiMAP, pjoin(RESULTS_path, 'fit_NRMSE.nii.gz') )
            PRINT(' [OK]')

        # corrected DWI (Free-Water)
        if self.get_config('doSaveCorrectedDWI') :
            if self.model.name == 'Free-Water' :
                PRINT('\t- DWI_corrected.nii.gz', end=' ')
                niiMAP_img = self.RESULTS['DWI_corrected']
                niiMAP     = nibabel.Nifti1Image( niiMAP_img, affine, hdr )
                niiMAP_hdr = niiMAP.header if nibabel.__version__ >= '2.0.0' else niiMAP.get_header()
                niiMAP_hdr['cal_min'] = 0
                niiMAP_hdr['cal_max'] = 1
                nibabel.save( niiMAP, pjoin(RESULTS_path, 'DWI_corrected.nii.gz') )
                PRINT(' [OK]')
            else :
                WARNING( f'"doSaveCorrectedDWI" option not supported for "{self.model.name}" model' )

        # voxelwise maps
        for i in range( len(self.model.maps_name) ) :
            PRINT(f'\t- fit_{self.model.maps_name[i]}.nii.gz', end=' ')
            niiMAP_img = self.RESULTS['MAPs'][:,:,:,i]
            niiMAP     = nibabel.Nifti1Image( niiMAP_img, affine, hdr )
            niiMAP_hdr = niiMAP.header if nibabel.__version__ >= '2.0.0' else niiMAP.get_header()
            niiMAP_hdr['descrip'] = self.model.maps_descr[i] + f' (AMICO v{self.get_config("version")})'
            niiMAP_hdr['cal_min'] = niiMAP_img.min()
            niiMAP_hdr['cal_max'] = niiMAP_img.max()
            niiMAP_hdr['scl_slope'] = 1
            niiMAP_hdr['scl_inter'] = 0
            nibabel.save( niiMAP, pjoin(RESULTS_path, f'fit_{self.model.maps_name[i]}.nii.gz' ) )
            PRINT(' [OK]')

        # modulated NDI and ODI maps (NODDI)
        if self.get_config('doSaveModulatedMaps'):
            if self.model.name == 'NODDI':
                mod_maps = [name + '_modulated' for name in self.model.maps_name[:2]]
                descr = [descr + ' modulated' for descr in self.model.maps_descr[:2]]
                for i in range(len(mod_maps)):
                    PRINT(f'\t- fit_{mod_maps[i]}.nii.gz', end=' ')
                    niiMAP_img = self.RESULTS['MAPs_mod'][:,:,:,i]
                    niiMAP     = nibabel.Nifti1Image( niiMAP_img, affine, hdr )
                    niiMAP_hdr = niiMAP.header if nibabel.__version__ >= '2.0.0' else niiMAP.get_header()
                    niiMAP_hdr['descrip'] = descr[i] + f' (AMICO v{self.get_config("version")})'
                    niiMAP_hdr['cal_min'] = niiMAP_img.min()
                    niiMAP_hdr['cal_max'] = niiMAP_img.max()
                    niiMAP_hdr['scl_slope'] = 1
                    niiMAP_hdr['scl_inter'] = 0
                    nibabel.save( niiMAP, pjoin(RESULTS_path, f'fit_{mod_maps[i]}.nii.gz' ) )
                    PRINT(' [OK]')
            else:
                WARNING(f'"doSaveModulatedMaps" option not supported for "{self.model.name}" model')

        # Directional average signal
        if save_dir_avg:
            if self.get_config('doDirectionalAverage'):
                PRINT('\t- dir_avg_signal.nii.gz', end=' ')
                niiMAP     = nibabel.Nifti1Image( self.niiDWI_img, affine, hdr )
                niiMAP_hdr = niiMAP.header if nibabel.__version__ >= '2.0.0' else niiMAP.get_header()
                niiMAP_hdr['descrip'] = 'Directional average signal of each shell' + f' (AMICO v{self.get_config("version")})'
                nibabel.save( niiMAP , pjoin(RESULTS_path, 'dir_avg_signal.nii.gz' ) )
                PRINT(' [OK]')

                PRINT('\t- dir_avg.scheme', end=' ')
                np.savetxt( pjoin(RESULTS_path, 'dir_avg.scheme' ), self.scheme.get_table(), fmt="%.06f", delimiter="\t", header=f'VERSION: {self.scheme.version}', comments='' )
                PRINT(' [OK]')
            else:
                WARNING('The directional average signal was not created (The option doDirectionalAverage is False).')

        LOG( '   [ DONE ]' )
