from __future__ import absolute_import, division, print_function

import numpy as np
import time
import glob
import sys
from os import makedirs, remove
from os.path import exists, join as pjoin, isfile, isdir

import nibabel
import pickle
import amico.scheme
from amico.preproc import debiasRician
import amico.lut
import amico.models
from amico.lut import is_valid, valid_dirs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from amico.util import PRINT, LOG, NOTE, WARNING, ERROR, get_verbose
from pkg_resources import get_distribution
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm


def setup( lmax=12, ndirs=None ) :
    """General setup/initialization of the AMICO framework.

    Parameters
    ----------
    lmax : int
        Maximum SH order to use for the rotation phase (default : 12).
        NB: change only if you know what you are doing.
     ndirs : int
        DEPRECATED. Now, all directions are precomputed.
    """
    if ndirs is not None:
        WARNING('"ndirs" parameter is deprecated')
    LOG( f'\n-> Precomputing rotation matrices:' )
    for n in tqdm(valid_dirs(), ncols=70, bar_format='   |{bar}| {percentage:4.1f}%', disable=(get_verbose()<3)):
        amico.lut.precompute_rotation_matrices( lmax, n )
    LOG('   [ DONE ]')



class Evaluation :
    """Class to hold all the information (data and parameters) when performing an
    evaluation with the AMICO framework.
    """

    def __init__( self, study_path, subject, output_path=None, verbose=2 ) :
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
        verbose : int
            Possible values: 2->show all, 1->hide progress bars, 0->hide all
        """
        self.niiDWI      = None # set by "load_data" method
        self.niiDWI_img  = None
        self.scheme      = None
        self.niiMASK     = None
        self.niiMASK_img = None
        self.model       = None # set by "set_model" method
        self.KERNELS     = None # set by "load_kernels" method
        self.RESULTS     = None # set by "fit" method
        self.mean_b0s    = None # set by "load_data" method
        self.htable      = None

        # store all the parameters of an evaluation with AMICO
        self.CONFIG = {}
        self.set_config('version', get_distribution('dmri-amico').version)
        self.set_config('study_path', study_path)
        self.set_config('subject', subject)
        self.set_config('DATA_path', pjoin( study_path, subject ))
        self.set_config('OUTPUT_path', output_path)

        self.set_config('peaks_filename', None)
        self.set_config('doNormalizeSignal', True)
        self.set_config('doKeepb0Intact', False)        # does change b0 images in the predicted signal
        self.set_config('doComputeNRMSE', False)
        self.set_config('doSaveCorrectedDWI', False)
        self.set_config('doMergeB0', False)             # Merge b0 volumes
        self.set_config('doDebiasSignal', False)        # Flag to remove Rician bias
        self.set_config('DWI-SNR', None)                # SNR of DWI image: SNR = b0/sigma
        self.set_config('doDirectionalAverage', False)  # To perform the directional average on the signal of each shell
        self.set_config('parallel_jobs', -1)            # Number of jobs to be used in multithread-enabled parts of code
        self.set_config('parallel_backend', 'loky')     # Backend to use for the joblib library
        self.set_config('verbose', verbose)

    def set_config( self, key, value ) :
        self.CONFIG[ key ] = value

    def get_config( self, key ) :
        return self.CONFIG.get( key )


    def load_data( self, dwi_filename='DWI.nii', scheme_filename='DWI.scheme', mask_filename=None, b0_thr=0 ) :
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
        """

        # Loading data, acquisition scheme and mask (optional)
        LOG( '\n-> Loading data:' )
        tic = time.time()

        PRINT('\t* DWI signal')
        if not isfile( pjoin(self.get_config('DATA_path'), dwi_filename) ):
            ERROR( 'DWI file not found' )
        self.set_config('dwi_filename', dwi_filename)
        self.niiDWI  = nibabel.load( pjoin(self.get_config('DATA_path'), dwi_filename) )
        self.niiDWI_img = self.niiDWI.get_data().astype(np.float32)
        hdr = self.niiDWI.header if nibabel.__version__ >= '2.0.0' else self.niiDWI.get_header()
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
            self.niiMASK  = nibabel.load( pjoin( self.get_config('DATA_path'), mask_filename) )
            self.niiMASK_img = self.niiMASK.get_data().astype(np.uint8)
            niiMASK_hdr = self.niiMASK.header if nibabel.__version__ >= '2.0.0' else self.niiMASK.get_header()
            PRINT('\t\t- dim    = %d x %d x %d' % self.niiMASK_img.shape[:3])
            PRINT('\t\t- pixdim = %.3f x %.3f x %.3f' % niiMASK_hdr.get_zooms()[:3])
            if self.niiMASK.ndim != 3 :
                ERROR( 'The provided MASK if 4D, but a 3D dataset is expected' )
            if self.get_config('dim') != self.niiMASK_img.shape[:3] :
                ERROR( 'MASK geometry does not match with DWI data' )
        else :
            self.niiMASK = None
            self.niiMASK_img = np.ones( self.get_config('dim') )
            PRINT('\t\t- not specified')
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
            idx = self.mean_b0s <= 0
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
        """Setup the specific parameters of the solver to fit the model.
        Dispatch to the proper function, depending on the model; a model shoudl provide a "set_solver" function to set these parameters.
        """
        if self.model is None :
            ERROR( 'Model not set; call "set_model()" method first' )
        self.set_config('solver_params', self.model.set_solver( **params ))


    def generate_kernels( self, regenerate = False, lmax = 12, ndirs = 32761 ) :
        """Generate the high-resolution response functions for each compartment.
        Dispatch to the proper function, depending on the model.

        Parameters
        ----------
        regenerate : boolean
            Regenerate kernels if they already exist (default : False)
        lmax : int
            Maximum SH order to use for the rotation procedure (default : 12)
        ndirs : int
            Number of directions on the half of the sphere representing the possible orientations of the response functions (default : 32761)
         """
        if self.scheme is None :
            ERROR( 'Scheme not loaded; call "load_data()" first' )
        if self.model is None :
            ERROR( 'Model not set; call "set_model()" method first' )
        if not is_valid(ndirs):
            ERROR( 'Unsupported value for ndirs.\nNote: Supported values for ndirs are [1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 32761 (default)]' )

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

        tic = time.time()
        LOG( f'\n-> Resampling LUT for subject "{self.get_config("subject")}":' )

        # auxiliary data structures
        idx_OUT, Ylm_OUT = amico.lut.aux_structures_resample( self.scheme, self.get_config('lmax') )

        # hash table
        self.htable = amico.lut.load_precomputed_hash_table( self.get_config('ndirs') )

        # Dispatch to the right handler for each model
        self.KERNELS = self.model.resample( self.get_config('ATOMS_path'), idx_OUT, Ylm_OUT, self.get_config('doMergeB0'), self.get_config('ndirs') )

        LOG( f'   [ {time.time() - tic:.1f} seconds ]')


    def fit( self ) :
        """Fit the model to the data iterating over all voxels (in the mask) one after the other.
        Call the appropriate fit() method of the actual model used.
        """
        def fit_voxel(self, ix, iy, iz, dirs, DTI) :
            """Perform the fit in a single voxel.
            """
            # prepare the signal
            y = self.niiDWI_img[ix, iy, iz, :].astype(np.float64)
            y[y < 0] = 0  # [NOTE] this should not happen!

            # fitting directions if not
            if not self.get_config('doDirectionalAverage') and DTI is not None :
                dirs = DTI.fit( y ).directions[0].reshape(-1, 3)

            # dispatch to the right handler for each model
            results, dirs, x, A = self.model.fit( y, dirs, self.KERNELS, self.get_config('solver_params'), self.htable)

            # compute fitting error
            if self.get_config('doComputeNRMSE') :
                y_est = np.dot( A, x )
                den = np.sum(y**2)
                NRMSE = np.sqrt( np.sum((y-y_est)**2) / den ) if den > 1e-16 else 0
            else :
                NRMSE = 0.0

            y_fw_corrected = None
            if self.get_config('doSaveCorrectedDWI') :

                if self.model.name == 'Free-Water' :
                    n_iso = len(self.model.d_isos)

                    # keep only FW components of the estimate
                    x[0:x.shape[0]-n_iso] = 0

                    # y_fw_corrected below is the predicted signal by the anisotropic part (no iso part)
                    y_fw_part = np.dot( A, x )

                    # y is the original signal
                    y_fw_corrected = y - y_fw_part
                    y_fw_corrected[ y_fw_corrected < 0 ] = 0 # [NOTE] this should not happen!

                    if self.get_config('doNormalizeSignal') and self.scheme.b0_count > 0 :
                        y_fw_corrected = y_fw_corrected * self.mean_b0s[ix,iy,iz]

                    if self.get_config('doKeepb0Intact') and self.scheme.b0_count > 0 :
                        # put original b0 data back in.
                        y_fw_corrected[self.scheme.b0_idx] = y[self.scheme.b0_idx]*self.mean_b0s[ix,iy,iz]

            return results, dirs, NRMSE, y_fw_corrected


        if self.niiDWI is None :
            ERROR( 'Data not loaded; call "load_data()" first' )
        if self.model is None :
            ERROR( 'Model not set; call "set_model()" first' )
        if self.KERNELS is None :
            ERROR( 'Response functions not generated; call "generate_kernels()" and "load_kernels()" first' )
        if self.KERNELS['model'] != self.model.id :
            ERROR( 'Response functions were not created with the same model' )
        n_jobs = self.get_config( 'parallel_jobs' )
        if n_jobs == -1 :
            n_jobs = cpu_count()
        elif n_jobs == 0 or n_jobs < -1:
            ERROR( 'Number of parallel jobs must be positive or -1' )
        parallel_backend = self.get_config( 'parallel_backend' )
        if parallel_backend not in ['loky','multiprocessing','threading']:
            ERROR( f'Backend "{parallel_backend}" is not recognized by joblib' )

        self.set_config('fit_time', None)
        totVoxels = np.count_nonzero(self.niiMASK_img)
        LOG( f'\n-> Fitting "{self.model.name}" model to {totVoxels} voxels:' )

        # setup fitting directions
        peaks_filename = self.get_config('peaks_filename')
        if peaks_filename is None :
            DIRs = np.zeros( [self.get_config('dim')[0], self.get_config('dim')[1], self.get_config('dim')[2], 3], dtype=np.float32 )
            nDIR = 1
            if self.get_config('doMergeB0'):
                gtab = gradient_table( np.hstack((0,self.scheme.b[self.scheme.dwi_idx])), np.vstack((np.zeros((1,3)),self.scheme.raw[self.scheme.dwi_idx,:3])) )
            else:
                gtab = gradient_table( self.scheme.b, self.scheme.raw[:,:3] )
            DTI = dti.TensorModel( gtab )
        else :
            if not isfile( pjoin(self.get_config('DATA_path'), peaks_filename) ):
                ERROR( 'PEAKS file not found' )
            niiPEAKS = nibabel.load( pjoin( self.get_config('DATA_path'), peaks_filename) )
            DIRs = niiPEAKS.get_data().astype(np.float32)
            nDIR = np.floor( DIRs.shape[3]/3 )
            PRINT('\t* peaks dim = %d x %d x %d x %d' % DIRs.shape[:4])
            if DIRs.shape[:3] != self.niiMASK_img.shape[:3] :
                ERROR( 'PEAKS geometry does not match with DWI data' )
            DTI = None

        # setup other output files
        MAPs = np.zeros( [self.get_config('dim')[0], self.get_config('dim')[1], self.get_config('dim')[2], len(self.model.maps_name)], dtype=np.float32 )

        # fit the model to the data
        # =========================
        t = time.time()

        ix, iy, iz = np.nonzero(self.niiMASK_img)
        n_per_thread = np.floor(totVoxels / n_jobs)
        idx = np.arange(0, totVoxels+1, n_per_thread, dtype=np.int32)
        idx[-1] = totVoxels

        estimates = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
            delayed(fit_voxel)(self, ix[i], iy[i], iz[i], DIRs[ix[i],iy[i],iz[i],:], DTI)
            for i in tqdm(range(totVoxels), ncols=70, bar_format='   |{bar}| {percentage:4.1f}%', disable=(get_verbose()<3))
        )

        self.set_config('fit_time', time.time()-t)
        LOG( '   [ %s ]' % ( time.strftime("%Hh %Mm %Ss", time.gmtime(self.get_config('fit_time')) ) ) )

        # store results
        self.RESULTS = {}

        for i in range(totVoxels) :
            MAPs[ix[i],iy[i],iz[i],:] = estimates[i][0]
            DIRs[ix[i],iy[i],iz[i],:] = estimates[i][1]
        self.RESULTS['DIRs']  = DIRs
        self.RESULTS['MAPs']  = MAPs

        if self.get_config('doComputeNRMSE') :
            NRMSE = np.zeros( [self.get_config('dim')[0], self.get_config('dim')[1], self.get_config('dim')[2]], dtype=np.float32 )
            for i in range(totVoxels) :
                NRMSE[ix[i],iy[i],iz[i]] = estimates[i][2]
            self.RESULTS['NRMSE'] = NRMSE

        if self.get_config('doSaveCorrectedDWI') :
            DWI_corrected = np.zeros(self.niiDWI.shape, dtype=np.float32)
            for i in range(totVoxels):
                DWI_corrected[ix[i],iy[i],iz[i],:] = estimates[i][3]
            self.RESULTS['DWI_corrected'] = DWI_corrected


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
            LOG( f'\n-> Saving output to "{RESULTS_path}/*":' )

            # delete previous output
            RESULTS_path = pjoin( self.get_config('DATA_path'), RESULTS_path )
        else:
            RESULTS_path = self.get_config('OUTPUT_path')
            if path_suffix :
                RESULTS_path = RESULTS_path +'_'+ path_suffix
            self.RESULTS['RESULTS_path'] = RESULTS_path
            LOG( f'\n-> Saving output to "{RESULTS_path}/*":' )

        if not exists( RESULTS_path ) :
            makedirs( RESULTS_path )
        else :
            for f in glob.glob( pjoin(RESULTS_path,'*') ) :
                remove( f )

        # configuration
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
            PRINT('\t- FIT_dir.nii.gz', end=' ')
            niiMAP_img = self.RESULTS['DIRs']
            niiMAP     = nibabel.Nifti1Image( niiMAP_img, affine, hdr )
            niiMAP_hdr = niiMAP.header if nibabel.__version__ >= '2.0.0' else niiMAP.get_header()
            niiMAP_hdr['cal_min'] = -1
            niiMAP_hdr['cal_max'] = 1
            niiMAP_hdr['scl_slope'] = 1
            niiMAP_hdr['scl_inter'] = 0
            nibabel.save( niiMAP, pjoin(RESULTS_path, 'FIT_dir.nii.gz') )
            PRINT(' [OK]')

        # fitting error
        if self.get_config('doComputeNRMSE') :
            PRINT('\t- FIT_nrmse.nii.gz', end=' ')
            niiMAP_img = self.RESULTS['NRMSE']
            niiMAP     = nibabel.Nifti1Image( niiMAP_img, affine, hdr )
            niiMAP_hdr = niiMAP.header if nibabel.__version__ >= '2.0.0' else niiMAP.get_header()
            niiMAP_hdr['cal_min'] = 0
            niiMAP_hdr['cal_max'] = 1
            niiMAP_hdr['scl_slope'] = 1
            niiMAP_hdr['scl_inter'] = 0
            nibabel.save( niiMAP, pjoin(RESULTS_path, 'FIT_nrmse.nii.gz') )
            PRINT(' [OK]')

        if self.get_config('doSaveCorrectedDWI') :
            if self.model.name == 'Free-Water' :
                PRINT('\t- dwi_fw_corrected.nii.gz', end=' ')
                niiMAP_img = self.RESULTS['DWI_corrected']
                niiMAP     = nibabel.Nifti1Image( niiMAP_img, affine, hdr )
                niiMAP_hdr = niiMAP.header if nibabel.__version__ >= '2.0.0' else niiMAP.get_header()
                niiMAP_hdr['cal_min'] = 0
                niiMAP_hdr['cal_max'] = 1
                nibabel.save( niiMAP, pjoin(RESULTS_path, 'dwi_fw_corrected.nii.gz') )
                PRINT(' [OK]')
            else :
                WARNING( f'"doSaveCorrectedDWI" option not supported for "{self.model.name}" model' )

        # voxelwise maps
        for i in range( len(self.model.maps_name) ) :
            PRINT(f'\t- FIT_{self.model.maps_name[i]}.nii.gz', end=' ')
            niiMAP_img = self.RESULTS['MAPs'][:,:,:,i]
            niiMAP     = nibabel.Nifti1Image( niiMAP_img, affine, hdr )
            niiMAP_hdr = niiMAP.header if nibabel.__version__ >= '2.0.0' else niiMAP.get_header()
            niiMAP_hdr['descrip'] = self.model.maps_descr[i] + f' (AMICO v{self.get_config("version")})'
            niiMAP_hdr['cal_min'] = niiMAP_img.min()
            niiMAP_hdr['cal_max'] = niiMAP_img.max()
            niiMAP_hdr['scl_slope'] = 1
            niiMAP_hdr['scl_inter'] = 0
            nibabel.save( niiMAP, pjoin(RESULTS_path, f'FIT_{self.model.maps_name[i]}.nii.gz' ) )
            PRINT(' [OK]')

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
                np.savetxt( pjoin(RESULTS_path, 'dir_avg.scheme' ), self.scheme.get_table(), fmt="%.06f", delimiter="\t", header=f"VERSION: {self.scheme.version}", comments='' )
                PRINT(' [OK]')
            else:
                WARNING('The directional average signal was not created (The option doDirectionalAverage is False).')

        LOG( '   [ DONE ]' )
