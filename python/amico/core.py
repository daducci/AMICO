import numpy as np
import time
import glob
import sys
from os import makedirs, remove
from os.path import exists, join as pjoin
import nibabel
import cPickle
import amico.scheme
import amico.lut
import amico.models
from amico.progressbar import ProgressBar
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti


def setup( lmax = 12 ) :
    """
    General setup/initialization of the AMICO framework.
    """
    amico.lut.precompute_rotation_matrices( lmax )



"""
Class to hold all the information (data and parameters) when performing an
evaluation with the AMICO framework.
"""
class Evaluation :

    def __init__( self, study_path, subject ) :
        """
        Setup the data structure with default values.

        Parameters
        ----------
        study_path : string
            The path to the folder containing all the subjects from one study
        subject : string
            The path (relative to previous folder) to the subject folder
        """
        self.niiDWI      = None # set by "load_data" method
        self.niiDWI_img  = None
        self.scheme      = None
        self.niiMASK     = None
        self.niiMASK_img = None
        self.model       = None # set by "set_model" method
        self.KERNELS     = None # set by "load_kernels" method
        self.RESULTS     = None # set by "fit" method

        # store all the parameters of an evaluation with AMICO
        self.CONFIG = {}
        self.CONFIG['study_path']  = study_path
        self.CONFIG['subject']     = subject
        self.CONFIG['DATA_path']   = pjoin( study_path, subject )

        self.CONFIG['doNormalizeSignal'] = False
        self.CONFIG['doMergeB0']	     = True
        self.CONFIG['doComputeNRMSE']	 = False

        self.CONFIG['optimization']      = {} # set by "set_model"


    def load_data( self, dwi_filename = 'DWI.nii', scheme_filename = 'DWI.scheme', mask_filename = None, b0_thr = 0 ) :
        """
        Load the diffusion signal and its corresponding acquisition scheme.

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
        tic = time.time()
        print '\n-> Loading data:'

        print '\t* DWI signal...'
        self.CONFIG['dwi_filename']    = dwi_filename
        self.niiDWI  = nibabel.load( pjoin( self.CONFIG['DATA_path'], dwi_filename) )
        self.niiDWI_img = self.niiDWI.get_data().astype(np.float32)
        hdr = self.niiDWI.header if nibabel.__version__ >= '2.0.0' else self.niiDWI.get_header()
        self.CONFIG['dim']    = self.niiDWI_img.shape[:3]
        self.CONFIG['pixdim'] = tuple( hdr.get_zooms()[:3] )
        print '\t\t- dim    = %d x %d x %d x %d' % self.niiDWI_img.shape
        print '\t\t- pixdim = %.3f x %.3f x %.3f' % self.CONFIG['pixdim']
        # % Scale signal intensities (if necessary)
        # if ( niiDWI.hdr.dime.scl_slope ~= 0 && ( niiDWI.hdr.dime.scl_slope ~= 1 || niiDWI.hdr.dime.scl_inter ~= 0 ) )
        #     fprintf( '\t\t- rescaling data\n' );
        # 	niiDWI.img = niiDWI.img * niiDWI.hdr.dime.scl_slope + niiDWI.hdr.dime.scl_inter;
        # end

        print '\t* Acquisition scheme...'
        self.CONFIG['scheme_filename'] = scheme_filename
        self.CONFIG['b0_thr'] = b0_thr
        self.scheme = amico.scheme.Scheme( pjoin( self.CONFIG['DATA_path'], scheme_filename), b0_thr )
        print '\t\t- %d samples, %d shells' % ( self.scheme.nS, len(self.scheme.shells) )
        print '\t\t- %d @ b=0' % ( self.scheme.b0_count ),
        for i in xrange(len(self.scheme.shells)) :
            print ', %d @ b=%.1f' % ( len(self.scheme.shells[i]['idx']), self.scheme.shells[i]['b'] ),
        print

        if self.scheme.nS != self.niiDWI_img.shape[3] :
            raise ValueError( 'Scheme does not match with DWI data' )

        print '\t* Binary mask...'
        if mask_filename is not None :
            self.niiMASK  = nibabel.load( pjoin( self.CONFIG['DATA_path'], mask_filename) )
            self.niiMASK_img = self.niiMASK.get_data().astype(np.uint8)
            niiMASK_hdr = self.niiMASK.header if nibabel.__version__ >= '2.0.0' else self.niiMASK.get_header()
            print '\t\t- dim    = %d x %d x %d' % self.niiMASK_img.shape[:3]
            print '\t\t- pixdim = %.3f x %.3f x %.3f' % niiMASK_hdr.get_zooms()[:3]
            if self.CONFIG['dim'] != self.niiMASK_img.shape[:3] :
                raise ValueError( 'MASK geometry does not match with DWI data' )
        else :
            self.niiMASK = None
            self.niiMASK_img = np.ones( self.CONFIG['dim'] )
            print '\t\t- not specified'
        print '\t\t- voxels = %d' % np.count_nonzero(self.niiMASK_img)

        print '   [ %.1f seconds ]' % ( time.time() - tic )


    def set_model( self, model_name ) :
        """
        Set the model to use to describe the signal contributions in each voxel.

        Parameters
        ----------
        model_name : string
            The name of the model (must match a class name in "amico.models" module)
        """
        # Call the specific model constructor
        if hasattr(amico.models, model_name ) :
            self.model = getattr(amico.models,model_name)()
        else :
            raise ValueError( 'Model "%s" not recognized' % model_name )

        self.CONFIG['model_name'] = self.model.name
        self.CONFIG['ATOMS_path'] = pjoin( self.CONFIG['study_path'], 'kernels', self.model.id )

        # setup default parameters for fitting the model (can be changed later on)
        self.set_solver( )


    def set_solver( self, **params ) :
        """
        Setup the specific parameters of the solver to fit the model.
        Dispatch to the proper function, depending on the model; a model shoudl provide a "set_solver" function to set these parameters.
        """
        if self.model is None :
            raise RuntimeError( 'Model not set; call "set_model()" method first.' )
        self.CONFIG['optimization']['params'] = self.model.set_solver( **params )


    def generate_kernels( self, regenerate = False, lmax = 12 ) :
        """
        Generate the high-resolution response functions for each compartment.
        Dispatch to the proper function, depending on the model.

        Parameters
        ----------
        regenerate : boolean
            Regenerate kernels if they already exist (default : False)
        lmax : int
            Maximum SH order to use for the rotation procedure (default : 12)
        """
        if self.scheme is None :
            raise RuntimeError( 'Scheme not loaded; call "load_data()" first.' )
        if self.model is None :
            raise RuntimeError( 'Model not set; call "set_model()" method first.' )

        # store some values for later use
        self.CONFIG['lmax'] = lmax
        self.model.nS = self.scheme.nS

        print '\n-> Simulating with "%s" model:' % self.model.name

        # check if kernels were already generated
        tmp = glob.glob( pjoin(self.CONFIG['ATOMS_path'],'A_*.npy') )
        if len(tmp)>0 and not regenerate :
            print '   [ Kernels already computed. Call "generate_kernels( regenerate=True )" to force regeneration. ]'
            return

        # create folder or delete existing files (if any)
        if not exists( self.CONFIG['ATOMS_path'] ) :
            makedirs( self.CONFIG['ATOMS_path'] )
        else :
            for f in glob.glob( pjoin(self.CONFIG['ATOMS_path'],'*') ) :
                remove( f )

        # auxiliary data structures
        aux = amico.lut.load_precomputed_rotation_matrices( lmax )
        idx_IN, idx_OUT = amico.lut.aux_structures_generate( self.scheme, lmax )

        # Dispatch to the right handler for each model
        tic = time.time()
        self.model.generate( self.CONFIG['ATOMS_path'], self.scheme, aux, idx_IN, idx_OUT )
        print '   [ %.1f seconds ]' % ( time.time() - tic )


    def load_kernels( self ) :
        """
        Load rotated kernels and project to the specific gradient scheme of this subject.
        Dispatch to the proper function, depending on the model.
        """
        if self.model is None :
            raise RuntimeError( 'Model not set; call "set_model()" method first.' )
        if self.scheme is None :
            raise RuntimeError( 'Scheme not loaded; call "load_data()" first.' )

        tic = time.time()
        print '\n-> Resampling kernels for subject "%s":' % self.CONFIG['subject']

        # auxiliary data structures
        idx_OUT, Ylm_OUT = amico.lut.aux_structures_resample( self.scheme, self.CONFIG['lmax'] )

        # Dispatch to the right handler for each model
        self.KERNELS = self.model.resample( self.CONFIG['ATOMS_path'], idx_OUT, Ylm_OUT )

        print '   [ %.1f seconds ]' % ( time.time() - tic )


    def fit( self ) :
        """
        Fit the model to the data iterating over all voxels (in the mask) one after the other.
        Call the appropriate fit() method of the actual model used.

        """
        if self.niiDWI is None :
            raise RuntimeError( 'Data not loaded; call "load_data()" first.' )
        if self.model is None :
            raise RuntimeError( 'Model not set; call "set_model()" first.' )
        if self.KERNELS is None :
            raise RuntimeError( 'Response functions not generated; call "generate_kernels()" and "load_kernels()" first.' )
        if self.KERNELS['model'] != self.model.id :
            raise RuntimeError( 'Response functions were not created with the same model.' )

        # setup output files
        MAPs  = np.zeros( [self.CONFIG['dim'][0], self.CONFIG['dim'][1], self.CONFIG['dim'][2], len(self.model.OUTPUT_names)], dtype=np.float32 )
        DIRs  = np.zeros( [self.CONFIG['dim'][0], self.CONFIG['dim'][1], self.CONFIG['dim'][2], 3], dtype=np.float32 )
        if self.CONFIG['doComputeNRMSE'] :
            NRMSE = np.zeros( [self.CONFIG['dim'][0], self.CONFIG['dim'][1], self.CONFIG['dim'][2]], dtype=np.float32 )

        # prepare DTI fitting
        gtab = gradient_table( self.scheme.b, self.scheme.raw[:,:3] )
        DTI = dti.TensorModel(gtab)

        # compute indices of samples to use
        idx = None
        if self.CONFIG['doMergeB0'] and self.scheme.b0_count > 0 :
            idx = np.append( self.scheme.dwi_idx, self.scheme.b0_idx[0] )

        # fit the model to the data
        # =========================
        t = time.time()
        totVoxels = np.count_nonzero(self.niiMASK_img)
        print '\n-> Fitting "%s" model separately to all %d voxels:' % ( self.model.name, totVoxels )

        progress = ProgressBar( n=totVoxels, prefix="   ", erase=True )
        for iz in xrange(self.CONFIG['dim'][2]) :
            for iy in xrange(self.CONFIG['dim'][1]) :
                for ix in xrange(self.CONFIG['dim'][0]) :
                    if self.niiMASK_img[ix,iy,iz]==0 :
                        continue

                    # prepare the signal
                    y = self.niiDWI_img[ix,iy,iz,:].astype(np.float64)
                    y[ y < 0 ] = 0 # [NOTE] this should not happen!

                    b0 = np.mean( y[self.scheme.b0_idx] )
                    if self.CONFIG['doMergeB0'] and self.scheme.b0_count > 0 :
                        y[self.scheme.b0_idx] = b0

                    if self.CONFIG['doNormalizeSignal'] and b0 > 1e-3:
                        y = y / b0

                    # find the MAIN DIFFUSION DIRECTION using DTI
                    dir = DTI.fit( y ).directions[0]
                    DIRs[ix,iy,iz,:] = dir

                    # dispatch to the right handler for each model
                    i1, i2 = amico.lut.dir_TO_lut_idx( dir )
                    y_est, MAPs[ix,iy,iz,:] = self.model.fit( y, i1, i2, self.KERNELS, idx, self.CONFIG['optimization']['params'] )

                    # compute fitting error
                    if self.CONFIG['doComputeNRMSE'] :
                        den = np.sum(y**2)
                        NRMSE[ix,iy,iz] = np.sqrt( np.sum((y-y_est)**2) / den ) if den > 1e-16 else 0

                    progress.update()

        self.CONFIG['optimization']['fit_time'] = time.time()-t
        print '   [ %s ]' % ( time.strftime("%Hh %Mm %Ss", time.gmtime(self.CONFIG['optimization']['fit_time']) ) )

        # store results
        self.RESULTS = {}
        self.RESULTS['DIRs']  = DIRs
        self.RESULTS['MAPs']  = MAPs
        if self.CONFIG['doComputeNRMSE'] :
            self.RESULTS['NRMSE'] = NRMSE


    def save_results( self, path_suffix = None ) :
        """
        Save the output (direction, maps etc).

        Parameters
        ----------
        path_suffix : string
            Text to be appended to the output path (default : None)
        """
        if self.RESULTS is None :
            raise RuntimeError( 'Model not fitted to the data; call "fit()" first.' )

        RESULTS_path = pjoin( 'AMICO', self.model.id )
        if path_suffix :
            # self.CONFIG['path_suffix'] = path_suffix
            RESULTS_path = RESULTS_path +'_'+ path_suffix
        self.RESULTS['RESULTS_path'] = RESULTS_path
        print '\n-> Saving output to "%s/*":' % RESULTS_path

        # delete previous output
        RESULTS_path = pjoin( self.CONFIG['DATA_path'], RESULTS_path )
        if not exists( RESULTS_path ) :
            makedirs( RESULTS_path )
        else :
            for f in glob.glob( pjoin(RESULTS_path,'*') ) :
                remove( f )

        # configuration
        print '\t- configuration',
        with open( pjoin(RESULTS_path,'config.pickle'), 'wb+' ) as fid :
            cPickle.dump( self.CONFIG, fid, protocol=2 )
        print ' [OK]'

        # estimated orientations
        print '\t- FIT_dir.nii',
        niiMAP_img = self.RESULTS['DIRs']
        affine     = self.niiDWI.affine if nibabel.__version__ >= '2.0.0' else self.niiDWI.get_affine()
        niiMAP     = nibabel.Nifti1Image( niiMAP_img, affine )
        niiMAP_hdr = niiMAP.header if nibabel.__version__ >= '2.0.0' else niiMAP.get_header()
        niiMAP_hdr['cal_min'] = -1
        niiMAP_hdr['cal_max'] = 1
        nibabel.save( niiMAP, pjoin(RESULTS_path, 'FIT_dir.nii.gz') )
        print ' [OK]'

        # fitting error
        if self.CONFIG['doComputeNRMSE'] :
            print '\t- FIT_nrmse.nii',
            niiMAP_img = self.RESULTS['NRMSE']
            niiMAP     = nibabel.Nifti1Image( niiMAP_img, affine )
            niiMAP_hdr = niiMAP.header if nibabel.__version__ >= '2.0.0' else niiMAP.get_header()
            niiMAP_hdr['cal_min'] = 0
            niiMAP_hdr['cal_max'] = 1
            nibabel.save( niiMAP, pjoin(RESULTS_path, 'FIT_nrmse.nii.gz') )
            print ' [OK]'

        # voxelwise maps
        for i in xrange( len(self.model.OUTPUT_names) ) :
            print '\t- AMICO/FIT_%s.nii' % self.model.OUTPUT_names[i],
            niiMAP_img = self.RESULTS['MAPs'][:,:,:,i]
            niiMAP     = nibabel.Nifti1Image( niiMAP_img, affine )
            niiMAP_hdr = niiMAP.header if nibabel.__version__ >= '2.0.0' else niiMAP.get_header()
            niiMAP_hdr['descrip'] = self.model.OUTPUT_descriptions[i]
            niiMAP_hdr['cal_min'] = niiMAP_img.min()
            niiMAP_hdr['cal_max'] = niiMAP_img.max()
            nibabel.save( niiMAP, pjoin(RESULTS_path, 'FIT_%s.nii.gz' % self.model.OUTPUT_names[i] ) )
            print ' [OK]'

        print '   [ DONE ]'
