import numpy as np
import numpy.matlib as matlib
import scipy
from os.path import exists, join as pjoin
from os import remove
import subprocess
import tempfile
import amico.lut
from amico.progressbar import ProgressBar
from dipy.core.gradients import gradient_table
from dipy.sims.voxel import single_tensor
import abc

import warnings
warnings.filterwarnings("ignore") # needed for a problem with spams
warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
    "[WARNING] %s " % message

# import the spams module, which is used only to fit the models in AMICO.
# But, on the other hand, using the models from COMMIT does not require that!
try :
    import spams
except ImportError:
    warnings.warn('Module "spams" does not seems to be installed; perhaps you will not be able to call the fit() functions of some models.')


class BaseModel( object ) :
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
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
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


    @abc.abstractmethod
    def set( self, *args, **kwargs ) :
        """For setting all the parameters specific to the model.
        NB: the parameters are model-dependent.
        """
        return


    @abc.abstractmethod
    def set_solver( self, *args, **kwargs ) :
        """For setting the parameters required by the solver to fit the model.
        NB: the parameters are model-dependent.

        Returns
        -------
        params : dictionary
            All the parameters that the solver will need to fit the model
        """
        return


    @abc.abstractmethod
    def generate( self, out_path, aux, idx_in, idx_out ) :
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
        """
        return


    @abc.abstractmethod
    def resample( self, in_path, idx_out, Ylm_out ) :
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

        Returns
        -------
        KERNELS : dictionary
            Contains the LUT and all corresponding details. In particular, it is
            required to have a field 'model' set to "self.if".
        """
        # KERNELS = {}
        # KERNELS['model'] = self.id
        # KERNELS['IC']    = np.zeros( (len(self.Rs),181,181,self.scheme.nS), dtype=np.float32 )
        # KERNELS['EC']    = np.zeros( (len(self.ICVFs),181,181,self.scheme.nS), dtype=np.float32 )
        # ...
        return


    @abc.abstractmethod
    def fit( self, y, dirs, KERNELS, params ) :
        """For fitting the model to the data.
        NB: do not change the signature!

        Parameters
        ----------
        y : array
            Diffusion signal at this voxel
        dirs : list of arrays
            Directions fitted in the voxel
        KERNELS : dictionary
            Contains the LUT and all corresponding details
        params : dictionary
            Parameters to be used by the solver

        Returns
        -------
        MAPs : list of floats
            Scalar values eastimated in each voxel
        dirs_mod : list of arrays
            Updated directions (if applicable), otherwise just return dirs
        x : array
            Coefficients of the fitting
        A : array
            Actual dictionary used in the fitting
        """
        return


class StickZeppelinBall( BaseModel ) :
    """Implements the Stick-Zeppelin-Ball model [1].

    The intra-cellular contributions from within the axons are modeled as "sticks", i.e.
    tensors with a given axial diffusivity (d_par) but null perpendicular diffusivity.
    Extra-cellular contributions are modeled as tensors with the same axial diffusivity
    as the sticks (d_par) and whose perpendicular diffusivities are calculated with a
    tortuosity model as a function of the intra-cellular volume fractions (ICVFs).
    Isotropic contributions are modeled as tensors with isotropic diffusivities (d_ISOs).

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

        self.d_par  = 1.7E-3                    # Parallel diffusivity [mm^2/s]
        self.ICVFs  = np.arange(0.3,0.9,0.1)    # Intra-cellular volume fraction(s) [0..1]
        self.d_ISOs = np.array([ 3.0E-3 ])      # Isotropic diffusivitie(s) [mm^2/s]


    def set( self, d_par, ICVFs, d_ISOs ) :
        self.d_par  = d_par
        self.ICVFs  = np.array( ICVFs )
        self.d_ISOs = np.array( d_ISOs )


    def set_solver( self ) :
        raise NotImplementedError


    def generate( self, out_path, aux, idx_in, idx_out ) :
        scheme_high = amico.lut.create_high_resolution_scheme( self.scheme, b_scale=1 )
        gtab = gradient_table( scheme_high.b, scheme_high.raw[:,0:3] )

        nATOMS = 1 + len(self.ICVFs) + len(self.d_ISOs)
        progress = ProgressBar( n=nATOMS, prefix="   ", erase=True )

        # Stick
        signal = single_tensor( gtab, evals=[0, 0, self.d_par] )
        lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, False )
        np.save( pjoin( out_path, 'A_001.npy' ), lm )
        progress.update()

        # Zeppelin(s)
        for d in [ self.d_par*(1.0-ICVF) for ICVF in self.ICVFs] :
            signal = single_tensor( gtab, evals=[d, d, self.d_par] )
            lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, False )
            np.save( pjoin( out_path, 'A_%03d.npy'%progress.i ), lm )
            progress.update()

        # Ball(s)
        for d in self.d_ISOs :
            signal = single_tensor( gtab, evals=[d, d, d] )
            lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, True )
            np.save( pjoin( out_path, 'A_%03d.npy'%progress.i ), lm )
            progress.update()


    def resample( self, in_path, idx_out, Ylm_out ) :
        KERNELS = {}
        KERNELS['model'] = self.id
        KERNELS['wmr']   = np.zeros( (1,181,181,self.scheme.nS), dtype=np.float32 )
        KERNELS['wmh']   = np.zeros( (len(self.ICVFs),181,181,self.scheme.nS), dtype=np.float32 )
        KERNELS['iso']   = np.zeros( (len(self.d_ISOs),self.scheme.nS), dtype=np.float32 )

        nATOMS = 1 + len(self.ICVFs) + len(self.d_ISOs)
        progress = ProgressBar( n=nATOMS, prefix="   ", erase=True )

        # Stick
        lm = np.load( pjoin( in_path, 'A_001.npy' ) )
        KERNELS['wmr'][0,...] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, False )
        progress.update()

        # Zeppelin(s)
        for i in xrange(len(self.ICVFs)) :
            lm = np.load( pjoin( in_path, 'A_%03d.npy'%progress.i ) )
            KERNELS['wmh'][i,...] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, False )
            progress.update()

        # Ball(s)
        for i in xrange(len(self.d_ISOs)) :
            lm = np.load( pjoin( in_path, 'A_%03d.npy'%progress.i ) )
            KERNELS['iso'][i,...] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, True )
            progress.update()

        return KERNELS


    def fit( self, y, dirs, KERNELS, params ) :
        raise NotImplementedError


class CylinderZeppelinBall( BaseModel ) :
    """Implements the Cylinder-Zeppelin-Ball model [1].

    The intra-cellular contributions from within the axons are modeled as "cylinders"
    with specific radii (Rs) and a given axial diffusivity (d_par).
    Extra-cellular contributions are modeled as tensors with the same axial diffusivity
    as the cylinders (d_par) and whose perpendicular diffusivities are calculated with a
    tortuosity model as a function of the intra-cellular volume fractions (ICVFs).
    Isotropic contributions are modeled as tensors with isotropic diffusivities (d_ISOs).

    NB: this model works only with schemes containing the full specification of
        the diffusion gradients (eg gradient strength, small delta etc).

    NB: this model requires Camino to be installed and properly configured
        in the system; in particular, the script "datasynth" must be placed
        in your system path.

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

        self.d_par  = 0.6E-3                                                         # Parallel diffusivity [mm^2/s]
        self.Rs     = np.concatenate( ([0.01],np.linspace(0.5,8.0,20.0)) ) * 1E-6    # Radii of the axons [meters]
        self.ICVFs  = np.arange(0.3,0.9,0.1)                                         # Intra-cellular volume fraction(s) [0..1]
        self.d_ISOs = np.array( [ 2.0E-3 ] )                                         # Isotropic diffusivitie(s) [mm^2/s]
        self.isExvivo  = False                                                       # Add dot compartment to dictionary (exvivo data)
        self.singleb0  = True                                                        # Merge b0 images into a single volume for fitting


    def set( self, d_par, Rs, ICVFs, d_ISOs ) :
        self.d_par  = d_par
        self.Rs     = np.array(Rs)
        self.ICVFs  = np.array(ICVFs)
        self.d_ISOs = np.array(d_ISOs)


    def set_solver( self, lambda1 = 0.0, lambda2 = 4.0 ) :
        params = {}
        params['mode']    = 2
        params['pos']     = True
        params['lambda1'] = lambda1
        params['lambda2'] = lambda2
        return params


    def generate( self, out_path, aux, idx_in, idx_out ) :
        if self.scheme.version != 1 :
            raise RuntimeError( 'This model requires a "VERSION: STEJSKALTANNER" scheme.' )

        scheme_high = amico.lut.create_high_resolution_scheme( self.scheme, b_scale=1E6 )
        filename_scheme = pjoin( out_path, 'scheme.txt' )
        np.savetxt( filename_scheme, scheme_high.raw, fmt='%15.8e', delimiter=' ', header='VERSION: STEJSKALTANNER', comments='' )

        # temporary file where to store "datasynth" output
        filename_signal = pjoin( tempfile._get_default_tempdir(), next(tempfile._get_candidate_names())+'.Bfloat' )

        nATOMS = len(self.Rs) + len(self.ICVFs) + len(self.d_ISOs)
        progress = ProgressBar( n=nATOMS, prefix="   ", erase=True )

        # Cylinder(s)
        for R in self.Rs :
            CMD = 'datasynth -synthmodel compartment 1 CYLINDERGPD %E 0 0 %E -schemefile %s -voxels 1 -outputfile %s 2> /dev/null' % ( self.d_par*1E-6, R, filename_scheme, filename_signal )
            subprocess.call( CMD, shell=True )
            if not exists( filename_signal ) :
                raise RuntimeError( 'Problems generating the signal with "datasynth"' )
            signal  = np.fromfile( filename_signal, dtype='>f4' )
            if exists( filename_signal ) :
                remove( filename_signal )

            lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, False )
            np.save( pjoin( out_path, 'A_%03d.npy'%progress.i ), lm )
            progress.update()

        # Zeppelin(s)
        for d in [ self.d_par*(1.0-ICVF) for ICVF in self.ICVFs] :
            CMD = 'datasynth -synthmodel compartment 1 ZEPPELIN %E 0 0 %E -schemefile %s -voxels 1 -outputfile %s 2> /dev/null' % ( self.d_par*1E-6, d*1e-6, filename_scheme, filename_signal )
            subprocess.call( CMD, shell=True )
            if not exists( filename_signal ) :
                raise RuntimeError( 'Problems generating the signal with "datasynth"' )
            signal  = np.fromfile( filename_signal, dtype='>f4' )
            if exists( filename_signal ) :
                remove( filename_signal )

            lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, False )
            np.save( pjoin( out_path, 'A_%03d.npy'%progress.i ), lm )
            progress.update()

        # Ball(s)
        for d in self.d_ISOs :
            CMD = 'datasynth -synthmodel compartment 1 BALL %E -schemefile %s -voxels 1 -outputfile %s 2> /dev/null' % ( d*1e-6, filename_scheme, filename_signal )
            subprocess.call( CMD, shell=True )
            if not exists( filename_signal ) :
                raise RuntimeError( 'Problems generating the signal with "datasynth"' )
            signal  = np.fromfile( filename_signal, dtype='>f4' )
            if exists( filename_signal ) :
                remove( filename_signal )

            lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, True )
            np.save( pjoin( out_path, 'A_%03d.npy'%progress.i ), lm )
            progress.update()


    def resample( self, in_path, idx_out, Ylm_out ) :
        KERNELS = {}
        KERNELS['model'] = self.id
        KERNELS['wmr'] = np.zeros( (len(self.Rs),181,181,self.scheme.nS,), dtype=np.float32 )
        KERNELS['wmh'] = np.zeros( (len(self.ICVFs),181,181,self.scheme.nS,), dtype=np.float32 )
        KERNELS['iso'] = np.zeros( (len(self.d_ISOs),self.scheme.nS,), dtype=np.float32 )

        nATOMS = len(self.Rs) + len(self.ICVFs) + len(self.d_ISOs)
        progress = ProgressBar( n=nATOMS, prefix="   ", erase=True )

        # Cylinder(s)
        for i in xrange(len(self.Rs)) :
            lm = np.load( pjoin( in_path, 'A_%03d.npy'%progress.i ) )
            KERNELS['wmr'][i,:,:,:] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, False )
            progress.update()

        # Zeppelin(s)
        for i in xrange(len(self.ICVFs)) :
            lm = np.load( pjoin( in_path, 'A_%03d.npy'%progress.i ) )
            KERNELS['wmh'][i,:,:,:] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, False )
            progress.update()

        # Ball(s)
        for i in xrange(len(self.d_ISOs)) :
            lm = np.load( pjoin( in_path, 'A_%03d.npy'%progress.i ) )
            KERNELS['iso'][i,:] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, True )
            progress.update()

        return KERNELS


    def fit( self, y, dirs, KERNELS, params ) :
        nD = dirs.shape[0]
        n1 = len(self.Rs)
        n2 = len(self.ICVFs)
        n3 = len(self.d_ISOs)
        if self.isExvivo:
            nATOMS = nD*(n1+n2)+n3+1
        else:
            nATOMS = nD*(n1+n2)+n3
        if self.singleb0:
            # prepare DICTIONARY from dirs and lookup tables
            A = np.ones( (1+self.scheme.dwi_count, nATOMS ), dtype=np.float64, order='F' )
            o = 0
            for i in xrange(nD) :
                i1, i2 = amico.lut.dir_TO_lut_idx( dirs[i] )
                A[1:,o:(o+n1)] = KERNELS['wmr'][:,i1,i2,self.scheme.dwi_idx].T
                o += n1
            for i in xrange(nD) :
                i1, i2 = amico.lut.dir_TO_lut_idx( dirs[i] )
                A[1:,o:(o+n2)] = KERNELS['wmh'][:,i1,i2,self.scheme.dwi_idx].T
                o += n2
            A[1:,o:o+n3] = KERNELS['iso'][:,self.scheme.dwi_idx].T
            y = np.hstack((y[self.scheme.b0_idx].mean(),y[self.scheme.dwi_idx]))
        else:
            # prepare DICTIONARY from dirs and lookup tables
            A = np.ones( (self.scheme.nS, nATOMS ), dtype=np.float64, order='F' )
            o = 0
            for i in xrange(nD) :
                i1, i2 = amico.lut.dir_TO_lut_idx( dirs[i] )
                A[:,o:(o+n1)] = KERNELS['wmr'][:,i1,i2,:].T
                o += n1
            for i in xrange(nD) :
                i1, i2 = amico.lut.dir_TO_lut_idx( dirs[i] )
                A[:,o:(o+n2)] = KERNELS['wmh'][:,i1,i2,:].T
                o += n2
            A[:,o:] = KERNELS['iso'].T

        # empty dictionary
        if A.shape[1] == 0 :
            return [0, 0, 0], None, None, None

        # fit
        x = spams.lasso( np.asfortranarray( y.reshape(-1,1) ), D=A, **params ).todense().A1

        # return estimates
        f1 = x[ :(nD*n1) ].sum()
        f2 = x[ (nD*n1):(nD*(n1+n2)) ].sum()
        v = f1 / ( f1 + f2 + 1e-16 )
        xIC = x[:nD*n1].reshape(-1,n1).sum(axis=0)
        a = 1E6 * 2.0 * np.dot(self.Rs,xIC) / ( f1 + 1e-16 )
        d = (4.0*v) / ( np.pi*a**2 + 1e-16 )
        return [v, a, d], dirs, x, A


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
        self.maps_name  = [ 'ICVF', 'OD', 'ISOVF' ]
        self.maps_descr = [ 'Intra-cellular volume fraction', 'Orientation dispersion', 'Isotropic volume fraction' ]

        self.dPar      = 1.7E-3
        self.dIso      = 3.0E-3
        self.IC_VFs    = np.linspace(0.1,0.99,12)
        self.IC_ODs    = np.hstack((np.array([0.03, 0.06]),np.linspace(0.09,0.99,10)))
        self.isExvivo  = False


    def set( self, dPar, dIso, IC_VFs, IC_ODs, isExvivo ):
        self.dPar      = dPar
        self.dIso      = dIso
        self.IC_VFs    = np.array( IC_VFs )
        self.IC_ODs    = np.array( IC_ODs )
        self.isExvivo  = isExvivo


    def set_solver( self, lambda1 = 5e-1, lambda2 = 1e-3 ):
        params = {}
        params['mode']    = 2
        params['pos']     = True
        params['lambda1'] = lambda1
        params['lambda2'] = lambda2
        return params


    def generate( self, out_path, aux, idx_in, idx_out ):
        scheme_high = amico.lut.create_high_resolution_scheme( self.scheme, b_scale = 1 )
        protocolHR = self.scheme2noddi( scheme_high )

        nATOMS = len(self.IC_ODs)*len(self.IC_VFs) + 1
        progress = ProgressBar( n=nATOMS, prefix="   ", erase=True )

        # Coupled contributions
        IC_KAPPAs = 1 / np.tan(self.IC_ODs*np.pi/2)
        for kappa in IC_KAPPAs:
            signal_ic = self.synth_meas_watson_SH_cyl_neuman_PGSE( np.array([self.dPar*1E-6, 0, kappa]), protocolHR['grad_dirs'], np.squeeze(protocolHR['gradient_strength']), np.squeeze(protocolHR['delta']), np.squeeze(protocolHR['smalldel']), np.array([0,0,1]), 0 )

            for v_ic in self.IC_VFs:
                dPerp = self.dPar*1E-6 * (1 - v_ic)
                signal_ec = self.synth_meas_watson_hindered_diffusion_PGSE( np.array([self.dPar*1E-6, dPerp, kappa]), protocolHR['grad_dirs'], np.squeeze(protocolHR['gradient_strength']), np.squeeze(protocolHR['delta']), np.squeeze(protocolHR['smalldel']), np.array([0,0,1]) )

                signal = v_ic*signal_ic + (1-v_ic)*signal_ec
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, False )
                np.save( pjoin( out_path, 'A_%03d.npy'%progress.i) , lm )
                progress.update()

        # Isotropic
        signal = self.synth_meas_iso_GPD( self.dIso*1E-6, protocolHR)
        lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, True )
        np.save( pjoin( out_path, 'A_%03d.npy'%progress.i) , lm )
        progress.update()


    def resample( self, in_path, idx_out, Ylm_out ):
        nATOMS = len(self.IC_ODs)*len(self.IC_VFs) + 1

        KERNELS = {}
        KERNELS['model'] = self.id
        KERNELS['wm']    = np.zeros( (nATOMS-1,181,181,self.scheme.nS), dtype=np.float32 )
        KERNELS['iso']   = np.zeros( self.scheme.nS, dtype=np.float32 )
        KERNELS['kappa'] = np.zeros( nATOMS-1, dtype=np.float32 )
        KERNELS['icvf']  = np.zeros( nATOMS-1, dtype=np.float32 )
        KERNELS['norms'] = np.zeros( (self.scheme.dwi_count, nATOMS-1) )

        progress = ProgressBar( n=nATOMS, prefix="   ", erase=True )

        # Coupled contributions
        for i in xrange( len(self.IC_ODs) ):
            for j in xrange( len(self.IC_VFs) ):
                lm = np.load( pjoin( in_path, 'A_%03d.npy'%progress.i ) )
                idx = progress.i - 1
                KERNELS['wm'][idx,:,:,:] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, False )
                KERNELS['kappa'][idx] = 1.0 / np.tan( self.IC_ODs[i]*np.pi/2.0 )
                KERNELS['icvf'][idx]  = self.IC_VFs[j]
                KERNELS['norms'][:,idx] = 1 / np.linalg.norm( KERNELS['wm'][idx,0,0,self.scheme.dwi_idx] ) # norm of coupled atoms (for l1 minimization)
                progress.update()

        # Isotropic
        lm = np.load( pjoin( in_path, 'A_%03d.npy'%progress.i ) )
        KERNELS['iso'] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, True )
        progress.update()

        return KERNELS


    def fit( self, y, dirs, KERNELS, params ) :
        nD = dirs.shape[0]
        if nD != 1 :
            raise RuntimeError( '"%s" model requires exactly 1 orientation' % self.name )

        # prepare DICTIONARY from dir and lookup tables
        nATOMS = len(self.IC_ODs)*len(self.IC_VFs) + 1
        if self.isExvivo == True :
            nATOMS += 1
        A = np.ones( (len(y), nATOMS), dtype=np.float64, order='F' )
        i1, i2 = amico.lut.dir_TO_lut_idx( dirs[0] )
        A[:,:-1] = KERNELS['wm'][:,i1,i2,:].T
        A[:,-1]  = KERNELS['iso']

        # estimate CSF partial volume (and isotropic restriction, if exvivo) and remove from signal
        x, _ = scipy.optimize.nnls( A, y )
        yy = y - x[-1]*A[:,-1]
        if self.isExvivo == True :
            yy = yy - x[-2]*A[:,-2]
        yy[ yy<0 ] = 0

        # estimate IC and EC compartments and promote sparsity
        An = A[ self.scheme.dwi_idx, :-1 ] * KERNELS['norms']
        yy = yy[ self.scheme.dwi_idx ].reshape(-1,1)
        x = spams.lasso( np.asfortranarray(yy), D=np.asfortranarray(An), **params ).todense().A1

        # debias coefficients
        x = np.append( x, 1 )
        if self.isExvivo == True :
            x = np.append( x, 1 )
        idx = x>0
        x[idx], _ = scipy.optimize.nnls( A[:,idx], y )

        # return estimates
        xx = x / ( x.sum() + 1e-16 )
        if self.isExvivo == True :
            xWM  = xx[:-2]
            fISO = xx[-2]
        else :
            xWM  = xx[:-1]
            fISO = xx[-1]
        xWM = xWM / ( xWM.sum() + 1e-16 )
        f1 = np.dot( KERNELS['icvf'], xWM )
        f2 = np.dot( (1.0-KERNELS['icvf']), xWM )
        v = f1 / ( f1 + f2 + 1e-16 )
        k = np.dot( KERNELS['kappa'], xWM )
        od = 2.0/np.pi * np.arctan2(1.0,k)

        return [v, od, fISO], dirs, x, A


    def scheme2noddi( self, scheme ):
        protocol = {}
        protocol['pulseseq'] = 'PGSE'
        protocol['schemetype'] = 'multishellfixedG'
        protocol['teststrategy'] = 'fixed'
        bval = scheme.b.copy()

        # set total number of measurements
        protocol['totalmeas'] = len(bval)

        # set the b=0 indices
        protocol['b0_Indices'] = np.nonzero(bval==0)[0]
        protocol['numZeros'] = len(protocol['b0_Indices'])

        # find the unique non-zero b-values
        B = np.unique(bval[bval>0])

        # set the number of shells
        protocol['M'] = len(B)
        protocol['N'] = np.zeros((len(B)))
        for i in range(len(B)):
            protocol['N'][i] = np.sum(bval==B[i])

        # maximum b-value in the s/mm^2 unit
        maxB = np.max(B)

        # set maximum G = 40 mT/m
        Gmax = 0.04

        # set smalldel and delta and G
        GAMMA = 2.675987E8
        tmp = np.power(3*maxB*1E6/(2*GAMMA*GAMMA*Gmax*Gmax),1.0/3.0)
        protocol['udelta'] = np.zeros((len(B)))
        protocol['usmalldel'] = np.zeros((len(B)))
        protocol['uG'] = np.zeros((len(B)))
        for i in range(len(B)):
            protocol['udelta'][i] = tmp
            protocol['usmalldel'][i] = tmp
            protocol['uG'][i] = np.sqrt(B[i]/maxB)*Gmax

        protocol['delta'] = np.zeros(bval.shape)
        protocol['smalldel'] = np.zeros(bval.shape)
        protocol['gradient_strength'] = np.zeros(bval.shape)

        for i in range(len(B)):
            tmp = np.nonzero(bval==B[i])
            for j in range(len(tmp[0])):
                    protocol['delta'][tmp[0][j]] = protocol['udelta'][i]
                    protocol['smalldel'][tmp[0][j]] = protocol['usmalldel'][i]
                    protocol['gradient_strength'][tmp[0][j]] = protocol['uG'][i]

        # load bvec
        protocol['grad_dirs'] = scheme.raw[:,0:3].copy()

        # make the gradient directions for b=0's [1 0 0]
        for i in range(protocol['numZeros']):
            protocol['grad_dirs'][protocol['b0_Indices'][i],:] = [1, 0, 0]

        # make sure the gradient directions are unit vectors
        for i in range(protocol['totalmeas']):
            protocol['grad_dirs'][i,:] = protocol['grad_dirs'][i,:]/np.linalg.norm(protocol['grad_dirs'][i,:])

        return protocol

    def synth_meas_watson_SH_cyl_neuman_PGSE( self, x, grad_dirs, G, delta, smalldel, fibredir, roots ):
        d=x[0]
        R=x[1]
        kappa=x[2]

        l_q = grad_dirs.shape[0]

        # Parallel component
        LePar = self.cyl_neuman_le_par_PGSE(d, G, delta, smalldel)

        # Perpendicular component
        LePerp = self.cyl_neuman_le_perp_PGSE(d, R, G, delta, smalldel, roots)

        ePerp = np.exp(LePerp)

        # Compute the Legendre weighted signal
        Lpmp = LePerp - LePar
        lgi = self.legendre_gaussian_integral(Lpmp, 6)

        # Compute the spherical harmonic coefficients of the Watson's distribution
        coeff = self.watson_SH_coeff(kappa)
        coeffMatrix = matlib.repmat(coeff, l_q, 1)

        # Compute the dot product between the symmetry axis of the Watson's distribution
        # and the gradient direction
        #
        # For numerical reasons, cosTheta might not always be between -1 and 1
        # Due to round off errors, individual gradient vectors in grad_dirs and the
        # fibredir are never exactly normal.  When a gradient vector and fibredir are
        # essentially parallel, their dot product can fall outside of -1 and 1.
        #
        # BUT we need make sure it does, otherwise the legendre function call below
        # will FAIL and abort the calculation!!!
        #
        cosTheta = np.dot(grad_dirs,fibredir)
        badCosTheta = abs(cosTheta)>1
        cosTheta[badCosTheta] = cosTheta[badCosTheta]/abs(cosTheta[badCosTheta])

        # Compute the SH values at cosTheta
        sh = np.zeros(coeff.shape)
        shMatrix = matlib.repmat(sh, l_q, 1)
        for i in range(7):
            shMatrix[:,i] = np.sqrt((i+1 - .75)/np.pi)
            # legendre function returns coefficients of all m from 0 to l
            # we only need the coefficient corresponding to m = 0
            # WARNING: make sure to input ROW vector as variables!!!
            # cosTheta is expected to be a COLUMN vector.
            tmp = np.zeros((l_q))
            for pol_i in range(l_q):
                tmp[pol_i] = scipy.special.lpmv(0, 2*i, cosTheta[pol_i])
            shMatrix[:,i] = shMatrix[:,i]*tmp

        E = np.sum(lgi*coeffMatrix*shMatrix, 1)
        # with the SH approximation, there will be no guarantee that E will be positive
        # but we need to make sure it does!!! replace the negative values with 10% of
        # the smallest positive values
        E[E<=0] = np.min(E[E>0])*0.1
        E = 0.5*E*ePerp

        return E

    def synth_meas_watson_hindered_diffusion_PGSE( self, x, grad_dirs, G, delta, smalldel, fibredir ):

        dPar = x[0]
        dPerp = x[1]
        kappa = x[2]

        # get the equivalent diffusivities
        dw = self.watson_hindered_diffusion_coeff(dPar, dPerp, kappa)

        xh = np.array([dw[0], dw[1]])

        E = self.synth_meas_hindered_diffusion_PGSE(xh, grad_dirs, G, delta, smalldel, fibredir)

        return E

    def cyl_neuman_le_par_PGSE( self, d, G, delta, smalldel ):
        # Line bellow used in matlab version removed as cyl_neuman_le_par_PGSE is called from synth_meas_watson_SH_cyl_neuman_PGSE which already casts x to d, R and kappa -> x replaced by d in arguments
        #d=x[0]

        # Radial wavenumbers
        GAMMA = 2.675987E8
        modQ = GAMMA*smalldel*G
        modQ_Sq = modQ*modQ

        # diffusion time for PGSE, in a matrix for the computation below.
        difftime = (delta-smalldel/3)

        # Parallel component
        LE =-modQ_Sq*difftime*d

        # Compute the Jacobian matrix
        #if(nargout>1)
        #    % dLE/d
        #    J = -modQ_Sq*difftime
        #end

        return LE

    def cyl_neuman_le_perp_PGSE( self, d, R, G, delta, smalldel, roots ):

        # When R=0, no need to do any calculation
        if (R == 0.00):
            LE = np.zeros(G.shape) # np.size(R) = 1
            return LE
        else:
            msg = "Python implementation for function noddi.cyl_neuman_le_perp_PGSE not yet validated for non-zero values"
            raise ValueError(msg)

    def legendre_gaussian_integral( self, Lpmp, n ):
        if n > 6:
            msg = 'The maximum value for n is 6, which correspondes to the 12th order Legendre polynomial'
            raise ValueError(msg)
        exact = Lpmp>0.05
        approx = Lpmp<=0.05

        mn = n + 1

        I = np.zeros((len(Lpmp),mn))
        sqrtx = np.sqrt(Lpmp[exact])
        I[exact,0] = np.sqrt(np.pi)*scipy.special.erf(sqrtx)/sqrtx
        dx = 1.0/Lpmp[exact]
        emx = -np.exp(-Lpmp[exact])
        for i in range(1,mn):
            I[exact,i] = emx + (i-0.5)*I[exact,i-1]
            I[exact,i] = I[exact,i]*dx

        # Computing the legendre gaussian integrals for large enough Lpmp
        L = np.zeros((len(Lpmp),n+1))
        for i in range(n+1):
            if i == 0:
                L[exact,0] = I[exact,0]
            elif i == 1:
                L[exact,1] = -0.5*I[exact,0] + 1.5*I[exact,1]
            elif i == 2:
                L[exact,2] = 0.375*I[exact,0] - 3.75*I[exact,1] + 4.375*I[exact,2]
            elif i == 3:
                L[exact,3] = -0.3125*I[exact,0] + 6.5625*I[exact,1] - 19.6875*I[exact,2] + 14.4375*I[exact,3]
            elif i == 4:
                L[exact,4] = 0.2734375*I[exact,0] - 9.84375*I[exact,1] + 54.140625*I[exact,2] - 93.84375*I[exact,3] + 50.2734375*I[exact,4]
            elif i == 5:
                L[exact,5] = -(63./256.)*I[exact,0] + (3465./256.)*I[exact,1] - (30030./256.)*I[exact,2] + (90090./256.)*I[exact,3] - (109395./256.)*I[exact,4] + (46189./256.)*I[exact,5]
            elif i == 6:
                L[exact,6] = (231./1024.)*I[exact,0] - (18018./1024.)*I[exact,1] + (225225./1024.)*I[exact,2] - (1021020./1024.)*I[exact,3] + (2078505./1024.)*I[exact,4] - (1939938./1024.)*I[exact,5] + (676039./1024.)*I[exact,6]

        # Computing the legendre gaussian integrals for small Lpmp
        x2=np.power(Lpmp[approx],2)
        x3=x2*Lpmp[approx]
        x4=x3*Lpmp[approx]
        x5=x4*Lpmp[approx]
        x6=x5*Lpmp[approx]
        for i in range(n+1):
            if i == 0:
                L[approx,0] = 2 - 2*Lpmp[approx]/3 + x2/5 - x3/21 + x4/108
            elif i == 1:
                L[approx,1] = -4*Lpmp[approx]/15 + 4*x2/35 - 2*x3/63 + 2*x4/297
            elif i == 2:
                L[approx,2] = 8*x2/315 - 8*x3/693 + 4*x4/1287
            elif i == 3:
                L[approx,3] = -16*x3/9009 + 16*x4/19305
            elif i == 4:
                L[approx,4] = 32*x4/328185
            elif i == 5:
                L[approx,5] = -64*x5/14549535
            elif i == 6:
                L[approx,6] = 128*x6/760543875

        return L

    def watson_SH_coeff( self, kappa ):

        if isinstance(kappa,np.ndarray):
            msg = 'noddi.py : watson_SH_coeff() not implemented for multiple kappa input yet.'
            raise ValueError(msg)

        # In the scope of AMICO only a single value is used for kappa
        n = 6

        C = np.zeros((n+1))
        # 0th order is a constant
        C[0] = 2*np.sqrt(np.pi)

        # Precompute the special function values
        sk = np.sqrt(kappa)
        sk2 = sk*kappa
        sk3 = sk2*kappa
        sk4 = sk3*kappa
        sk5 = sk4*kappa
        sk6 = sk5*kappa
        sk7 = sk6*kappa
        k2 = np.power(kappa,2)
        k3 = k2*kappa
        k4 = k3*kappa
        k5 = k4*kappa
        k6 = k5*kappa
        k7 = k6*kappa

        erfik = scipy.special.erfi(sk)
        ierfik = 1/erfik
        ek = np.exp(kappa)
        dawsonk = 0.5*np.sqrt(np.pi)*erfik/ek

        if kappa > 0.1:

            # for large enough kappa
            C[1] = 3*sk - (3 + 2*kappa)*dawsonk
            C[1] = np.sqrt(5)*C[1]*ek
            C[1] = C[1]*ierfik/kappa

            C[2] = (105 + 60*kappa + 12*k2)*dawsonk
            C[2] = C[2] -105*sk + 10*sk2
            C[2] = .375*C[2]*ek/k2
            C[2] = C[2]*ierfik

            C[3] = -3465 - 1890*kappa - 420*k2 - 40*k3
            C[3] = C[3]*dawsonk
            C[3] = C[3] + 3465*sk - 420*sk2 + 84*sk3
            C[3] = C[3]*np.sqrt(13*np.pi)/64/k3
            C[3] = C[3]/dawsonk

            C[4] = 675675 + 360360*kappa + 83160*k2 + 10080*k3 + 560*k4
            C[4] = C[4]*dawsonk
            C[4] = C[4] - 675675*sk + 90090*sk2 - 23100*sk3 + 744*sk4
            C[4] = np.sqrt(17)*C[4]*ek
            C[4] = C[4]/512/k4
            C[4] = C[4]*ierfik

            C[5] = -43648605 - 22972950*kappa - 5405400*k2 - 720720*k3 - 55440*k4 - 2016*k5
            C[5] = C[5]*dawsonk
            C[5] = C[5] + 43648605*sk - 6126120*sk2 + 1729728*sk3 - 82368*sk4 + 5104*sk5
            C[5] = np.sqrt(21*np.pi)*C[5]/4096/k5
            C[5] = C[5]/dawsonk

            C[6] = 7027425405 + 3666482820*kappa + 872972100*k2 + 122522400*k3  + 10810800*k4 + 576576*k5 + 14784*k6
            C[6] = C[6]*dawsonk
            C[6] = C[6] - 7027425405*sk + 1018467450*sk2 - 302630328*sk3 + 17153136*sk4 - 1553552*sk5 + 25376*sk6
            C[6] = 5*C[6]*ek
            C[6] = C[6]/16384/k6
            C[6] = C[6]*ierfik

        # for very large kappa
        if kappa>30:
            lnkd = np.log(kappa) - np.log(30)
            lnkd2 = lnkd*lnkd
            lnkd3 = lnkd2*lnkd
            lnkd4 = lnkd3*lnkd
            lnkd5 = lnkd4*lnkd
            lnkd6 = lnkd5*lnkd
            C[1] = 7.52308 + 0.411538*lnkd - 0.214588*lnkd2 + 0.0784091*lnkd3 - 0.023981*lnkd4 + 0.00731537*lnkd5 - 0.0026467*lnkd6
            C[2] = 8.93718 + 1.62147*lnkd - 0.733421*lnkd2 + 0.191568*lnkd3 - 0.0202906*lnkd4 - 0.00779095*lnkd5 + 0.00574847*lnkd6
            C[3] = 8.87905 + 3.35689*lnkd - 1.15935*lnkd2 + 0.0673053*lnkd3 + 0.121857*lnkd4 - 0.066642*lnkd5 + 0.0180215*lnkd6
            C[4] = 7.84352 + 5.03178*lnkd - 1.0193*lnkd2 - 0.426362*lnkd3 + 0.328816*lnkd4 - 0.0688176*lnkd5 - 0.0229398*lnkd6
            C[5] = 6.30113 + 6.09914*lnkd - 0.16088*lnkd2 - 1.05578*lnkd3 + 0.338069*lnkd4 + 0.0937157*lnkd5 - 0.106935*lnkd6
            C[6] = 4.65678 + 6.30069*lnkd + 1.13754*lnkd2 - 1.38393*lnkd3 - 0.0134758*lnkd4 + 0.331686*lnkd5 - 0.105954*lnkd6

        if kappa <= 0.1:
            # for small kappa
            C[1] = 4/3*kappa + 8/63*k2
            C[1] = C[1]*np.sqrt(np.pi/5)

            C[2] = 8/21*k2 + 32/693*k3
            C[2] = C[2]*(np.sqrt(np.pi)*0.2)

            C[3] = 16/693*k3 + 32/10395*k4
            C[3] = C[3]*np.sqrt(np.pi/13)

            C[4] = 32/19305*k4
            C[4] = C[4]*np.sqrt(np.pi/17)

            C[5] = 64*np.sqrt(np.pi/21)*k5/692835

            C[6] = 128*np.sqrt(np.pi)*k6/152108775

        return C

    def watson_hindered_diffusion_coeff( self, dPar, dPerp, kappa ):

        dw = np.zeros(2)
        dParMdPerp = dPar - dPerp

        if kappa < 1e-5:
            dParP2dPerp = dPar + 2.*dPerp
            k2 = kappa*kappa
            dw[0] = dParP2dPerp/3.0 + 4.0*dParMdPerp*kappa/45.0 + 8.0*dParMdPerp*k2/945.0
            dw[1] = dParP2dPerp/3.0 - 2.0*dParMdPerp*kappa/45.0 - 4.0*dParMdPerp*k2/945.0
        else:
            sk = np.sqrt(kappa)
            dawsonf = 0.5*np.exp(-kappa)*np.sqrt(np.pi)*scipy.special.erfi(sk)
            factor = sk/dawsonf
            dw[0] = (-dParMdPerp+2.0*dPerp*kappa+dParMdPerp*factor)/(2.0*kappa)
            dw[1] = (dParMdPerp+2.0*(dPar+dPerp)*kappa-dParMdPerp*factor)/(4.0*kappa)

        return dw

    def synth_meas_hindered_diffusion_PGSE( self, x, grad_dirs, G, delta, smalldel, fibredir ):

        dPar=x[0]
        dPerp=x[1]

        # Radial wavenumbers
        GAMMA = 2.675987E8
        modQ = GAMMA*smalldel*G
        modQ_Sq = np.power(modQ,2.0)

        # Angles between gradient directions and fibre direction.
        cosTheta = np.dot(grad_dirs,fibredir)
        cosThetaSq = np.power(cosTheta,2.0)
        sinThetaSq = 1.0-cosThetaSq

        # b-value
        bval = (delta-smalldel/3.0)*modQ_Sq

        # Find hindered signals
        E=np.exp(-bval*((dPar - dPerp)*cosThetaSq + dPerp))

        return E

    def synth_meas_iso_GPD( self, d, protocol ):
        if protocol['pulseseq'] != 'PGSE' and protocol['pulseseq'] != 'STEAM':
            msg = 'synth_meas_iso_GPD() : Protocol %s not translated from NODDI matlab code yet' % protocol['pulseseq']
            raise ValueError(msg)

        GAMMA = 2.675987E8
        modQ = GAMMA*protocol['smalldel'].transpose()*protocol['gradient_strength'].transpose()
        modQ_Sq = np.power(modQ,2)
        difftime = protocol['delta'].transpose()-protocol['smalldel']/3.0
        return np.exp(-difftime*modQ_Sq*d)


class FreeWater( BaseModel ) :
    """Implements the Free-Water model.
    """
    def __init__( self ) :
        self.id         = 'FreeWater'
        self.name       = 'Free-Water'
        self.type       = 'Human'

        if self.type == 'Mouse' :
            self.maps_name  = [ 'FiberVolume', 'FW', 'FW_blood', 'FW_csf' ]
            self.maps_descr = [ 'fiber volume fraction',
                                'Isotropic free-water volume fraction',
                                'FW blood', 'FW csf' ]

            # for mouse imaging
            self.d_par   = 1.0E-3
            self.d_perps = np.linspace(0.15,0.55,10)*1E-3
            self.d_isos  = [1.5e-3, 3e-3]


        else :
            self.maps_name  = [ 'FiberVolume', 'FW' ]
            self.maps_descr = [ 'fiber volume fraction',
                                'Isotropic free-water volume fraction']
            self.d_par   = 1.0E-3                       # Parallel diffusivity [mm^2/s]
            self.d_perps = np.linspace(0.1,1.0,10)*1E-3 # Parallel diffusivities [mm^2/s]
            self.d_isos  = [ 2.5E-3 ]                   # Isotropic diffusivities [mm^2/s]


    def set( self, d_par, d_perps, d_isos, type ) :
        self.d_par   = d_par
        self.d_perps = d_perps
        self.d_isos  = d_isos
        self.type    = type

        if self.type == 'Mouse' :
            self.maps_name  = [ 'FiberVolume', 'FW', 'FW_blood', 'FW_csf' ]
            self.maps_descr = [ 'fiber volume fraction',
                                'Isotropic free-water volume fraction',
                                'FW blood', 'FW csf' ]

        else :
            self.maps_name  = [ 'FiberVolume', 'FW' ]
            self.maps_descr = [ 'fiber volume fraction',
                                'Isotropic free-water volume fraction']

        print '      %s settings for Freewater elimination... ' % self.type
        print '             -iso  compartments: ', self.d_isos
        print '             -perp compartments: ', self.d_perps
        print '             -para compartments: ', self.d_par



    def set_solver( self, lambda1 = 0.0, lambda2 = 1e-3 ):
        params = {}
        params['mode']    = 2
        params['pos']     = True
        params['lambda1'] = lambda1
        params['lambda2'] = lambda2

        # need more regul for mouse data
        if self.type == 'Mouse' :
            lambda2 = 0.25

        return params


    def generate( self, out_path, aux, idx_in, idx_out ) :
        scheme_high = amico.lut.create_high_resolution_scheme( self.scheme, b_scale=1 )
        gtab = gradient_table( scheme_high.b, scheme_high.raw[:,0:3] )

        nATOMS = len(self.d_perps) + len(self.d_isos)
        progress = ProgressBar( n=nATOMS, prefix="   ", erase=True )

        # Tensor compartment(s)
        for d in self.d_perps :
            signal = single_tensor( gtab, evals=[d, d, self.d_par] )
            lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, False )
            np.save( pjoin( out_path, 'A_%03d.npy'%progress.i ), lm )
            progress.update()

        # Isotropic compartment(s)
        for d in self.d_isos :
            signal = single_tensor( gtab, evals=[d, d, d] )
            lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, True )
            np.save( pjoin( out_path, 'A_%03d.npy'%progress.i ), lm )
            progress.update()


    def resample( self, in_path, idx_out, Ylm_out ) :
        KERNELS = {}
        KERNELS['model'] = self.id
        KERNELS['D']     = np.zeros( (len(self.d_perps),181,181,self.scheme.nS), dtype=np.float32 )
        KERNELS['CSF']   = np.zeros( (len(self.d_isos),self.scheme.nS), dtype=np.float32 )

        nATOMS = len(self.d_perps) + len(self.d_isos)
        progress = ProgressBar( n=nATOMS, prefix="   ", erase=True )

        # Tensor compartment(s)
        for i in xrange(len(self.d_perps)) :
            lm = np.load( pjoin( in_path, 'A_%03d.npy'%progress.i ) )
            KERNELS['D'][i,...] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, False )
            progress.update()

        # Isotropic compartment(s)
        for i in xrange(len(self.d_isos)) :
            lm = np.load( pjoin( in_path, 'A_%03d.npy'%progress.i ) )
            KERNELS['CSF'][i,...] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, True )
            progress.update()

        return KERNELS


    def fit( self, y, dirs, KERNELS, params ) :
        nD = dirs.shape[0]
        if nD > 1 : # model works only with one direction
            raise RuntimeError( '"%s" model requires exactly 1 orientation' % self.name )

        n1 = len(self.d_perps)
        n2 = len(self.d_isos)
        nATOMS = n1+n2
        if nATOMS == 0 : # empty dictionary
            return [0, 0], None, None, None

        # prepare DICTIONARY from dir and lookup tables
        i1, i2 = amico.lut.dir_TO_lut_idx( dirs[0] )
        A = np.zeros( (len(y), nATOMS), dtype=np.float64, order='F' )
        A[:,:(nD*n1)] = KERNELS['D'][:,i1,i2,:].T
        A[:,(nD*n1):] = KERNELS['CSF'].T

        # fit
        x = spams.lasso( np.asfortranarray( y.reshape(-1,1) ), D=A, **params ).todense().A1

        # return estimates
        v = x[ :n1 ].sum() / ( x.sum() + 1e-16 )

        # checking that there is more than 1 isotropic compartment
        if self.type == 'Mouse' :
            v_blood = x[ n1 ] / ( x.sum() + 1e-16 )
            v_csf = x[ n1+1 ] / ( x.sum() + 1e-16 )

            return [ v, 1-v, v_blood, v_csf ], dirs, x, A

        else :
            return [ v, 1-v ], dirs, x, A
