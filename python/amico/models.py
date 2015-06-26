import numpy as np
from os.path import exists, join as pjoin
from os import remove
import subprocess
import tempfile
import amico.lut
from amico.progressbar import ProgressBar
import spams
import warnings
warnings.filterwarnings("ignore") # needed for a problem with spams


class CylinderZeppelinBall :
    """
    Simulate the response functions according to the Cylinder-Zeppelin-Ball model.

    The contributions of the tracts are modeled as "cylinders" with specific radii (Rs)
    and a given axial diffusivity (d_par).
    Extra-cellular contributions are modeled as tensors with the same axial diffusivity
    as the sticks (d_par) and whose perpendicular diffusivities are calculated with a tortuosity
    model as a function of the intra-cellular volume fractions (ICVFs).
    Isotropic contributions are modeled as tensors with isotropic diffusivities (d_ISOs).

    NB: this models works only with schemes containing the full specification of
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
        self.id     = 'CylinderZeppelinBall'
        self.name   = 'Cylinder-Zeppelin-Ball'
        self.d_par  = 0.6E-3                                                         # Parallel diffusivity [mm^2/s]
        self.Rs     = np.concatenate( ([0.01],np.linspace(0.5,8.0,20.0)) ) * 1E-6    # Radii of the axons [meters]
        self.ICVFs  = np.arange(0.3,0.9,0.1)                                         # Intra-cellular volume fraction(s) [0..1]
        self.d_ISOs = [ 2.0E-3 ]                                                     # Isotropic diffusivitie(s) [mm^2/s]

        # generated scalar maps
        self.OUTPUT_names        = [ 'v', 'a', 'd' ]
        self.OUTPUT_descriptions = [ 'Intra-cellular volume fraction', 'Mean axonal diameter', 'Axonal density' ]


    def set( self, d_par, Rs, ICVFs, d_ISOs ) :
        self.d_par  = d_par
        self.Rs     = Rs
        self.ICVFs  = ICVFs
        self.d_ISOs = d_ISOs


    def set_solver( self, mode = 2, pos = True, lambda1 = 0.0, lambda2 = 4.0 ) :
        params = {}
        params = {}
        params['mode']    = mode
        params['pos']     = pos
        params['lambda1'] = lambda1
        params['lambda2'] = lambda2
        return params


    def generate( self, out_path, scheme, aux, idx_in, idx_out ) :
        if scheme.version != 1 :
            raise RuntimeError( 'This model requires a "VERSION: STEJSKALTANNER" scheme.' )

        # create a high-resolution scheme to pass to 'datasynth'
        scheme_high = amico.lut.create_high_resolution_scheme( scheme, b_scale=1E6 )
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
        KERNELS['IC']    = np.zeros( (self.nS,len(self.Rs),181,181), dtype=np.float32 )
        KERNELS['EC']    = np.zeros( (self.nS,len(self.ICVFs),181,181), dtype=np.float32 )
        KERNELS['ISO']   = np.zeros( (self.nS,len(self.d_ISOs)), dtype=np.float32 )

        nATOMS = len(self.Rs) + len(self.ICVFs) + len(self.d_ISOs)
        progress = ProgressBar( n=nATOMS, prefix="   ", erase=True )

        # Cylinder(s)
        for i in xrange(len(self.Rs)) :
            lm = np.load( pjoin( in_path, 'A_%03d.npy'%progress.i ) )
            KERNELS['IC'][:,i,:,:] = amico.lut.resample_kernel( lm, self.nS, idx_out, Ylm_out, False )
            progress.update()

        # Zeppelin(s)
        for i in xrange(len(self.ICVFs)) :
            lm = np.load( pjoin( in_path, 'A_%03d.npy'%progress.i ) )
            KERNELS['EC'][:,i,:,:] = amico.lut.resample_kernel( lm, self.nS, idx_out, Ylm_out, False )
            progress.update()

        # Ball(s)
        for i in xrange(len(self.d_ISOs)) :
            lm = np.load( pjoin( in_path, 'A_%03d.npy'%progress.i ) )
            KERNELS['ISO'][:,i] = amico.lut.resample_kernel( lm, self.nS, idx_out, Ylm_out, True )
            progress.update()

        return KERNELS


    def fit( self, y, i1, i2, KERNELS, idx, params ) :
        n1 = len(self.Rs)
        n2 = len(self.ICVFs)
        n3 = len(self.d_ISOs)

        # prepare DICTIONARY from LUT
        A = np.zeros( (len(y), n1+n2+n3 ), dtype=np.float64, order='F' )
        A[:,0:n1]       = KERNELS['IC'][:,:,i1,i2]
        A[:,n1:(n1+n2)] = KERNELS['EC'][:,:,i1,i2]
        A[:,(n1+n2):]   = KERNELS['ISO'][:,:]

        # fit
        if idx is None :
            Ar = A
            yr = np.asfortranarray( y.reshape(-1,1) )
        else :
            Ar = np.zeros( (len(idx),n1+n2+n3), dtype=np.float64, order='F' )
            Ar[:,:] = A[idx,:]
            yr = np.asfortranarray( y[idx].reshape(-1,1) )
        x = spams.lasso( yr, D=Ar, **params ).todense().A1

        # estimated signal
        y_est = np.dot( A, x )

        # return estimates
        f1 = x[ :n1 ].sum()
        f2 = x[ n1:(n1+n2) ].sum()
        v = f1 / ( f1 + f2 + 1e-16 )
        a = 1E6 * 2.0 * np.dot(self.Rs,x[:n1]) / ( f1 + 1e-16 )
        d = (4.0*v) / ( np.pi*a**2 + 1e-16 )
        return [ y_est, [v, a, d] ]
