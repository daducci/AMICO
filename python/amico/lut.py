import numpy as np
import os.path
import cPickle
from dipy.data.fetcher import dipy_home
from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import real_sym_sh_basis
import amico.scheme


def precompute_rotation_matrices( lmax = 12 ) :
    """
    Precompute the rotation matrices to rotate the high-resolution kernels (500 directions per shell)

    Parameters
    ----------
    lmax : int
        Maximum SH order to use for the rotation phase (default : 12)
    """
    filename = os.path.join( dipy_home, 'AMICO_aux_matrices_lmax=%d.pickle'%lmax )
    if os.path.isfile( filename ) :
        return

    print '\n-> Precomputing rotation matrices for l_max=%d:' % lmax
    AUX = {}
    AUX['lmax'] = lmax

    # load file with 500 directions
    grad = np.loadtxt( os.path.join(os.path.dirname(amico.__file__), '500_dirs.txt') )
    for i in xrange(grad.shape[0]) :
        grad[i,:] /= np.linalg.norm( grad[i,:] )
        if grad[i,1] < 0 :
            grad[i,:] = -grad[i,:] # to ensure they are in the spherical range [0,180]x[0,180]

    # matrix to fit the SH coefficients
    _, theta, phi = cart2sphere( grad[:,0], grad[:,1], grad[:,2] )
    tmp, _, _ = real_sym_sh_basis( lmax, theta, phi )
    AUX['fit'] = np.dot( np.linalg.pinv( np.dot(tmp.T,tmp) ), tmp.T )

    # matrices to rotate the functions in SH space
    AUX['Ylm_rot'] = np.zeros( (181,181), dtype=np.object )
    for ox in xrange(181) :
        for oy in xrange(181) :
            tmp, _, _ = real_sym_sh_basis( lmax, ox/180.0*np.pi, oy/180.0*np.pi )
            AUX['Ylm_rot'][ox,oy] = tmp

    # auxiliary data to perform rotations
    AUX['const'] = np.zeros( AUX['fit'].shape[0], dtype=np.float64 )
    AUX['idx_m0'] = np.zeros( AUX['fit'].shape[0], dtype=np.int32 )
    i = 0
    for l in xrange(0,AUX['lmax']+1,2) :
        const  = np.sqrt(4.0*np.pi/(2.0*l+1.0))
        idx_m0 = (l*l + l + 2.0)/2.0 - 1
        for m in xrange(-l,l+1) :
            AUX['const'][i]  = const
            AUX['idx_m0'][i] = idx_m0
            i += 1

    with open( filename, 'wb+' ) as fid :
        cPickle.dump( AUX, fid, protocol=2 )

    print '   [ DONE ]'


def load_precomputed_rotation_matrices( lmax = 12 ) :
    """
    Load precomputed the rotation matrices to rotate the high-resolution kernels

    Parameters
    ----------
    lmax : int
        Maximum SH order to use for the rotation phase (default : 12)
    """
    filename = os.path.join( dipy_home, 'AMICO_aux_matrices_lmax=%d.pickle'%lmax )
    if not os.path.exists( filename ) :
        raise RuntimeError( 'Auxiliary matrices not found; call "lut.precompute_rotation_matrices()" first.' )
    return cPickle.load( open(filename,'rb') )


def aux_structures_generate( scheme, lmax = 12 ) :
    """
    Compute the auxiliary data structures to generate the high-resolution kernels

    Parameters
    ----------
    scheme : Scheme class
        Acquisition scheme of the acquired signal
    lmax : int
        Maximum SH order to use for the rotation phase (default : 12)

    Returns
    -------
    idx_IN : numpy array
        Indices of the samples belonging to each shell
    idx_OUT : numpy array
        Indices of the SH corresponding to each shell
    """
    nSH = (lmax+1)*(lmax+2)/2
    idx_IN  = []
    idx_OUT = []
    for s in xrange( len(scheme.shells) ) :
        idx_IN.append( range(500*s,500*(s+1)) )
        idx_OUT.append( range(nSH*s,nSH*(s+1)) )
    return ( idx_IN, idx_OUT )


def aux_structures_resample( scheme, lmax = 12 ) :
    """
    Compute the auxiliary data structures to resample the kernels to the original acquisition scheme

    Parameters
    ----------
    scheme : Scheme class
        Acquisition scheme of the acquired signal
    lmax : int
        Maximum SH order to use for the rotation phase (default : 12)

    Returns
    -------
    idx_OUT : numpy array
        Indices of the samples belonging to each shell
    Ylm_OUT : numpy array
        Operator to transform each shell from Spherical harmonics to original signal space
    """
    nSH = (lmax+1)*(lmax+2)/2
    idx_OUT = np.zeros( scheme.dwi_count, dtype=np.int32 )
    Ylm_OUT = np.zeros( (scheme.dwi_count,nSH*len(scheme.shells)), dtype=np.float32 ) # matrix from SH to real space
    idx = 0
    for s in xrange( len(scheme.shells) ) :
        nS = len( scheme.shells[s]['idx'] )
        idx_OUT[ idx:idx+nS ] = scheme.shells[s]['idx']
        _, theta, phi = cart2sphere( scheme.shells[s]['grad'][:,0], scheme.shells[s]['grad'][:,1], scheme.shells[s]['grad'][:,2] )
        tmp, _, _ = real_sym_sh_basis( lmax, theta, phi )
        Ylm_OUT[ idx:idx+nS, nSH*s:nSH*(s+1) ] = tmp
        idx += nS
    return ( idx_OUT, Ylm_OUT )


def rotate_kernel( K, AUX, idx_IN, idx_OUT, is_isotropic ) :
    """
    Rotate a spherical function.

    Parameters
    ----------
    K : numpy.ndarray
        Spherical function (in signal space) to rotate
    AUX : dictionary
        Auxiliary data structures needed to rotate functions in SH space
    idx_IN : list of list
        Index of samples in input kernel (K) belonging to each shell
    idx_OUT : list of list
        Index of samples in output kernel (K) belonging to each shell
    is_isotropic : boolean
        Indentifies whether K is an isotropic function or not

    Returns
    -------
    KRlm = numpy.array
        Spherical function (in SH space) rotated to 181x181 directions distributed
        on a hemisphere
    """
    # project kernel K to SH space
    Klm = []
    for s in xrange(len(idx_IN)) :
        Klm.append( np.dot( AUX['fit'], K[ idx_IN[s] ] ) )

    n = len(idx_IN)*AUX['fit'].shape[0]

    if is_isotropic == False :
        # fit SH and rotate kernel to 181*181 directions
        KRlm = np.zeros( (181,181,n), dtype=np.float32 )
        for ox in xrange(181) :
            for oy in xrange(181) :
                Ylm_rot = AUX['Ylm_rot'][ox,oy]
                for s in xrange(len(idx_IN)) :
                    KRlm[ox,oy,idx_OUT[s]] = AUX['const'] * Klm[s][AUX['idx_m0']] * Ylm_rot
    else :
        # simply fit SH
        KRlm = np.zeros( n, dtype=np.float32 )
        for s in xrange(len(idx_IN)) :
            KRlm[idx_OUT[s]] = Klm[s].astype(np.float32)

    return KRlm


def resample_kernel( KRlm, nS, idx_out, Ylm_out, is_isotropic ) :
    """
    Resample a spherical function

    Parameters
    ----------
    KRlm : numpy.array
        Rotated spherical functions (in SH space) to project
    nS : integer
        Number of samples in the subject's acquisition scheme
    idx_out : list of list
        Index of samples in output kernel
    Ylm_out : numpy.array
        Matrix to project back all shells from SH space to signal space (of the subject)
    is_isotropic : boolean
        Indentifies whether Klm is an isotropic function or not

    Returns
    -------
    KR = numpy.array
        Rotated spherical functions projected to signal space of the subject
    """
    if is_isotropic == False :
        KR = np.ones( (nS,181,181), dtype=np.float32 )
        for ox in xrange(181) :
            for oy in xrange(181) :
                KR[idx_out,ox,oy] = np.dot( Ylm_out, KRlm[ox,oy,:] ).astype(np.float32)
    else :
        KR = np.ones( nS, dtype=np.float32 )
        KR[idx_out] = np.dot( Ylm_out, KRlm ).astype(np.float32)
    return KR


def dir_TO_lut_idx( direction ) :
    """
    Compute the index in the kernel LUT corresponding to the estimated direction

    Parameters
    ----------
    direction : float array
        Orientation in 3D space

    Returns
    -------
    i1 : int
        First index (theta)
    i2 : int
        Second index (phi)
    """
    # ensure upper hemisphere (for LUT)
    if direction[1]<0 :
         direction = -direction

    i2 = np.arctan2(direction[1], direction[0]) % ( 2.0*np.pi )
    if i2 < 0 :
        i2 = i2 + 2.0*np.pi % ( 2.0*np.pi )

    if i2 > np.pi :
        i2 = np.arctan2( -direction[1], -direction[0] ) % ( 2.0*np.pi )
        i1 = np.arctan2( np.sqrt( direction[0]*direction[0] + direction[1]*direction[1] ), -direction[2] )
    else :
        i1 = np.arctan2( np.sqrt( direction[0]*direction[0] + direction[1]*direction[1] ), direction[2] );

    i1 = np.round( i1/np.pi*180.0 )
    i2 = np.round( i2/np.pi*180.0 )
    if i1<0 or i1>180 or i2<0 or i2>180 :
        raise Exception( '[amico.lut.dir_TO_lut_idx] index out of bounds (%d,%d)' % (i1,i2) )

    return i1, i2


def create_high_resolution_scheme( scheme, b_scale = 1 ) :
    """
    Create an high-resolution version of a scheme to be used for kernel rotation (500 directions per shell).
    All other parameters of the scheme remain the same.

    Parameters
    ----------
    scheme : Scheme class
        Original acquisition scheme
    b_scale : float
        If needed, apply a scaling to the b-values (default : 1)
    """
    # load HR directions and ensure they are in the spherical range [0,180]x[0,180]
    grad = np.loadtxt( os.path.join(os.path.dirname(amico.__file__), '500_dirs.txt') )
    for i in xrange(grad.shape[0]) :
        grad[i,:] /= np.linalg.norm( grad[i,:] )
        if grad[i,1] < 0 :
            grad[i,:] = -grad[i,:]

    n = len( scheme.shells )
    raw = np.zeros( (500*n, 4 if scheme.version==0 else 7) )
    row = 0
    for i in xrange(n) :
        raw[row:row+500,0:3] = grad
        if scheme.version == 0 :
            raw[row:row+500,3] = scheme.shells[i]['b'] * b_scale
        else :
            raw[row:row+500,3] = scheme.shells[i]['G']
            raw[row:row+500,4] = scheme.shells[i]['Delta']
            raw[row:row+500,5] = scheme.shells[i]['delta']
            raw[row:row+500,6] = scheme.shells[i]['TE']
        row += 500

    return amico.scheme.Scheme( raw )
