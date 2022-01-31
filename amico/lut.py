from __future__ import absolute_import, division, print_function

import numpy as np
from os import makedirs
import sys
from os.path import isdir, isfile, join as pjoin
import pickle
from dipy.data.fetcher import dipy_home
from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import real_sym_sh_basis
import amico.scheme
from amico.util import LOG, NOTE, WARNING, ERROR


def valid_dirs():
    """Return the list of the supported number of directions.

    Returns
    -------
    List of the supported number of directions
    """
    return np.arange(start=500, stop=10500, step=500).tolist() + [1, 32761]


def is_valid( ndirs ):
    """Check if the given ndirs value is supported by AMICO

    Parameters
    ----------
    ndirs : int
        Number of directions on the half of the sphere representing the possible orientations of the response functions

    Returns
    -------
    A bool value which indicates if the number of directions is supported
    """
    for value in valid_dirs():
        if ndirs == value:
            return True

    return False


def load_directions( ndirs ):
    """Load the directions on the half of the sphere

    Parameters
    ----------
    ndirs : int
        Number of directions on the half of the sphere representing the possible orientations of the response functions

    Returns
    -------
    directions : np.array(shape=(ndirs, 3))
        Array with the 3D directions in cartesian coordinates
    """
    amicopath = amico.__file__
    pos = len(amicopath) - 1
    while(amicopath[pos] != '/'):
        pos = pos - 1

    amicopath = amicopath[0 : pos] + '/directions/'

    filename = 'ndirs=%d.bin' % ndirs

    directions = np.fromfile(amicopath + filename, dtype=np.float64)
    directions = np.reshape(directions, (ndirs, 3))

    return directions


def load_precomputed_hash_table( ndirs ):
    """Load the pre-computed hash table to map high resolution directions into low resolution directions

    Parameters
    ----------
    ndirs : int
        Number of directions on the half of the sphere representing the possible orientations of the response functions

    Returns
    -------
    hash_table : np.array(shape=ndirs)
        Array with the indexes for every high resolution direction
    """
    amicopath = amico.__file__
    pos = len(amicopath) - 1
    while(amicopath[pos] != '/'):
        pos = pos - 1

    amicopath = amicopath[0 : pos] + '/directions/'

    filename = 'htable_ndirs=%d.bin' % ndirs

    hash_table = np.fromfile(amicopath + filename, dtype=np.int16)

    return hash_table


def precompute_rotation_matrices( lmax, ndirs ) :
    """Precompute the rotation matrices to rotate the high-resolution kernels (500 directions/shell).

    Parameters
    ----------
    lmax : int
        Maximum SH order to use for the rotation phase (default : 12)
    ndirs : int
        Number of directions on the half of the sphere representing the possible orientations of the response functions (default : 32761)
    """
    if not isdir(dipy_home) :
        makedirs(dipy_home)
    filename = pjoin( dipy_home, 'AMICO_aux_matrices_lmax=%d_ndirs=%d.pickle' % (lmax,ndirs) )
    if isfile( filename ) :
        return

    directions = load_directions(ndirs)

    AUX = {}
    AUX['lmax'] = lmax
    AUX['ndirs'] = ndirs

    # matrix to fit the SH coefficients
    _, theta, phi = cart2sphere( grad[:,0], grad[:,1], grad[:,2] )
    tmp, _, _ = real_sym_sh_basis( lmax, theta, phi )
    AUX['fit'] = np.dot( np.linalg.pinv( np.dot(tmp.T,tmp) ), tmp.T )

    # matrices to rotate the functions in SH space
    AUX['Ylm_rot'] = np.zeros( ndirs, dtype=np.object )
    for i in range(ndirs):
        _, theta, phi = cart2sphere(directions[i][0], directions[i][1], directions[i][2])
        tmp, _, _ = real_sym_sh_basis( lmax, theta, phi )
        AUX['Ylm_rot'][i] = tmp.reshape(-1)

    # auxiliary data to perform rotations
    AUX['const'] = np.zeros( AUX['fit'].shape[0], dtype=np.float64 )
    AUX['idx_m0'] = np.zeros( AUX['fit'].shape[0], dtype=np.int32 )
    i = 0
    for l in range(0,AUX['lmax']+1,2) :
        const  = np.sqrt(4.0*np.pi/(2.0*l+1.0))
        idx_m0 = (l*l + l + 2.0)/2.0 - 1
        for m in range(-l,l+1) :
            AUX['const'][i]  = const
            AUX['idx_m0'][i] = idx_m0
            i += 1

    with open( filename, 'wb+' ) as fid :
        pickle.dump( AUX, fid, protocol=2 )


def load_precomputed_rotation_matrices( lmax, ndirs ) :
    """Load precomputed the rotation matrices to rotate the high-resolution kernels.

    Parameters
    ----------
    lmax : int
        Maximum SH order to use for the rotation phase
    ndirs : int
        Number of directions on the half of the sphere representing the possible orientations of the response functions
    """
    filename = pjoin( dipy_home, 'AMICO_aux_matrices_lmax=%d_ndirs=%d.pickle' % (lmax,ndirs) )
    if not isfile( filename ) :
        str_lmax = ''
        str_ndirs = ''
        str_sep = ''
        if lmax != 12:
            str_lmax = 'lmax={}'.format(lmax)
        if ndirs != 32761:
            str_ndirs = 'ndirs={}'.format(ndirs)
        if str_lmax != '' and str_ndirs != '':
            str_sep = ' , '
        ERROR( 'Auxiliary matrices not found; call "amico.core.setup({}{}{})" first.'.format(str_lmax, str_sep, str_ndirs) )
    with open( filename, 'rb' ) as rotation_matrices_file:
        if sys.version_info.major == 3:
            aux = pickle.load( rotation_matrices_file, fix_imports=True, encoding='bytes' )
            # Pickle files written by Python 2 are loaded with byte
            # keys, whereas those written by Python 3 are loaded with
            # str keys, even when both are written using protocol=2 as
            # in precompute_rotation_matrices
            result_aux = {(k.decode() if hasattr(k,"decode") else k): v for k, v in aux.items()}
            return result_aux
        else:
            return pickle.load( rotation_matrices_file )


def aux_structures_generate( scheme, lmax = 12 ) :
    """Compute the auxiliary data structures to generate the high-resolution kernels.

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
    nSH = (lmax+1)*(lmax+2)//2
    idx_IN  = []
    idx_OUT = []
    for s in range( len(scheme.shells) ) :
        idx_IN.append( range(500*s,500*(s+1)) )
        idx_OUT.append( range(nSH*s,nSH*(s+1)) )
    return ( idx_IN, idx_OUT )


def aux_structures_resample( scheme, lmax = 12 ) :
    """Compute the auxiliary data structures to resample the kernels to the original acquisition scheme.

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
    nSH = (lmax+1)*(lmax+2)//2
    idx_OUT = np.zeros( scheme.dwi_count, dtype=np.int32 )
    Ylm_OUT = np.zeros( (scheme.dwi_count,nSH*len(scheme.shells)), dtype=np.float32 ) # matrix from SH to real space
    idx = 0
    for s in range( len(scheme.shells) ) :
        nS = len( scheme.shells[s]['idx'] )
        idx_OUT[ idx:idx+nS ] = scheme.shells[s]['idx']
        _, theta, phi = cart2sphere( scheme.shells[s]['grad'][:,0], scheme.shells[s]['grad'][:,1], scheme.shells[s]['grad'][:,2] )
        tmp, _, _ = real_sym_sh_basis( lmax, theta, phi )
        Ylm_OUT[ idx:idx+nS, nSH*s:nSH*(s+1) ] = tmp
        idx += nS
    return ( idx_OUT, Ylm_OUT )


def rotate_kernel( K, AUX, idx_IN, idx_OUT, is_isotropic, ndirs ) :
    """Rotate a response function (symmetric about z-axis).

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
    ndirs : int
        Number of directions on the half of the sphere representing the possible orientations of the response functions

    Returns
    -------
    KRlm = numpy.array
        Spherical function (in SH space) rotated to 181x181 directions distributed
        on a hemisphere
    """
    # project kernel K to SH space
    Klm = []
    for s in range(len(idx_IN)) :
        Klm.append( np.dot( AUX['fit'], K[ idx_IN[s] ] ) )

    n = len(idx_IN)*AUX['fit'].shape[0]

    if is_isotropic == False :
        # fit SH and rotate kernel to 181*181 directions
        KRlm = np.zeros( (ndirs,n), dtype=np.float32 )
        for i in range(ndirs) :
            Ylm_rot = AUX['Ylm_rot'][i]
            for s in range(len(idx_IN)) :
                KRlm[i,idx_OUT[s]] = AUX['const'] * Klm[s][AUX['idx_m0']] * Ylm_rot
    else :
        # simply fit SH
        KRlm = np.zeros( n, dtype=np.float32 )
        for s in range(len(idx_IN)) :
            KRlm[idx_OUT[s]] = Klm[s].astype(np.float32)

    return KRlm


def resample_kernel( KRlm, nS, idx_out, Ylm_out, is_isotropic, ndirs ) :
    """Project/resample a spherical function to signal space.

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
    ndirs : int
        Number of directions on the half of the sphere representing the possible orientations of the response functions

    Returns
    -------
    KR = numpy.array
        Rotated spherical functions projected to signal space of the subject
    """
    if is_isotropic == False :
        KR = np.ones( (ndirs,nS), dtype=np.float32 )
        try:
            for i in range(ndirs) :
                KR[i,idx_out] = np.dot( Ylm_out, KRlm[i,:] ).astype(np.float32)
        except:
            ERROR( 'Outdated LUT. Call "generate_kernels( regenerate=True )" to update the LUT' )
    else :
        KR = np.ones( nS, dtype=np.float32 )
        try:
            for i in range(ndirs) :
                KR[idx_out] = np.dot( Ylm_out, KRlm ).astype(np.float32)
        except:
            ERROR( 'Outdated LUT. Call "generate_kernels( regenerate=True )" to update the LUT' )
    return KR


def dir_TO_lut_idx( direction, htable ) :
    """Compute the index in the kernel LUT corresponding to the estimated direction.

    Parameters
    ----------
    direction : float array
        Orientation in 3D space
    htable : np.array(shape=32761)
        Hash table to map a high resolution 3D direction into a low resolution direction

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

    i1 = np.round( i1/np.pi*180.0 ).astype(int)
    i2 = np.round( i2/np.pi*180.0 ).astype(int)
    if i1<0 or i1>180 or i2<0 or i2>180 :
        raise RuntimeError( '"amico.lut.dir_TO_lut_idx" index out of bounds (%d,%d)' % (i1,i2) )

    return htable[i1*181 + i2]


def create_high_resolution_scheme( scheme, b_scale = 1 ) :
    """Create an high-resolution version of a scheme to be used for kernel rotation (500 directions per shell).
    All other parameters of the scheme remain the same.

    Parameters
    ----------
    scheme : Scheme class
        Original acquisition scheme
    b_scale : float
        If needed, apply a scaling to the b-values (default : 1)
    """
    n = len( scheme.shells )
    raw = np.zeros( (500*n, 4 if scheme.version==0 else 7) )
    row = 0
    for i in range(n) :
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


"""Gradient orientations for each shell of the high-resolution response functions.
NB: directions are built to ensure proper indexing in the [0,180]x[0,180] range of the lookup tables.
"""
grad = np.array([
    [-0.0490038,  0.9190721,  0.3910307 ],
    [ 0.7261449,  0.3010601, -0.6181233 ],
    [-0.6832152,  0.2550804, -0.6842156 ],
    [-0.8447361,  0.5018432,  0.1859419 ],
    [ 0.7303634,  0.6193082,  0.2881434 ],
    [-0.0509968,  0.0389975,  0.9979371 ],
    [-0.0179996,  0.8709800, -0.4909887 ],
    [-0.4441819,  0.4942024,  0.7473061 ],
    [ 0.9895082,  0.0860442,  0.1160596 ],
    [ 0.4698201,  0.8546727, -0.2209154 ],
    [ 0.4118960,  0.3998990,  0.8187933 ],
    [-0.5519743,  0.7899633, -0.2669876 ],
    [ 0.1229202,  0.4766905, -0.8704348 ],
    [-0.8483882,  0.1410646,  0.5102335 ],
    [ 0.3411076,  0.7882487,  0.5121616 ],
    [-0.3610025,  0.5290037, -0.7680054 ],
    [-0.4719906,  0.8499830,  0.2339953 ],
    [ 0.8560779,  0.4810438, -0.1890172 ],
    [ 0.7969295,  0.1619857,  0.5819485 ],
    [-0.4670874,  0.0090017,  0.8841654 ],
    [ 0.0130045,  0.9983450, -0.0560194 ],
    [-0.8824493,  0.3871971, -0.2671360 ],
    [-0.0170007,  0.5360212,  0.8440333 ],
    [ 0.4420323,  0.6510475, -0.6170450 ],
    [-0.3650675,  0.0580107, -0.9291719 ],
    [-0.9770420,  0.0040002,  0.2130092 ],
    [ 0.4061086,  0.9022414,  0.1450388 ],
    [-0.6271361,  0.6141333,  0.4791040 ],
    [-0.3539851,  0.7719676, -0.5279778 ],
    [ 0.6582818,  0.4722021,  0.5862510 ],
    [ 0.4229953,  0.3219965, -0.8469907 ],
    [-0.2119635,  0.7538703,  0.6218930 ],
    [-0.9115607,  0.1039499, -0.3978083 ],
    [-0.3110842,  0.9472563, -0.0770208 ],
    [ 0.6788843,  0.6318923, -0.3739362 ],
    [-0.1349580,  0.2859111, -0.9487050 ],
    [-0.6469877,  0.2299956,  0.7269862 ],
    [ 0.9040954,  0.3970419,  0.1580167 ],
    [-0.7572928,  0.6472502, -0.0870336 ],
    [ 0.1430137,  0.2840271,  0.9480905 ],
    [ 0.2330642,  0.8942464, -0.3821053 ],
    [-0.6638055,  0.5308445, -0.5268457 ],
    [ 0.1570514,  0.7102326,  0.6862248 ],
    [ 0.8947830,  0.2139481, -0.3919050 ],
    [ 0.5937516,  0.0799665,  0.8006650 ],
    [-0.0059986,  0.6878432, -0.7258345 ],
    [ 0.6652272,  0.7462549,  0.0240082 ],
    [-0.2770963,  0.2760960,  0.9203199 ],
    [-0.9622964,  0.2680826,  0.0460142 ],
    [ 0.1330404,  0.9702945,  0.2020613 ],
    [-0.7897450,  0.4048692,  0.4608512 ],
    [ 0.1939681,  0.1929683, -0.9618418 ],
    [-0.2359971,  0.9519881,  0.1949976 ],
    [ 0.8844065,  0.2721251,  0.3791743 ],
    [-0.4631902,  0.3071261, -0.8313413 ],
    [ 0.7000431,  0.0660041, -0.7110437 ],
    [-0.2000220,  0.9281021, -0.3140345 ],
    [ 0.5496906,  0.7046034,  0.4487474 ],
    [-0.6697194,  0.7266956,  0.1529359 ],
    [ 0.2370056,  0.7220170, -0.6500153 ],
    [ 0.9603842,  0.2601041, -0.1000400 ],
    [-0.4071366,  0.7562538,  0.5121719 ],
    [ 0.3551218,  0.1800618,  0.9173147 ],
    [-0.8087825,  0.3239129, -0.4908680 ],
    [ 0.2809716,  0.9549036, -0.0959903 ],
    [ 0.5920657,  0.4830536, -0.6450716 ],
    [ 0.3819515,  0.5929247,  0.7089100 ],
    [-0.2370657,  0.5061402,  0.8292297 ],
    [-0.7108312,  0.6248516, -0.3229233 ],
    [-0.1460262,  0.5190932, -0.8421512 ],
    [-0.9269194,  0.2429789,  0.2859751 ],
    [ 0.8093781,  0.5872743,  0.0040019 ],
    [-0.9760171,  0.1420025, -0.1650029 ],
    [ 0.1350134,  0.8700866,  0.4740472 ],
    [-0.2741398,  0.0480245,  0.9604900 ],
    [-0.5178141,  0.8546932, -0.0369867 ],
    [ 0.4821104,  0.7671757, -0.4230969 ],
    [-0.7501422,  0.0390074, -0.6601251 ],
    [-0.1429375,  0.0479790, -0.9885681 ],
    [-0.6320177,  0.4340122,  0.6420180 ],
    [ 0.7987899,  0.4288872, -0.4218891 ],
    [-0.4801229,  0.6321619, -0.6081557 ],
    [ 0.5469699,  0.7989561,  0.2499863 ],
    [ 0.6160826,  0.3100415,  0.7240970 ],
    [ 0.0280122,  0.9624197, -0.2701178 ],
    [ 0.3240951,  0.5061486, -0.7992346 ],
    [-0.2680190,  0.8750621,  0.4030286 ],
    [ 0.8433011,  0.0590211, -0.5341907 ],
    [-0.8858335,  0.4619132, -0.0439917 ],
    [ 0.1870006,  0.4890015,  0.8520026 ],
    [-0.1840026,  0.7460104, -0.6400090 ],
    [ 0.7925670,  0.4497543,  0.4117750 ],
    [-0.4751271,  0.2480664,  0.8442259 ],
    [ 0.6616729,  0.7276403, -0.1809106 ],
    [ 0.9427799,  0.0829806,  0.3229246 ],
    [-0.0450160,  0.7202561,  0.6922461 ],
    [ 0.5549376,  0.1899786, -0.8099089 ],
    [-0.3859249,  0.8678312, -0.3129391 ],
    [-0.5468857,  0.1249739, -0.8278270 ],
    [-0.7190194,  0.6210168,  0.3120084 ],
    [-0.0600047,  0.9910783,  0.1190094 ],
    [-0.0609758,  0.2848868,  0.9566198 ],
    [-0.9984274,  0.0320137,  0.0460197 ],
    [ 0.0459845,  0.3158932, -0.9476797 ],
    [ 0.3148837,  0.8946695,  0.3168829 ],
    [-0.5690950,  0.4360728, -0.6971164 ],
    [ 0.4889010,  0.8718235, -0.0299939 ],
    [-0.4499782,  0.6289695,  0.6339693 ],
    [ 0.1638897,  0.8264440, -0.5386376 ],
    [ 0.9150645,  0.4030284, -0.0150011 ],
    [-0.9251355,  0.0210031,  0.3790555 ],
    [ 0.5272347,  0.5912632,  0.6102716 ],
    [-0.2959200,  0.3179140, -0.9007564 ],
    [-0.6480149,  0.7590175, -0.0630014 ],
    [ 0.3610847,  0.1570368, -0.9192156 ],
    [-0.8318195,  0.1329711, -0.5388831 ],
    [-0.1509909,  0.9819406, -0.1139931 ],
    [-0.7558844,  0.2689589,  0.5969087 ],
    [ 0.6199132,  0.5859180, -0.5219269 ],
    [ 0.1749967,  0.1219977,  0.9769814 ],
    [-0.5581792,  0.7052264, -0.4371403 ],
    [-0.5377449,  0.7586401,  0.3678255 ],
    [ 0.7599263,  0.3249685,  0.5629454 ],
    [-0.0399955,  0.8349057,  0.5489380 ],
    [ 0.9100091,  0.3180032, -0.2660027 ],
    [ 0.1520170,  0.6170688, -0.7720861 ],
    [-0.9413827,  0.3091257, -0.1350549 ],
    [ 0.2149015,  0.9755528,  0.0459789 ],
    [-0.6079678,  0.0689963,  0.7909581 ],
    [ 0.4999220,  0.2339635,  0.8338699 ],
    [ 0.8138015,  0.5538649,  0.1759571 ],
    [-0.1929689,  0.8518629, -0.4869216 ],
    [-0.8637772,  0.4018963,  0.3039216 ],
    [-0.1230604,  0.4172049,  0.9004422 ],
    [-0.7898326,  0.4798983, -0.3819190 ],
    [ 0.5720023,  0.3580014, -0.7380030 ],
    [-0.3870979,  0.9172321,  0.0940238 ],
    [ 0.3150197,  0.9130571, -0.2590162 ],
    [ 0.9541617,  0.2320393,  0.1890320 ],
    [ 0.0160026,  0.1390225, -0.9901599 ],
    [ 0.3058977,  0.7077633,  0.6367870 ],
    [-0.2338876,  0.6286980, -0.7416437 ],
    [-0.2699103,  0.6327896,  0.7257587 ],
    [ 0.7637476,  0.6057998, -0.2229263 ],
    [-0.7915877,  0.6066840,  0.0729620 ],
    [ 0.8709351,  0.0579957,  0.4879636 ],
    [ 0.3459763,  0.7849462, -0.5139648 ],
    [ 0.4440566,  0.8151039,  0.3720474 ],
    [ 0.8128622,  0.2109642, -0.5429080 ],
    [ 0.2740036,  0.3570046,  0.8930116 ],
    [ 0.2530431,  0.3570609, -0.8991533 ],
    [-0.7030109,  0.3880060, -0.5960092 ],
    [-0.3921188,  0.3811155,  0.8372537 ],
    [-0.0340003,  0.9650092,  0.2600025 ],
    [-0.9727311,  0.1679536,  0.1599558 ],
    [ 0.6731205,  0.5921060,  0.4430793 ],
    [-0.2779411,  0.1679644, -0.9457995 ],
    [-0.4300951,  0.8871961, -0.1670369 ],
    [-0.7568679,  0.0859850,  0.6478870 ],
    [ 0.6677500,  0.7247286,  0.1699364 ],
    [ 0.6916453,  0.1289339,  0.7106355 ],
    [ 0.1439576,  0.9727136, -0.1819464 ],
    [-0.8152589,  0.5441728, -0.1980629 ],
    [-0.1590055,  0.1550053,  0.9750336 ],
    [ 0.9890366,  0.1260047, -0.0770028 ],
    [ 0.0980152,  0.6140955,  0.7831218 ],
    [ 0.0099964,  0.7847156, -0.6197754 ],
    [-0.6973355,  0.4982397,  0.5152479 ],
    [ 0.4518146,  0.5257842, -0.7207042 ],
    [-0.3928033,  0.8475755,  0.3568213 ],
    [-0.9144715,  0.2381228, -0.3271687 ],
    [ 0.5768474,  0.7597991, -0.2999207 ],
    [ 0.4720640,  0.0280038,  0.8811194 ],
    [-0.0429804,  0.4507940, -0.8915926 ],
    [ 0.1649588,  0.9247693,  0.3429145 ],
    [ 0.5229336,  0.4579418,  0.7189087 ],
    [-0.3560931,  0.6711755, -0.6501700 ],
    [ 0.7227810,  0.4388670, -0.5338383 ],
    [-0.5629856,  0.8189791,  0.1109972 ],
    [-0.3939974,  0.1359991,  0.9089941 ],
    [ 0.9537592,  0.1419642, -0.2649331 ],
    [-0.5799838,  0.2659926, -0.7699784 ],
    [-0.9161846,  0.3790764,  0.1300262 ],
    [-0.2019147,  0.9785866,  0.0399831 ],
    [ 0.8665763,  0.4048021,  0.2918573 ],
    [-0.5601174,  0.3700775,  0.7411553 ],
    [ 0.5401805,  0.8362794,  0.0940314 ],
    [-0.0620315,  0.9284718, -0.3661861 ],
    [ 0.0609908,  0.4059389,  0.9118628 ],
    [-0.1940654,  0.8372822,  0.5111723 ],
    [-0.9705512,  0.0219898, -0.2398891 ],
    [-0.2890548,  0.4400834, -0.8501611 ],
    [ 0.6650033,  0.2020010, -0.7190036 ],
    [-0.6619027,  0.7168946, -0.2189678 ],
    [-0.8655336,  0.2588605,  0.4287690 ],
    [ 0.1560106,  0.7980543,  0.5820396 ],
    [ 0.7458162,  0.6608371, -0.0839793 ],
    [ 0.1670410,  0.0580142, -0.9842417 ],
    [-0.4441946,  0.7923471, -0.4181832 ],
    [-0.6591447,  0.0760167, -0.7481642 ],
    [ 0.3101779,  0.6283605, -0.7134093 ],
    [-0.5229587,  0.6699471,  0.5269584 ],
    [ 0.3578667,  0.9336523,  0.0149944 ],
    [ 0.9636392,  0.2619019,  0.0529802 ],
    [ 0.3009442,  0.5329012,  0.7908533 ],
    [-0.1320390,  0.6251845,  0.7692270 ],
    [-0.0069993,  0.5809402, -0.8139162 ],
    [ 0.7909601,  0.5139740, -0.3319832 ],
    [-0.5670536,  0.5410511, -0.6210587 ],
    [ 0.0419944,  0.1559790,  0.9868673 ],
    [-0.7796251,  0.5187505,  0.3508313 ],
    [ 0.3748565,  0.8496746, -0.3708580 ],
    [ 0.8494480,  0.2198571,  0.4796883 ],
    [ 0.4550394,  0.7010606,  0.5490475 ],
    [ 0.5631281,  0.0600137, -0.8241875 ],
    [-0.9878336,  0.1519744, -0.0329944 ],
    [ 0.0760361,  0.9954730,  0.0570271 ],
    [ 0.2911156,  0.0500199,  0.9553794 ],
    [-0.2721304,  0.9374491, -0.2171040 ],
    [-0.6256907,  0.7326378,  0.2678676 ],
    [-0.7756820,  0.2299058, -0.5877591 ],
    [ 0.5340360,  0.6780458, -0.5050341 ],
    [ 0.6269267,  0.7029178,  0.3359607 ],
    [ 0.9000270,  0.0920028, -0.4260128 ],
    [ 0.0959645,  0.9106627, -0.4018511 ],
    [-0.1809990,  0.9309949,  0.3169983 ],
    [-0.4502423,  0.4302315, -0.7824211 ],
    [-0.5460052,  0.5310050,  0.6480062 ],
    [ 0.7050130,  0.2670049,  0.6570122 ],
    [ 0.3269940,  0.2619952, -0.9079832 ],
    [-0.6609408,  0.6129451, -0.4329613 ],
    [ 0.5788122,  0.8077379, -0.1119637 ],
    [-0.8309555,  0.5549703, -0.0389979 ],
    [-0.2639596,  0.3929399,  0.8808652 ],
    [ 0.8692212,  0.4911250,  0.0570145 ],
    [-0.1080168,  0.1720268, -0.9791523 ],
    [-0.9261829,  0.1350267,  0.3520695 ],
    [ 0.2740327,  0.9411125,  0.1980237 ],
    [-0.8571174,  0.3410467, -0.3860529 ],
    [ 0.8294436,  0.3187862, -0.4586923 ],
    [-0.3329531,  0.7079002,  0.6229122 ],
    [-0.2938947,  0.8586922, -0.4198495 ],
    [ 0.3978826,  0.2949130,  0.8687438 ],
    [ 0.4539036,  0.4219104, -0.7848332 ],
    [-0.4168004,  0.1869105, -0.8895740 ],
    [-0.5578224,  0.1729449,  0.8117415 ],
    [ 0.7249090,  0.4749404,  0.4989374 ],
    [ 0.1139902,  0.7299372, -0.6739420 ],
    [-0.3461002,  0.9122641,  0.2190634 ],
    [ 0.2668595,  0.8575486,  0.4397685 ],
    [ 0.9138511,  0.3769386, -0.1509754 ],
    [-0.6960177,  0.7170183,  0.0380010 ],
    [ 0.9343645,  0.1920749,  0.3001171 ],
    [-0.0509894,  0.9817968, -0.1829621 ],
    [ 0.7772495,  0.0500161, -0.6272014 ],
    [ 0.0379876,  0.7717481,  0.6347928 ],
    [ 0.3478545,  0.0419824, -0.9366081 ],
    [-0.1279618,  0.6628022, -0.7377798 ],
    [ 0.8181010,  0.0020002,  0.5750710 ],
    [-0.9307813,  0.3649143, -0.0219948 ],
    [-0.7211421,  0.3740737,  0.5831149 ],
    [ 0.3930690,  0.9081594, -0.1440253 ],
    [ 0.2329503,  0.6228670,  0.7468406 ],
    [ 0.1410863,  0.3642228, -0.9205631 ],
    [ 0.7493470,  0.6533025,  0.1080500 ],
    [-0.5479636,  0.8229453, -0.1499900 ],
    [-0.9955795,  0.0269886, -0.0899620 ],
    [ 0.2480069,  0.2260063,  0.9420264 ],
    [ 0.7069526,  0.5389639, -0.4579693 ],
    [-0.3490925,  0.5451445,  0.7622020 ],
    [-0.1659524,  0.3968861, -0.9027410 ],
    [-0.7674771,  0.6083782,  0.2021257 ],
    [-0.1120145,  0.8111050, -0.5740743 ],
    [ 0.6159181,  0.4129451,  0.6709108 ],
    [-0.7498302,  0.4468988, -0.4878895 ],
    [ 0.4340267,  0.8640531,  0.2550157 ],
    [ 0.3498003,  0.7205887, -0.5986583 ],
    [-0.1700007,  0.2760011,  0.9460038 ],
    [-0.4079686,  0.9129297, -0.0109992 ],
    [ 0.7993570,  0.5152301,  0.3091381 ],
    [ 0.6530007,  0.3750004, -0.6580007 ],
    [-0.4731886,  0.7072819, -0.5252093 ],
    [ 0.0519974,  0.9299535,  0.3639818 ],
    [-0.9476849,  0.1439521, -0.2849053 ],
    [-0.7231233,  0.1790305,  0.6671138 ],
    [ 0.5960847,  0.1900270,  0.7801108 ],
    [ 0.0520261,  0.0170085,  0.9985009 ],
    [ 0.1790278,  0.9431462, -0.2800434 ],
    [-0.9357394,  0.2939182,  0.1949457 ],
    [-0.3129290,  0.8088164,  0.4978870 ],
    [ 0.9605880,  0.2039125, -0.1889190 ],
    [ 0.2190147,  0.5240351, -0.8230551 ],
    [-0.4581961,  0.5422321, -0.7043015 ],
    [-0.0139953,  0.6357851,  0.7717392 ],
    [-0.0919775,  0.9957566,  0.0029993 ],
    [ 0.8317983,  0.5448679, -0.1059743 ],
    [ 0.4701389,  0.1470435, -0.8702572 ],
    [-0.8767681,  0.4498810, -0.1699551 ],
    [ 0.5988521,  0.5978523,  0.5328684 ],
    [-0.6307292,  0.6727111,  0.3868339 ],
    [-0.5648659,  0.0249941, -0.8248041 ],
    [ 0.5911434,  0.7031705, -0.3950958 ],
    [-0.8710928,  0.0450048,  0.4890521 ],
    [ 0.1509748,  0.9868352, -0.0579903 ],
    [ 0.9138935,  0.0009999,  0.4059527 ],
    [-0.2878596,  0.1639200,  0.9435397 ],
    [ 0.4261095,  0.5011288,  0.7531936 ],
    [-0.6171898,  0.7142197, -0.3301015 ],
    [ 0.0899816,  0.2179555, -0.9718018 ],
    [ 0.9883099,  0.1500470,  0.0270085 ],
    [ 0.2561711,  0.8405613, -0.4773187 ],
    [-0.7198776,  0.1579731, -0.6758851 ],
    [-0.1109995,  0.8819960,  0.4579979 ],
    [ 0.8612955,  0.3921345, -0.3231108 ],
    [ 0.0379997,  0.3129978,  0.9489934 ],
    [-0.2698903,  0.7626900, -0.5877611 ],
    [-0.8103546,  0.3051335,  0.5002189 ],
    [ 0.4369133,  0.7718468,  0.4619083 ],
    [ 0.5258869,  0.5748764, -0.6268652 ],
    [-0.4811160,  0.8692095,  0.1140275 ],
    [ 0.8241640,  0.3450687,  0.4490894 ],
    [-0.4899260,  0.3489473,  0.7988794 ],
    [-0.2500149,  0.0650039, -0.9660575 ],
    [ 0.5890630,  0.8080865, -0.0010001 ],
    [-0.7470557,  0.6310470, -0.2090156 ],
    [ 0.7498482,  0.1719652, -0.6388706 ],
    [-0.6261588,  0.3510890, -0.6961765 ],
    [-0.1280564,  0.5202290,  0.8443716 ],
    [-0.1330177,  0.9711292,  0.1980263 ],
    [-0.9646730,  0.1019654,  0.2429177 ],
    [-0.2479255,  0.5408375, -0.8037585 ],
    [-0.1270756,  0.9595706, -0.2511494 ],
    [ 0.4589234,  0.1369771,  0.8778534 ],
    [-0.8696435,  0.4878000,  0.0759689 ],
    [ 0.9173877,  0.3211357,  0.2350994 ],
    [-0.6030504,  0.5600468,  0.5680474 ],
    [ 0.4780636,  0.8191089, -0.3170422 ],
    [-0.8665941,  0.2119007, -0.4517884 ],
    [ 0.3399670,  0.4129599, -0.8449180 ],
    [ 0.2529067,  0.7797123,  0.5727887 ],
    [-0.4581143,  0.7811949,  0.4241058 ],
    [-0.6791270,  0.0360067,  0.7331371 ],
    [ 0.0669749,  0.8416848, -0.5357994 ],
    [ 0.0899486,  0.4997142,  0.8615074 ],
    [ 0.6889855,  0.6749858, -0.2639945 ],
    [-0.3809964,  0.3469967, -0.8569919 ],
    [ 0.3010129,  0.9480408,  0.1030044 ],
    [ 0.7539318,  0.0809927,  0.6519410 ],
    [-0.4648109,  0.8416575, -0.2748881 ],
    [-0.1079442,  0.7765986,  0.6206792 ],
    [ 0.5271252,  0.2860679, -0.8001901 ],
    [ 0.9909178,  0.0379968, -0.1289893 ],
    [-0.1629429,  0.0439846,  0.9856546 ],
    [ 0.6252395,  0.7402836,  0.2470947 ],
    [-0.9382623,  0.2660744, -0.2210618 ],
    [-0.0470086,  0.3510641, -0.9351707 ],
    [ 0.5158842,  0.3479219,  0.7828243 ],
    [-0.2971149,  0.9533685,  0.0530205 ],
    [-0.7210285,  0.5480216,  0.4240167 ],
    [-0.5820029,  0.6210031, -0.5250026 ],
    [ 0.4209162,  0.6488709,  0.6338739 ],
    [-0.4929657,  0.1109923,  0.8629400 ],
    [ 0.9308762,  0.3589523,  0.0679910 ],
    [ 0.0879780,  0.5658586, -0.8197951 ],
    [ 0.0459870,  0.9847219,  0.1679526 ],
    [-0.8735642,  0.3288360,  0.3588210 ],
    [-0.1439260,  0.8975383, -0.4167856 ],
    [-0.8000796,  0.5200517, -0.2990298 ],
    [ 0.7699119,  0.3479602, -0.5349388 ],
    [-0.3719239,  0.6188735,  0.6918585 ],
    [-0.4618206,  0.0969623, -0.8816575 ],
    [ 0.7351066,  0.5620815,  0.3790550 ],
    [-0.5671563,  0.7942188,  0.2180601 ],
    [ 0.9709558,  0.1089950,  0.2129903 ],
    [ 0.1839461,  0.3888861,  0.9027355 ],
    [ 0.4451614,  0.7252630, -0.5251904 ],
    [-0.6560912,  0.3250452,  0.6810947 ],
    [ 0.6702786,  0.7383068, -0.0750312 ],
    [ 0.0380005,  0.8720122,  0.4880068 ],
    [ 0.2619565,  0.1179804, -0.9578410 ],
    [ 0.9451654,  0.0650114, -0.3200560 ],
    [ 0.2539415,  0.9487813, -0.1879567 ],
    [-0.0779569,  0.7385920, -0.6696301 ],
    [ 0.8960251,  0.1380039,  0.4220118 ],
    [-0.6039559,  0.7969418,  0.0109992 ],
    [-0.2270111,  0.2470121, -0.9420462 ],
    [-0.3718554,  0.2748931,  0.8866552 ],
    [-0.9874546,  0.1439205,  0.0649641 ],
    [ 0.5341707,  0.7662449,  0.3571141 ],
    [-0.6421500,  0.4601074, -0.6131432 ],
    [ 0.3901239,  0.5931884, -0.7042236 ],
    [-0.1718924,  0.6915672,  0.7015610 ],
    [-0.3468802,  0.9206820, -0.1789382 ],
    [-0.0540163,  0.1680508,  0.9842973 ],
    [ 0.8791649,  0.4700881, -0.0780146 ],
    [-0.6260776,  0.1740216, -0.7600943 ],
    [ 0.4510032,  0.8910062,  0.0520004 ],
    [-0.2989622,  0.9038857,  0.3059613 ],
    [ 0.6959788,  0.3739886,  0.6129813 ],
    [ 0.6299553,  0.1089923, -0.7689454 ],
    [-0.8249662,  0.4959797,  0.2709889 ],
    [-0.3421591,  0.6092834, -0.7153327 ],
    [ 0.3089836,  0.4479763,  0.8389555 ],
    [ 0.6552028,  0.5001548, -0.5661753 ],
    [ 0.0469978,  0.9869536, -0.1539928 ],
    [-0.9678974,  0.2419744, -0.0679928 ],
    [ 0.2249076,  0.9336164,  0.2788854 ],
    [-0.8161273,  0.0660103,  0.5740896 ],
    [ 0.3809724,  0.0699949,  0.9219332 ],
    [ 0.1950409,  0.6701404, -0.7161500 ],
    [ 0.8532193,  0.4791232,  0.2060530 ],
    [-0.6957743,  0.7067707, -0.1279585 ],
    [ 0.8836868,  0.2968948, -0.3618717 ],
    [-0.3350486,  0.4620670,  0.8211191 ],
    [-0.8671869,  0.0620134, -0.4941065 ],
    [ 0.0499912,  0.0619891, -0.9968241 ],
    [ 0.5550872,  0.8061266, -0.2050322 ],
    [-0.3650940,  0.8182107, -0.4441144 ],
    [ 0.0689832,  0.6938314,  0.7168258 ],
    [-0.7348450,  0.6658595,  0.1289728 ],
    [ 0.6677166,  0.0329860,  0.7436844 ],
    [-0.1000311,  0.5871826, -0.8032498 ],
    [ 0.5618618,  0.5218716,  0.6418421 ],
    [-0.7272568,  0.5411911, -0.4221490 ],
    [-0.5330741,  0.4630644,  0.7080984 ],
    [ 0.1640471,  0.8772518, -0.4511295 ],
    [ 0.7425535,  0.5846484, -0.3268035 ],
    [ 0.3519096,  0.8467824,  0.3988975 ],
    [ 0.2071001,  0.2831368, -0.9364524 ],
    [-0.2210369,  0.9731625, -0.0640107 ],
    [ 0.8604860,  0.1569063, -0.4847105 ],
    [-0.0319966,  0.4179557,  0.9079038 ],
    [-0.5217486,  0.3638247, -0.7716282 ],
    [ 0.7408896,  0.6439041,  0.1909715 ],
    [-0.5399609,  0.7119484,  0.4489675 ],
    [ 0.9987498,  0.0479880,  0.0139965 ],
    [ 0.5298072,  0.4518356, -0.7177388 ],
    [ 0.1859441,  0.0409877,  0.9817050 ],
    [-0.8943551,  0.3911553,  0.2170862 ],
    [-0.7569190,  0.3179660, -0.5709389 ],
    [ 0.0669857,  0.9567957,  0.2829396 ],
    [-0.5230905,  0.7701332, -0.3650632 ],
    [-0.8017699,  0.2009423,  0.5628385 ],
    [ 0.2399899,  0.7799672, -0.5779757 ],
    [ 0.7938166,  0.2489425,  0.5548718 ],
    [ 0.5989727,  0.7889641,  0.1369938 ],
    [-0.3858989,  0.0379900,  0.9217585 ],
    [ 0.2979677,  0.6469298,  0.7019238 ],
    [ 0.0420157,  0.4331618, -0.9003363 ],
    [-0.2850817,  0.7712210,  0.5691631 ],
    [-0.2880402,  0.9011257, -0.3240452 ],
    [ 0.9480337,  0.3160112, -0.0370013 ],
    [-0.9110141,  0.3920061, -0.1280020 ],
    [ 0.3898532,  0.8766700, -0.2818939 ],
    [ 0.6369296,  0.2859684, -0.7159209 ],
    [-0.3510938,  0.2430649, -0.9042415 ],
    [-0.4798702,  0.8207780,  0.3099162 ],
    [-0.5690424,  0.2690200,  0.7770579 ],
    [-0.9447232,  0.0439871, -0.3249048 ],
    [ 0.3208138,  0.2778388,  0.9054745 ],
    [ 0.0250035,  0.9341322, -0.3560504 ],
    [ 0.7468674,  0.6648820,  0.0109980 ],
    [ 0.1619919,  0.9799510,  0.1159942 ],
    [-0.8917945,  0.1659618,  0.4209030 ],
    [-0.2630175,  0.6970464, -0.6670444 ],
    [ 0.4331852,  0.2341001, -0.8703722 ],
    [-0.8411455,  0.4180723, -0.3430594 ],
    [ 0.6171932,  0.6542048,  0.4371368 ],
    [-0.1750326,  0.8991677,  0.4010748 ],
    [-0.0320059,  0.2320427, -0.9721789 ],
    [ 0.8447762,  0.4608779, -0.2719279 ],
    [ 0.6627525,  0.2169190,  0.7167324 ],
    [-0.7645189,  0.6445944, -0.0029981 ],
    [-0.2140354,  0.5770955,  0.7881304 ],
    [ 0.1139699,  0.2019466,  0.9727427 ],
    [ 0.8569272,  0.3639691,  0.3649690 ],
    [ 0.3899016,  0.9187680, -0.0619844 ],
    [-0.4551327,  0.8862584, -0.0860251 ],
    [-0.8126612,  0.4358183,  0.3868387 ],
    [ 0.0740275,  0.8223060,  0.5642099 ],
    [ 0.2330348,  0.4460667, -0.8641292 ],
    [ 0.9303960,  0.2200937, -0.2931248 ],
    [-0.5051460,  0.2210639, -0.8342411 ],
    [ 0.1810461,  0.5731459,  0.7992034 ],
    [ 0.6170836,  0.6440873, -0.4520613 ],
    [-0.4042586,  0.7044506, -0.5833732 ],
    [-0.4401814,  0.6902845,  0.5742366 ],
    [-0.7812387,  0.1220373, -0.6121871 ],
    [-0.1720806,  0.9784580,  0.1140534 ],
    [-0.6662196,  0.1360448,  0.7332416 ],
    [-0.9874370,  0.0750332,  0.1390615 ],
    [ 0.0699656,  0.6626737, -0.7456329 ],
    [-0.1878933,  0.3557980,  0.9154802 ],
    [ 0.4901243,  0.8522161,  0.1830464 ],
    [-0.6951964,  0.6771913,  0.2410681 ],
    [ 0.9750673,  0.1860128,  0.1210083 ],
    [ 0.3931658,  0.8083408, -0.4381847 ],
    [ 0.4719920,  0.5549906,  0.6849884 ],
    [-0.3651733,  0.4492132, -0.8153870 ],
    [-0.6117541,  0.7716898, -0.1739301 ],
])
