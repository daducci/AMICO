import numpy as np
from re import match as re_match
import os.path

class Scheme :
    """A class to hold information about an acquisition scheme.

    The scheme can be specified in two formats:
    - as a Nx4 matrix: the first three columns are the gradient directions and
      the fourth is the b-value (s/mm^2).
    - as a Nx7 matrix: the first three columns are the gradient directions, and
      the remaining four are: the gradient strength (T/m), big delta (s),
      small delta (s) and echo time (s), respectively.

    The "Camino header line" (eg. VERSION: BVECTOR) is optional.
    """

    def __init__( self, data, b0_thr = 0 ) :
        """Initialize the acquisition scheme.

        Parameters
        ----------
        data : string or numpy.ndarray
            The filename of the scheme or a matrix containing the actual values
        b0_thr : float
            The threshold on the b-values to identify the b0 images (default: 0)
        """
        if type(data) is str :
            # try loading from file
            try :
                n = 0 # headers lines to skip to get to the numeric data
                with open(data) as fid :
                    for line in fid :
                        if re_match( r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', line.strip() ) :
                            break
                        n += 1
                tmp = np.loadtxt( data, skiprows=n )
            except :
                raise IOError( 'Unable to open scheme file' )
            self.load_from_table( tmp, b0_thr )
        else :
            # try loading from matrix
            self.load_from_table( data, b0_thr )


    def load_from_table( self, data, b0_thr = 0 ) :
        """Build the structure from an input matrix.

        The first three columns represent the gradient directions.
        Then, we accept two formats to describe each gradient:
            - if the shape of data is Nx4, the 4^ column is the b-value;
            - if the shape of data is Nx7, the last 4 columns are, respectively, the gradient strength, big delta, small delta and TE.

        Parameters
        ----------
        data : numpy.ndarray
            Matrix containing tall the values.
        b0_thr : float
            The threshold on the b-values to identify the b0 images (default: 0)
        """
        self.raw = data

        # number of samples
        self.nS = self.raw.shape[0]

        # set/calculate the b-values
        if self.raw.shape[1] == 4 :
            self.version = 0
            self.b = self.raw[:,3]
        elif self.raw.shape[1] == 7 :
            self.version = 1
            self.b = ( 267.513e6 * self.raw[:,3] * self.raw[:,5] )**2 * (self.raw[:,4] - self.raw[:,5]/3.0) * 1e-6 # in mm^2/s
        else :
            raise ValueError( 'Unrecognized scheme format' )

        # store information about the volumes
        self.b0_thr    = b0_thr
        self.b0_idx    = np.where( self.b <= b0_thr )[0]
        self.b0_count  = len( self.b0_idx )
        self.dwi_idx   = np.where( self.b > b0_thr )[0]
        self.dwi_count = len( self.dwi_idx )

        # ensure the directions are in the spherical range [0,180]x[0,180]
        idx = np.where( self.raw[:,1] < 0 )[0]
        self.raw[idx,0:3] = -self.raw[idx,0:3]

        # store information about each shell in a dictionary
        self.shells = []

        tmp = np.ascontiguousarray( self.raw[:,3:] )
        schemeUnique, schemeUniqueInd = np.unique( tmp.view([('', tmp.dtype)]*tmp.shape[1]), return_index=True )
        schemeUnique = schemeUnique.view(tmp.dtype).reshape((schemeUnique.shape[0], tmp.shape[1]))
        schemeUnique = [tmp[index] for index in sorted(schemeUniqueInd)]
        bUnique = [self.b[index] for index in sorted(schemeUniqueInd)]
        for i in xrange(len(schemeUnique)) :
            if bUnique[i] <= b0_thr :
                continue
            shell = {}
            shell['b'] = bUnique[i]
            if self.version == 0 :
                shell['G']     = None
                shell['Delta'] = None
                shell['delta'] = None
                shell['TE']    = None
            else :
                shell['G']     = schemeUnique[i][0]
                shell['Delta'] = schemeUnique[i][1]
                shell['delta'] = schemeUnique[i][2]
                shell['TE']    = schemeUnique[i][3]

            shell['idx']  = np.where( self.b==bUnique[i] )[0]
            shell['grad'] = self.raw[shell['idx'],0:3]
            self.shells.append( shell )


    @property
    def nS( self ) :
        return self.b0_count + self.dwi_count
