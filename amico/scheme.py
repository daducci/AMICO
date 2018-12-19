from __future__ import absolute_import, division, print_function

import numpy as np
from re import match as re_match
from dipy.utils.six.moves import xrange # see http://nipy.org/dipy/devel/python3.html
#import os.path

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
        if data.ndim == 1 :
            data = np.expand_dims( data, axis=0 )
        self.raw = data


        # number of samples
        # self.nS = self.raw.shape[0] JL: incomplete getter/setter incompatible with 3.6; this is never used any as getter always returns derived value

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
        
        tmp1,tmp2 = np.unique(self.raw[:,3:],axis=0,return_index=True) # Find unique b-values or combinations of G, Delta, delta and TE (and the 1st index of each of those shells)
        tmp1 = tmp1[self.b[tmp2] > 0,:] # Remove b0
        tmp2 = tmp2[self.b[tmp2] > 0] # Remove b0
        # For each shell, find its indexes, grad, b-value and shell parameters
        for tmp3 in np.argsort(tmp2):
            shell_features = tmp1[tmp3,:]
            shell = {}
            shell['idx'] = np.where((self.raw[:,3:] == shell_features).all(axis=1))
            shell['grad'] = self.raw[shell['idx'],:3]
            shell['b'] = np.unique(self.b[shell['idx']])[0]
            if self.version == 0:
                shell['G']     = None
                shell['Delta'] = None
                shell['delta'] = None
                shell['TE']    = None
            else :
                shell['G']     = shell_features[0]
                shell['Delta'] = shell_features[1]
                shell['delta'] = shell_features[2]
                shell['TE']    = shell_features[3]
            self.shells.append( shell )


    @property
    def nS( self ) :
        return self.b0_count + self.dwi_count


