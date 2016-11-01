import numpy as np
import os.path

def fsl2scheme( bvalsFilename, bvecsFilename, schemeFilename = None, bStep = 1.0):
    """Create a scheme file from bvals+bvecs and write to file.

    If required, b-values can be rounded up to a specific threshold (bStep parameter).

    Parameters
    ----------
    :param str bvalsFilename: The path to bval file.
    :param str bvecsFilename: The path to bvec file.
    :param str schemeFilename: The path to output scheme file (optional).
    :param float or list or np.bStep: If bStep is a scalar, round b-values to nearest integer multiple of bStep. If bStep is a list, it is treated as an array of shells in increasing order. B-values will be forced to the nearest shell value.
    """

    if not os.path.exists(bvalsFilename):
        raise RuntimeError( 'bvals file not exist:' + bvalsFilename )
    if not os.path.exists(bvecsFilename):
        raise RuntimeError( 'bvecs file not exist:' + bvecsFilename )

    if schemeFilename is None:
        schemeFilename = os.path.splitext(bvalsFilename)[0]+".scheme"

    # load files and check size
    bvecs = np.loadtxt( bvecsFilename )
    bvals = np.loadtxt( bvalsFilename )

    if bvecs.ndim !=2 or bvals.ndim != 1 or bvecs.shape[0] != 3 or bvecs.shape[1] != bvals.shape[0]:
        raise RuntimeError( 'incorrect/incompatible bval/bvecs files' )

    # if requested, round the b-values
    bStep = np.array(bStep, dtype = np.float)
    if bStep.size == 1 and bStep > 1.0:
        print "-> Rounding b-values to nearest multiple of %s" % np.array_str(bStep)
        bvals = np.round(bvals/bStep) * bStep
    elif bStep.size > 1:
        print "-> Setting b-values to the closest shell in %s" % np.array_str(bStep)
        for i in range(0, bvals.size):
            diff = min(abs(bvals[i] - bStep))
            ind = np.argmin(abs(bvals[i] - bStep))

            # warn if b > 99 is set to 0, possible error
            if (bStep[ind] == 0.0 and diff > 100) or (bStep[ind] > 0.0 and diff > bStep[ind] / 20.0):
                # For non-zero shells, warn if actual b-value is off by more than 5%. For zero shells, warn above 50. Assuming s / mm^2
                print "   Warning: measurement %d has b-value %d, being forced to %d\n'" % i, bvals[i], bStep[ind]

            bvals[i] = bStep[ind]

    # write corresponding scheme file
    np.savetxt( schemeFilename, np.c_[bvecs.T, bvals], fmt="%.06f", delimiter="\t", header="VERSION: BVECTOR", comments='' )
    print "-> Writing scheme file to [ %s ]" % schemeFilename
    return schemeFilename
