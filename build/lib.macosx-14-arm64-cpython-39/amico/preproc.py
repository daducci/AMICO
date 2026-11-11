import numpy as np
from scipy.optimize import minimize
import scipy.special
from amico.util import get_verbose
from dicelib.ui import ProgressBar

# Kaden's functionals
def F_norm_Diff_K(E0,Signal,sigma_diff):
    # ------- SMT functional
    sig2   = sigma_diff**2.0
    F_norm =  np.sum( ( Signal - np.sqrt( (np.pi*sig2)/2.0) * scipy.special.eval_laguerre(1.0/2.0, -1.0 * (E0**2.0) / (2.0*sig2), out=None)  )**2.0 )
    return  np.array(F_norm)

def der_Diff(E0,Signal,sigma_diff):
    E0   = np.array(E0)
    sig2 = sigma_diff**2.0
    k1   = np.sqrt((np.pi*sig2)/2.0)
    ET  = -1.0*(E0**2.0)/(2.0*sig2)
    der1 = 2.0 * ( Signal - k1 * scipy.special.eval_laguerre(0.5, ET) )
    der2 = k1 * scipy.special.hyp1f1( 0.5, 2.0, ET ) * (-0.5/(2.0*sig2)) * E0
    return der1 * der2

def debiasRician(DWI,SNR,mask,scheme):
    debiased_DWI = np.zeros(DWI.shape)
    with ProgressBar(total=mask.sum(), disable=get_verbose()<3) as pbar:
        for ix in range(DWI.shape[0]):
            for iy in range(DWI.shape[1]):
                for iz in range(DWI.shape[2]):
                    if mask[ix,iy,iz]:
                        b0 = DWI[ix,iy,iz,scheme.b0_idx].mean()
                        sigma_diff = b0/SNR
                        init_guess = DWI[ix,iy,iz,:].copy()
                        tmp = minimize(F_norm_Diff_K, init_guess, args=(init_guess,sigma_diff), method = 'L-BFGS-B', jac=der_Diff)
                        debiased_DWI[ix,iy,iz] = tmp.x
                        pbar.update()
    return debiased_DWI
