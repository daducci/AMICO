import scipy.special
import numpy as np

_am_ = scipy.special.jnp_zeros(1, 60)

def compute_PGSE_Sum(lambda_, beta, delta, DELTA, diff):
    """Computes PGSE waveform term in eq 8 from [1].
    
    References
    ----------
    .. [1] Ianus et al. (2013) Gaussian phase distribution approximations for oscillating gradient spin echo diffusion MRI. JMR, 227: 25-34
    """
    that_sum = 0
    for n in range(len(lambda_)):
        term1 = beta[n]/diff
        term2 = lambda_[n]**2 * diff
        that_sum += term1/term2 * (lambda_[n]*diff*delta - 1 + np.exp(-lambda_[n]*diff*delta) + np.exp(-lambda_[n]*diff*DELTA) - 0.5*np.exp(-lambda_[n]*diff*(DELTA-delta)) - 0.5*np.exp(-lambda_[n]*diff*(DELTA+delta)))
    return that_sum

def CylinderGPD(diff,theta,phi,R,scheme):
    """Analytic computation of the Cylinder signal for N diffusion gradients [1].

    Attributes
    ----------
    diff : float
        Diffusivity (m^2/s)
    theta : float
        Polar angle of the vector defining the fiber orientation (radians)
    phi : float
        Azimuth angle of the vector defining the fiber orientation (radians)
    R : float
        Cylinder radius (meters)
    scheme : Scheme object
        PGSE scheme object encoding diffusion gradient orientations, amplitudes, durations and separations
        
    References
    ----------
    .. [1] Ianus et al. (2013) Gaussian phase distribution approximations for oscillating gradient spin echo diffusion MRI. JMR, 227: 25-34
    """
    n_samples = scheme.nS
    dir = scheme.raw[:,:3].copy()
    for k in range(scheme.nS):
        dir[k,:] /= np.linalg.norm(dir[k,:])
    G = dir*np.tile(scheme.raw[:,3],(3,1)).T
    S = np.zeros(n_samples)
    GAMMA = 2.6751525E8
    t = scheme.raw[:,4]-scheme.raw[:,5]/3.
    unitGn = np.zeros(n_samples)
    idx = (scheme.raw[:,3] > 0)
    cyl_vector = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
    unitGn[idx] = np.dot(cyl_vector[None,:],G[idx].T) / (scheme.raw[:,3][idx] * np.linalg.norm(cyl_vector))
    alpha = np.arccos(unitGn)
    lambda_ = (_am_/R)**2
    beta_factor = 2*(R/_am_)**2 / (_am_**2-1)
    this_sum = np.zeros(n_samples)
    for i in range(n_samples):
        if this_sum[i] == 0:
            idx = (scheme.raw[:,5] == scheme.raw[i,5]) * (scheme.raw[:,4] == scheme.raw[i,4])
            this_sum[idx] = compute_PGSE_Sum(lambda_,beta_factor,scheme.raw[i,5],scheme.raw[i,4],diff)
    Sperp = np.exp(-2 * GAMMA**2 * scheme.raw[:,3]**2 * np.sin(alpha)**2 * this_sum)
    # the restricted parallel signal
    Spar = np.exp(-t  * GAMMA**2 * scheme.raw[i,5]**2 * scheme.raw[:,3]**2 * np.cos(alpha)**2 * diff)
    # the restricted signal
    return Sperp * Spar

def Zeppelin(diffPar,theta,phi,diffPerp,scheme):
    """Analytic computation of the Zeppelin signal for a given scheme.
    diffPar : float
        Parallel diffusivity (m^2/s)
    theta : float
        Polar angle of the vector defining the Zeppelin orientation (radians)
    phi : float
        Azimuth angle of the vector defining the Zeppelin orientation (radians)
    diffPerp : float
        Perpendicular diffusivity (m^2/s)
    scheme : Scheme object
        PGSE scheme encoding diffusion gradient orientations and b-values
    """
    sinT = np.sin(theta)
    cosT = np.cos(theta)
    sinP = np.sin(phi)
    cosP = np.cos(phi)
    # Zeppelin orientation
    n = np.array([cosP * sinT,sinP * sinT,cosT])
    zepSignal = np.zeros(scheme.nS)
    for i in range(scheme.nS):
        dir_norm = np.linalg.norm(scheme.raw[i,:3])
        if dir_norm > 0:
            gd = scheme.raw[i,:3]/dir_norm
        else:
            gd = scheme.raw[i,:3]
        #dot product of the cylinder orientation n and the wavenumber q
        dotqn= np.dot(n,gd)
        zepSignal[i] = np.exp(-scheme.b*1E6*((diffPar-diffPerp)*dotqn**2 + diffPerp))
    return zepSignal

def Ball(diff,scheme):
    """Analytic computation of the Ball signal for a given scheme.
    diff : float
        Free diffusivity (m^2/s)
    scheme : Scheme object
        PGSE scheme encoding diffusion gradient orientations and b-values
    """
    return np.exp(-scheme.b*diff*1E6)