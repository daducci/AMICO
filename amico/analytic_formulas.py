import scipy.special
import numpy as np

_GAMMA_ = 2.6751525E8
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

def CylinderGPD(diff,theta,phi,R,scheme,gyro_ratio=_GAMMA_):
    """Analytic computation of the Cylinder signal for N diffusion gradients of a PGSE protocol [1].
        
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
    gyro_ratio : float (optional)
        Gyromagnetic ratio.
    
    References
    ----------
    .. [1] Ianus et al. (2013) Gaussian phase distribution approximations for oscillating gradient spin echo diffusion MRI. JMR, 227: 25-34
    """
    n_samples = scheme.nS
    dir = scheme.raw[:,:3].copy()
    modG = scheme.raw[:,3].copy()
    DELTA = scheme.raw[:,4].copy()
    delta = scheme.raw[:,5].copy()
    G = dir*np.tile(modG,(3,1)).T
    modG = np.linalg.norm(G,axis=1)
    t = DELTA-delta/3.
    unitGn = np.zeros(n_samples)
    idx = (modG > 0.0)
    cyl_vector = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])[:,None]
    unitGn[idx] = np.dot(G[idx,:],cyl_vector)[:,0] / (modG[idx] * np.linalg.norm(cyl_vector))
    alpha = np.arccos(unitGn)
    lambda_ = (_am_/R)**2
    beta_factor = 2*(R/_am_)**2 / (_am_**2-1)
    this_sum = np.zeros(n_samples)
    for i in range(n_samples):
        if this_sum[i] == 0:
            idx = (delta == delta[i]) * (DELTA == DELTA[i])
            this_sum[idx] = compute_PGSE_Sum(lambda_,beta_factor,delta[i],DELTA[i],diff)
    Sperp = np.exp(-2 * gyro_ratio**2 * modG**2 * np.sin(alpha)**2 * this_sum)
    Spar = np.exp(-t  * gyro_ratio**2 * delta**2 * modG**2 * np.cos(alpha)**2 * diff)
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
        zepSignal[i] = np.exp(-scheme.b[i]*1E6*((diffPar-diffPerp)*dotqn**2 + diffPerp))
    return zepSignal

def Ball(diff,scheme):
    """Analytic computation of the Ball signal for a given scheme.
    diff : float
        Free diffusivity (m^2/s)
    scheme : Scheme object
        PGSE scheme encoding diffusion gradient orientations and b-values
    """
    return np.exp(-scheme.b*diff*1E6)
