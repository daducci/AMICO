from abc import ABC, abstractmethod
import numpy as np
from scipy.special import erf, erfi, lpmv
from amico.util import ERROR

# Limits the required precision in gpd sum
_REQUIRED_PRECISION = 1e-7

# Proton gyromagnetic ratio (used in NODDI toolbox)
_GAMMA = 2.675987e8

def _gpd_sum(am, big_delta, small_delta, diff, radius, n):
    sum = 0.0
    for am_ in am:
        dam = diff * am_ * am_
        e11 = -dam * small_delta
        e2 = -dam * big_delta
        dif = big_delta - small_delta
        e3 = -dam * dif
        plus = big_delta + small_delta
        e4 = -dam * plus
        nom = 2 * dam * small_delta - 2 + (2 * np.exp(e11)) + (2 * np.exp(e2)) - np.exp(e3) - np.exp(e4)
        denom = dam * dam * am_ * am_ * (radius * radius * am_ * am_ - n)
        term = nom / denom
        sum += term
        if term < _REQUIRED_PRECISION * sum:
            break
    return sum

def _scheme2noddi(scheme):
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
    tmp = np.power(3*maxB*1E6/(2*_GAMMA*_GAMMA*Gmax*Gmax),1.0/3.0)
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

# TENSOR-BASED
class BaseTensor(ABC):
    """
    Abstract class for tensor-based compartments.

    Attributes
    ----------
    scheme : amico.scheme.Scheme
        A scheme object containing the acquisition information.
    """
    def __init__(self, scheme):
        """
        Initialize the tensor-based compartment.

        Parameters
        ----------
        scheme : amico.scheme.Scheme
            A scheme object containing the acquisition information.
        """
        self.scheme = scheme

    @abstractmethod
    def get_signal(self):
        """
        Return the computed signal.
        NOTE: Subclasses must implement this method.
        """
        pass

    def _get_signal(self, evals):
        """
        Compute the signal from the tensor-based compartment.

        Parameters
        ----------
        evals : ndarray
            The eigenvalues of the tensor.

        Returns
        -------
        signal : ndarray
            The computed signal.
        """
        evecs = np.eye(3)
        d = np.linalg.multi_dot([evecs, np.diag(evals), evecs.T])
        g_dir = self.scheme.raw[:, :3]
        b = self.scheme.b
        signal = np.zeros(len(self.scheme.raw))
        for i, g in enumerate(g_dir):
            signal[i] = np.exp(-b[i] * np.linalg.multi_dot([g.T, d, g]))
        return signal

# TENSOR
class Tensor(BaseTensor):
    """
    Class to generate the signal from a tensor model.

    Attributes
    ----------
    scheme : amico.scheme.Scheme
        A scheme object containing the acquisition information.
    """
    def get_signal(self, diff_par, diff_perp1, diff_perp2):
        """
        Compute the signal from the tensor compartment.

        Parameters
        ----------
        diff_par : float
            The parallel diffusivity of the tensor.
        diff_perp1 : float
            The first perpendicular diffusivity of the tensor.
        diff_perp2 : float
            The second perpendicular diffusivity of the tensor.

        Returns
        -------
        signal : ndarray
            The computed signal.
        """
        evals = np.array([diff_perp1, diff_perp2, diff_par])
        return super()._get_signal(evals)

# STICK
class Stick(BaseTensor):
    """
    Class to generate the signal from a stick compartment.

    Attributes
    ----------
    scheme : amico.scheme.Scheme
        A scheme object containing the acquisition information.
    """
    def get_signal(self, diff):
        """
        Compute the signal from the stick compartment.

        Parameters
        ----------
        diff : float
            The parallel diffusivity of the stick.

        Returns
        -------
        signal : ndarray
            The computed signal.
        """
        evals = np.array([0, 0, diff])
        return super()._get_signal(evals)

# ZEPPELIN
class Zeppelin(BaseTensor):
    """
    Class to generate the signal from a zeppelin compartment.

    Attributes
    ----------
    scheme : amico.scheme.Scheme
        A scheme object containing the acquisition information.
    """
    def get_signal(self, diff_par, diff_perp):
        """
        Compute the signal from the zeppelin compartment.

        Parameters
        ----------
        diff_par : float
            The parallel diffusivity of the zeppelin.
        diff_perp : float
            The perpendicular diffusivity of the zeppelin.

        Returns
        -------
        signal : ndarray
            The computed signal.
        """
        evals = np.array([diff_perp, diff_perp, diff_par])
        return super()._get_signal(evals)

# BALL
class Ball(BaseTensor):
    """
    Class to generate the signal from a ball compartment.

    Attributes
    ----------
    scheme : amico.scheme.Scheme
        A scheme object containing the acquisition information.
    """
    def get_signal(self, diff):
        """
        Compute the signal from the ball compartment.

        Parameters
        ----------
        diff : float
            The diffusivity of the ball.

        Returns
        -------
        signal : ndarray
            The computed signal.
        """
        evals = np.array([diff, diff, diff])
        return super()._get_signal(evals)

# SPHERE
class SphereGPD:
    """
    Class to generate the signal from a sphere compartment with GPD approximation.

    Attributes
    ----------
    scheme : amico.scheme.Scheme
        A scheme object containing the acquisition information.
    """
    _AM = np.array([
    2.08157597781810, 5.94036999057271, 9.20584014293667,
    12.4044450219020, 15.5792364103872, 18.7426455847748,
    21.8996964794928, 25.0528252809930, 28.2033610039524,
    31.3520917265645, 34.4995149213670, 37.6459603230864,
    40.7916552312719, 43.9367614714198, 47.0813974121542,
    50.2256516491831, 53.3695918204908, 56.5132704621986,
    59.6567290035279, 62.8000005565198, 65.9431119046553,
    69.0860849466452, 72.2289377620154, 75.3716854092873,
    78.5143405319308, 81.6569138240367, 84.7994143922025,
    87.9418500396598, 91.0842274914688, 94.2265525745684,
    97.3688303629010, 100.511065295271, 103.653261271734,
    106.795421732944, 109.937549725876, 113.079647958579,
    116.221718846033, 116.221718846033, 119.363764548757,
    122.505787005472, 125.647787960854, 128.789768989223,
    131.931731514843, 135.073676829384, 138.215606107009,
    141.357520417437, 144.499420737305, 147.641307960079,
    150.783182904724, 153.925046323312, 157.066898907715,
    166.492397790874, 169.634212946261, 172.776020008465,
    175.917819411203, 179.059611557741, 182.201396823524,
    185.343175558534, 188.484948089409, 191.626714721361
    ])

    _last_big_delta = 0.0
    _last_small_delta = 0.0
    _last_diff = 0.0
    _last_radius = 0.0
    _last_sum = 0.0

    def __init__(self, scheme):
        """
        Initialize the sphere compartment.

        Parameters
        ----------
        scheme : amico.scheme.Scheme
            A scheme object containing the acquisition information.
        """
        self.scheme = scheme

    def get_signal(self, diff, radius):
        """
        Compute the signal from the sphere compartment.

        Parameters
        ----------
        diff : float
            The diffusivity of the sphere.
        radius : float
            The radius of the sphere.

        Returns
        -------
        signal : ndarray
            The computed signal.
        """
        diff *= 1e-6
        am = self._AM / radius
        signal = np.zeros(len(self.scheme.raw))
        for i, raw in enumerate(self.scheme.raw):
            g_dir = raw[0:3]
            g = raw[3]
            big_delta = raw[4]
            small_delta = raw[5]
            if np.all(g_dir == 0):
                signal[i] = 1
            else:
                g_mods = g_dir * g
                g_mod = np.sqrt(np.dot(g_mods, g_mods))
                if big_delta != self._last_big_delta or small_delta != self._last_small_delta or diff != self._last_diff or radius != self._last_radius:
                    self._last_big_delta = big_delta
                    self._last_small_delta = small_delta
                    self._last_diff = diff
                    self._last_radius = radius
                    self._last_sum = _gpd_sum(am, big_delta, small_delta, diff, radius, 2)
                signal[i] = np.exp(-2 * _GAMMA * _GAMMA * g_mod * g_mod * self._last_sum)
        return signal

# ASTROSTICKS
class Astrosticks:
    """
    Class to generate the signal from an astrosticks compartment (sticks with distributed orientations).

    Attributes
    ----------
    scheme : amico.scheme.Scheme
        A scheme object containing the acquisition information.
    """
    def __init__(self, scheme):
        """
        Initialize the astrosticks compartment.

        Parameters
        ----------
        scheme : amico.scheme.Scheme
            A scheme object containing the acquisition information.
        """
        self.scheme = scheme

    def get_signal(self, diff):
        """
        Compute the signal from the astrosticks compartment.

        Parameters
        ----------
        diff : float
            The diffusivity of the sticks.

        Returns
        -------
        signal : ndarray
            The computed signal.
        """
        signal = np.zeros(len(self.scheme.raw))
        for i, raw in enumerate(self.scheme.raw):
            g_dir = raw[0:3]
            g = raw[3]
            b = self.scheme.b[i]
            if np.all(g_dir == 0):
                signal[i] = 1
            else:
                l_perp = 0
                l_par = -b / (g * g) * diff
                signal[i] = np.sqrt(np.pi) * 1 / (2 * g * np.sqrt(l_perp - l_par)) * np.exp(g * g * l_perp) * erf(g * np.sqrt(l_perp - l_par))
        return signal

# CYLINDER
class CylinderGPD:
    """
    Class to generate the signal from a cylinder compartment with GPD approximation.

    Attributes
    ----------
    scheme : amico.scheme.Scheme
        A scheme object containing the acquisition information.
    """
    _AM = np.array([
    1.84118307861360, 5.33144196877749,  8.53631578218074,
    11.7060038949077, 14.8635881488839, 18.0155278304879,
    21.1643671187891, 24.3113254834588, 27.4570501848623,
    30.6019229722078, 33.7461812269726, 36.8899866873805,
    40.0334439409610, 43.1766274212415, 46.3195966792621,
    49.4623908440429, 52.6050411092602, 55.7475709551533,
    58.8900018651876, 62.0323477967829, 65.1746202084584,
    68.3168306640438, 71.4589869258787, 74.6010956133729,
    77.7431620631416, 80.8851921057280, 84.0271895462953,
    87.1691575709855, 90.3110993488875, 93.4530179063458,
    96.5949155953313, 99.7367932203820, 102.878653768715,
    106.020498619541, 109.162329055405, 112.304145672561,
    115.445950418834, 118.587744574512, 121.729527118091,
    124.871300497614, 128.013065217171, 131.154821965250,
    134.296570328107, 137.438311926144, 140.580047659913,
    143.721775748727, 146.863498476739, 150.005215971725,
    153.146928691331, 156.288635801966, 159.430338769213,
    162.572038308643, 165.713732347338, 168.855423073845,
    171.997111729391, 175.138794734935, 178.280475036977,
    181.422152668422, 184.563828222242, 187.705499575101
    ])

    _last_big_delta = 0.0
    _last_small_delta = 0.0
    _last_diff = 0.0
    _last_radius = 0.0
    _last_sum = 0.0

    def __init__(self, scheme):
        """
        Initialize the cylinder compartment.

        Parameters
        ----------
        scheme : amico.scheme.Scheme
            A scheme object containing the acquisition information.
        """
        self.scheme = scheme

    def get_signal(self, diff, radius, theta=0, phi=0):
        """
        Compute the signal from the cylinder compartment.

        Parameters
        ----------
        diff : float
            The diffusivity of the cylinder.
        radius : float
            The radius of the cylinder.
        theta : float
            The angle theta of the cylinder.
        phi : float
            The angle phi of the cylinder.
        """
        diff *= 1e-6
        am = self._AM / radius
        n = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
        n_mod = np.sqrt(np.sum(n * n))
        signal = np.zeros(len(self.scheme.raw))
        for i, raw in enumerate(self.scheme.raw):
            g_dir = raw[0:3]
            g = raw[3]
            big_delta = raw[4]
            small_delta = raw[5]
            if np.all(g_dir == 0):
                signal[i] = 1
            else:
                g_mods = g_dir * g
                g_mod = np.sqrt(np.dot(g_mods, g_mods))
                gn = np.dot(g_mods, n)
                unit_gn = 0
                if g_mod == 0:
                    unit_gn = 0
                else:
                    unit_gn = gn / (g_mod * n_mod)
                omega = np.arccos(unit_gn)
                if big_delta != self._last_big_delta or small_delta != self._last_small_delta or diff != self._last_diff or radius != self._last_radius:
                    self._last_big_delta = big_delta
                    self._last_small_delta = small_delta
                    self._last_diff = diff
                    self._last_radius = radius
                    self._last_sum = _gpd_sum(am, big_delta, small_delta, diff, radius, 1)
                sr_perp = np.exp(-2 * _GAMMA * _GAMMA * g_mod * g_mod * np.sin(omega) * np.sin(omega) * self._last_sum)
                t = big_delta - small_delta / 3
                sr_par = np.exp(-t * (_GAMMA * small_delta * g_mod * np.cos(omega) * (_GAMMA * small_delta * g_mod * np.cos(omega))) * diff)
                signal[i]= sr_perp * sr_par
        return signal

# NODDI
class NODDIIntraCellular:
    def __init__(self, scheme):
        self.scheme = scheme
        self.protocol_hr = _scheme2noddi(self.scheme)

    def get_signal(self, diff_par, kappa):
        diff_par *= 1e-6
        return self._synth_meas_watson_SH_cyl_neuman_PGSE(
            np.array([diff_par, 0, kappa]),
            self.protocol_hr['grad_dirs'],
            np.squeeze(self.protocol_hr['gradient_strength']),
            np.squeeze(self.protocol_hr['delta']),
            np.squeeze(self.protocol_hr['smalldel']),
            np.array([0, 0, 1]))

    # Intra-cellular signal
    def _synth_meas_watson_SH_cyl_neuman_PGSE(self, x, grad_dirs, G, delta, smalldel, fibredir):
        d=x[0]
        R=x[1]
        kappa=x[2]

        l_q = grad_dirs.shape[0]

        # Parallel component
        LePar = self._cyl_neuman_le_par_PGSE(d, G, delta, smalldel)

        # Perpendicular component
        LePerp = self._cyl_neuman_le_perp_PGSE(d, R, G)

        ePerp = np.exp(LePerp)

        # Compute the Legendre weighted signal
        Lpmp = LePerp - LePar
        lgi = self._legendre_gaussian_integral(Lpmp, 6)

        # Compute the spherical harmonic coefficients of the Watson's distribution
        coeff = self._watson_SH_coeff(kappa)
        coeffMatrix = np.tile(coeff, (l_q, 1))

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
        shMatrix = np.tile(sh, (l_q, 1))
        for i in range(7):
            shMatrix[:,i] = np.sqrt((i+1 - .75)/np.pi)
            # legendre function returns coefficients of all m from 0 to l
            # we only need the coefficient corresponding to m = 0
            # WARNING: make sure to input ROW vector as variables!!!
            # cosTheta is expected to be a COLUMN vector.
            tmp = np.zeros((l_q))
            for pol_i in range(l_q):
                tmp[pol_i] = lpmv(0, 2*i, cosTheta[pol_i])
            shMatrix[:,i] = shMatrix[:,i]*tmp

        E = np.sum(lgi*coeffMatrix*shMatrix, 1)
        # with the SH approximation, there will be no guarantee that E will be positive
        # but we need to make sure it does!!! replace the negative values with 10% of
        # the smallest positive values
        E[E<=0] = np.min(E[E>0])*0.1
        E = 0.5*E*ePerp
        return E

    def _cyl_neuman_le_par_PGSE(self, d, G, delta, smalldel):
        # Line bellow used in matlab version removed as cyl_neuman_le_par_PGSE is called from synth_meas_watson_SH_cyl_neuman_PGSE which already casts x to d, R and kappa -> x replaced by d in arguments
        #d=x[0]

        # Radial wavenumbers
        modQ = _GAMMA*smalldel*G
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

    def _cyl_neuman_le_perp_PGSE(self, d, R, G):
        # When R=0, no need to do any calculation
        if (R == 0.00):
            LE = np.zeros(G.shape) # np.size(R) = 1
            return LE
        else:
            ERROR( '"cyl_neuman_le_perp_PGSE" not yet validated for non-zero values' )

    def _legendre_gaussian_integral(self, Lpmp, n):
        if n > 6:
            ERROR( 'The maximum value for n is 6, which corresponds to the 12th order Legendre polynomial' )
        exact = Lpmp>0.05
        approx = Lpmp<=0.05

        mn = n + 1

        I = np.zeros((len(Lpmp),mn))
        sqrtx = np.sqrt(Lpmp[exact])
        I[exact,0] = np.sqrt(np.pi)*erf(sqrtx)/sqrtx
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

    def _watson_SH_coeff(self, kappa):
        if isinstance(kappa,np.ndarray):
            ERROR( '"watson_SH_coeff()" not implemented for multiple kappa input yet' )

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

        erfik = erfi(sk)
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

class NODDIExtraCellular:
    def __init__(self, scheme):
        self.scheme = scheme
        self.protocol_hr = _scheme2noddi(self.scheme)

    def get_signal(self, diff_par, kappa, vol_ic):
        diff_par *= 1e-6
        diff_perp = diff_par * (1 - vol_ic)
        return self._synth_meas_watson_hindered_diffusion_PGSE(
            np.array([diff_par, diff_perp, kappa]),
            self.protocol_hr['grad_dirs'],
            np.squeeze(self.protocol_hr['gradient_strength']),
            np.squeeze(self.protocol_hr['delta']),
            np.squeeze(self.protocol_hr['smalldel']),
            np.array([0, 0, 1]))

    # Extra-cellular signal
    def _synth_meas_watson_hindered_diffusion_PGSE(self, x, grad_dirs, G, delta, smalldel, fibredir):
        dPar = x[0]
        dPerp = x[1]
        kappa = x[2]

        # get the equivalent diffusivities
        dw = self._watson_hindered_diffusion_coeff(dPar, dPerp, kappa)

        xh = np.array([dw[0], dw[1]])

        E = self._synth_meas_hindered_diffusion_PGSE(xh, grad_dirs, G, delta, smalldel, fibredir)
        return E

    def _watson_hindered_diffusion_coeff(self, dPar, dPerp, kappa):
        dw = np.zeros(2)
        dParMdPerp = dPar - dPerp

        if kappa < 1e-5:
            dParP2dPerp = dPar + 2.*dPerp
            k2 = kappa*kappa
            dw[0] = dParP2dPerp/3.0 + 4.0*dParMdPerp*kappa/45.0 + 8.0*dParMdPerp*k2/945.0
            dw[1] = dParP2dPerp/3.0 - 2.0*dParMdPerp*kappa/45.0 - 4.0*dParMdPerp*k2/945.0
        else:
            sk = np.sqrt(kappa)
            dawsonf = 0.5*np.exp(-kappa)*np.sqrt(np.pi)*erfi(sk)
            factor = sk/dawsonf
            dw[0] = (-dParMdPerp+2.0*dPerp*kappa+dParMdPerp*factor)/(2.0*kappa)
            dw[1] = (dParMdPerp+2.0*(dPar+dPerp)*kappa-dParMdPerp*factor)/(4.0*kappa)
        return dw

    def _synth_meas_hindered_diffusion_PGSE(self, x, grad_dirs, G, delta, smalldel, fibredir):
        dPar=x[0]
        dPerp=x[1]

        # Radial wavenumbers
        modQ = _GAMMA*smalldel*G
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

class NODDIIsotropic:
    def __init__(self, scheme):
        self.scheme = scheme
        self.protocol_hr = _scheme2noddi(self.scheme)

    def get_signal(self, diff_iso):
        diff_iso *= 1e-6
        return self._synth_meas_iso_GPD(diff_iso, self.protocol_hr)

    # Isotropic signal
    def _synth_meas_iso_GPD(self, d, protocol):
        if protocol['pulseseq'] != 'PGSE' and protocol['pulseseq'] != 'STEAM':
            ERROR( 'synth_meas_iso_GPD() : Protocol %s not translated from NODDI matlab code yet' % protocol['pulseseq'] )

        modQ = _GAMMA*protocol['smalldel'].transpose()*protocol['gradient_strength'].transpose()
        modQ_Sq = np.power(modQ,2)
        difftime = protocol['delta'].transpose()-protocol['smalldel']/3.0
        return np.exp(-difftime*modQ_Sq*d)
