import numpy as np
import scipy.optimize
from numba import njit, prange
import pandas as pd
import scipy.signal as sig
from matplotlib import pyplot as plt
from numpy.linalg import norm   # for shriley bckgrd
from scipy import ndimage
from scipy.optimize import curve_fit, minimize
from scipy.signal import convolve2d
from scipy.fft import fft, ifft, ifftshift

import piva.constants as const

from itertools import groupby
from typing import Union, Callable, Any


# +--------------------------------+ #
# | Fitting functions and routines | # ========================================
# +--------------------------------+ #


def gaussian(x: np.ndarray, a: float = 1, mu: float = 0, sigma: float = 1) \
        -> np.ndarray:
    r"""
        Gaussian line shape function represented as:

    .. math::
        G(x) = a_0 \exp{\bigg(-\frac{(x - \mu)^2}{2\sigma^2}\bigg)}

    :param x: arguments
    :param a: normalization factor
    :param mu: expected value
    :param sigma: standard deviation
    :return: function values
    """

    return a * np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)


def two_gaussians(x: np.ndarray, a0: float = 1, mu0: float = 0,
                  sigma0: float = 1, a1: float = 1, mu1: float = 0,
                  sigma1: float = 1) -> np.ndarray:
    """
    Two overlapping gaussians. See :func:`gaussian` for more details.

    :param x: arguments
    :param a0: normalization factor of the 1st gaussian
    :param mu0: expected value of the 1st gaussian
    :param sigma0: standard deviation of the 1st gaussian
    :param a1: normalization factor of the 2nd gaussian
    :param mu1: expected value of the 2nd gaussian
    :param sigma1: standard deviation of the 2nd gaussian
    :return: function values
    """

    return gaussian(x, a0, mu0, sigma0) + gaussian(x, a1, mu1, sigma1)


def lorentzian(x: np.ndarray, a0: float, mu: float, gamma: float,
               resol: float = 0) -> np.ndarray:
    r"""
    Lorentzian line shape function represented as:

    .. math::
     L(x) = \frac{a_0}{2\pi} \frac{\Gamma}{(x-\mu)^2 + \frac{1}{4} \Gamma^2}

    :param x: arguments
    :param a0: normalization factor
    :param mu: expected value
    :param gamma: full width at half maximum
    :param resol: resolution parameter for convolving with gaussian;
            optional, default = 0
    :return: function values
    """

    y = a0 * (1 / (2 * np.pi)) * gamma / ((x - mu) ** 2 + (.5 * gamma) ** 2)
    return ndimage.gaussian_filter(y, resol)


def asym_lorentzian(x: np.ndarray, a0: float, mu: float, gamma: float,
                    alpha: float = 0, resol: float = 0) -> np.ndarray:
    r"""
    Asymmetric Lorentzian line shape represented as:

    .. math::
        L(x) = \frac{a_0}{2\pi} \frac{\Gamma(x)}{(x-\mu)^2 +
        \frac{1}{4} \Gamma(x)^2}

    where

    .. math::
        \Gamma(x) = \frac{2 \Gamma_0}{1 + \exp{(\alpha(x - \mu))}}

    :param x: arguments
    :param a0: normalization factor
    :param mu: expected value
    :param gamma: full width at half maximum
    :param alpha: asymmetric factor
    :param resol: resolution parameter for convolving with gaussian;
            optional, default = 0
    :return: function values
    """

    if resol == 0:
        gamma *= 2 / (1 + np.exp(alpha * (x - mu)))
        return a0 * (1 / (2 * np.pi)) * gamma / \
               ((x - mu) ** 2 + (.5 * gamma) ** 2)
    else:
        gamma *= 2 / (1 + np.exp(alpha * (x - mu)))
        y = a0 * (1 / (2 * np.pi)) * gamma / \
            ((x - mu) ** 2 + (.5 * gamma) ** 2)
        return ndimage.gaussian_filter(y, sigma=resol)


def two_lorentzians(x: np.ndarray, a0: float, mu0: float, gamma0: float,
                    a1: float, mu1: float, gamma1: float,
                    resol: float = 0) -> np.ndarray:
    """
    Two overlapping lorentzians. See :func:`lorentzian` for more details.

    :param x: arguments
    :param a0: normalization factor of the 1st lorentzian
    :param mu0: expected value of the 1st lorentzian
    :param gamma0: full width at half maximum of the 1st lorentzian
    :param a1: normalization factor of the 2nd lorentzian
    :param mu1: expected value of the 2nd lorentzian
    :param gamma1: full width at half maximum of the 2nd lorentzian
    :param resol: resolution parameter for convolving with gaussian;
                  optional, default = 0
    :return: function values
    """

    y = lorentzian(x, a0, mu0, gamma0) + lorentzian(x, a1, mu1, gamma1)
    return ndimage.gaussian_filter(y, resol)


def lorentzian_dublet(x: np.ndarray, *p: Union[list, np.ndarray, tuple],
                      delta: float = 1, line: str = 'f') -> np.ndarray:
    """
    Return a two lorentzian line shapes representing doublet of the XPS
    core-level spectra.

    :param x: arguments (energies)
    :param p: lorentzian parameters. See :func:`lorentzian` for more details.
    :param delta: energy splitting of the doublet spectral lines.
    :param line: type of the orbital line. Can be `'p'`, `'d'` or `'f'`.
    :return: function values (spectra)
    """

    if line == 'p':
        coeff = 0.5
    elif line == 'd':
        coeff = 2/3
    elif line == 'f':
        coeff = .75
    else:
        coeff = 1
    p = p[0]
    return lorentzian(x, p[0], p[1], p[2]) + \
           lorentzian(x, p[0] * coeff, p[1] - delta, p[2])


def fit_n_dublets(data: np.ndarray, x: np.ndarray,
                  a0: Union[float, list, np.ndarray, tuple],
                  mu: Union[float, list, np.ndarray, tuple],
                  gamma: Union[float, list, np.ndarray, tuple],
                  delta: float, constr: tuple = None,
                  fit_delta: bool = False,
                  line: str = 'd') -> list:
    """
    Fit multiple lorentzian doublets, corresponding to different chemical
    shifts of the core-level line.

    :param data: XPS data
    :param x: arguments (energy axis)
    :param a0: normalization factors of each doublet.
               len(a0) corresponds to number of individual chemical shifts.
    :param mu: expected values of each doublet.
               len(mu) corresponds to number of individual chemical shifts.
    :param gamma: full width at half maximum of each doublet.
                  len(gamma) corresponds to number of individual chemical shifts.
    :param delta: energy splitting between doublet lines
    :param constr: constrains on the fitting, according to
                   :func:`scipy.curve_fit` format
    :param fit_delta: if :py:obj:`True`, fit also splitting delta;
                      default :py:obj:`False`
    :param line: type of the orbital line. Can be `'p'`, `'d'` or `'f'`.
    :return: results of the fitting in format:

        - ``res[0]`` - :class:`np.ndarray`; fitted_profiles,
        - ``res[1]`` - :class:`np.ndarray`; fitted parameters,
        - ``res[2]`` - :class:`np.ndarray`; parameters' errors,
        - ``res[3]`` - :class:`np.ndarray`; parameters' covariance matrix,
        - ``res[4]`` - :class:`pandas.DataFrame`, contains results in
          summarized form
    """

    pars = []
    pcovs = []
    errs = []
    pi = []
    for i in range(len(mu)):
        pi.append(a0[i])
        pi.append(mu[i])
        pi.append(gamma[i])

    if fit_delta:
        def fit_func(x, a0, mu, gamma, delta):
            return lorentzian_dublet(x, a0, mu, gamma, delta, line=line)
    else:
        def fit_func(x, *pars):
            n = len(pars) // 3
            res = lorentzian_dublet(x, pars[:3], delta=delta, line=line)
            for i in range(1, n):
                ip, ik = 3 * i, 3 * (i + 1)
                res += lorentzian_dublet(x, pars[ip:ik], delta=delta, line=line)
            """ Wrapper around delta and line."""
            return res

    if constr is None:
        popt, pcov = curve_fit(fit_func, x, data, p0=pi)
    else:
        popt, pcov = curve_fit(fit_func, x, data, p0=pi, bounds=constr)
    if fit_delta:
        print('not working yet')
        return
    else:
        fitted_profiles = lorentzian_dublet(x, popt[0:3], delta=delta)
        pars.append(popt[0:3])
        pcovs.append(pcov[0:3])
        errs.append(np.sqrt(np.diag(pcov[0:3])))
        for i in range(1, len(mu)):
            ip, ik = 3 * i, 3 * (i + 1)
            fitted_profiles += lorentzian_dublet(x, popt[ip:ik], delta=delta)
            pars.append(popt[ip:ik])
            pcovs.append(pcov[ip:ik])
            errs.append(np.sqrt(np.diag(np.abs(pcov[ip:ik]))))

    # create panda dataframe with results
    mus = []
    mus2 = []
    gammas = []
    dmus = []
    dgammas = []
    as0 = []
    as1 = []
    if len(pars[0]) == 3:
        for i in range(len(pars)):
            mus.append(pars[i][1])
            dmus.append(errs[i][1])
            gammas.append(pars[i][2])
            dgammas.append(errs[i][2])
            mus2.append(pars[i][1]-delta)
            as0.append(pars[i][0])
        d = {'a0': np.array(as0),
             'mu_0, eV': mus,
             # 'dmu_0, eV': dmus,
             # 'mu_1, eV': mus2,
             'Gamma, eV': gammas
             # 'dGamma, eV': dgammas,
             }
    elif len(pars[0]) == 5:
        for i in range(len(pars)):
            gammas.append(pars[i][2])
            dgammas.append(errs[i][2])
        d = {'Gamma, deg/A^-1': gammas,
             'dGamma, deg/A^-1': dgammas,
             }
    df = pd.DataFrame(data=d)

    return [fitted_profiles, pars, errs, pcovs, df]


def shirley_calculate(x: np.ndarray, y: np.ndarray, tol: float = 1e-5, 
                      maxit: int = 10) -> np.ndarray:
    """
    Calculate the best auto-Shirley background S for a dataset (``x``,``y``).
    Finds the biggest peak and then uses the minimum value either side of this
    peak as the terminal points of the Shirley background.

    .. note::
        Implemented from:
        https://github.com/kaneod/physics/blob/master/python/specs.py
    
    :param x: arguments, energies
    :param y: values, XPS signal/intensity
    :param tol: convergence criterion
    :param maxit: maximum number of iterations
    :return: shirley background
    """

    # Make sure we've been passed arrays and not lists.
    x = np.array(x)
    y = np.array(y)

    # Sanity check: Do we actually have data to process here?
    if not (x.any() and y.any()):
        print("specs.shirley_calculate: One of the arrays x or y is empty. "
              "Returning zero background.")
        return np.zeros(x.shape)

    # Next ensure the energy values are *decreasing* in the array,
    # if not, reverse them.
    if x[0] < x[-1]:
        is_reversed = True
        x = x[::-1]
        y = y[::-1]
    else:
        is_reversed = False

    # Locate the biggest peak.
    maxidx = abs(y - np.amax(y)).argmin()

    # It's possible that maxidx will be 0 or -1. If that is the case,
    # we can't use this algorithm, we return a zero background.
    if maxidx == 0 or maxidx >= len(y) - 1:
        print("specs.shirley_calculate: Boundaries too high for algorithm: "
              "returning a zero background.")
        return np.zeros(x.shape)

    # Locate the minima either side of maxidx.
    lmidx = abs(y[0:maxidx] - np.amin(y[0:maxidx])).argmin()
    rmidx = abs(y[maxidx:] - np.amin(y[maxidx:])).argmin() + maxidx
    xl = x[lmidx]
    yl = y[lmidx]
    xr = x[rmidx]
    yr = y[rmidx]

    # Max integration index
    imax = rmidx - 1

    # Initial value of the background shape B. The total background S = yr + B,
    # and B is equal to (yl - yr) below lmidx and initially zero above.
    B = np.zeros(x.shape)
    B[:lmidx] = yl - yr
    Bnew = B.copy()

    it = 0
    while it < maxit:
        # Calculate new k = (yl - yr) / (int_(xl)^(xr) J(x') - yr - B(x') dx')
        ksum = 0.0
        for i in range(lmidx, imax):
            ksum += (x[i] - x[i + 1]) * 0.5 * (y[i] + y[i + 1]
                                               - 2 * yr - B[i] - B[i + 1])
        k = (yl - yr) / ksum
        # Calculate new B
        for i in range(lmidx, rmidx):
            ysum = 0.0
            for j in range(i, imax):
                ysum += (x[j] - x[j + 1]) * \
                        0.5 * (y[j] + y[j + 1] - 2 * yr - B[j] - B[j + 1])
            Bnew[i] = k * ysum
        # If Bnew is close to B, exit.
        if norm(Bnew - B) < tol:
            B = Bnew.copy()
            break
        else:
            B = Bnew.copy()
        it += 1

    if it >= maxit:
        print("specs.shirley_calculate: "
              "Max iterations exceeded before convergence.")
    if is_reversed:
        return (yr + B)[::-1]
    else:
        return yr + B


def get_linear(points: list) -> Callable:
    """
    Get linear fit between two points.

    :param points: list of two points in a format:
            points = [[``x0``, ``x1``], [``y0``, ``y1``]]
    :return: linear function: f(x) = ax + b
    """

    x = points[0]
    y = points[1]

    pars = np.polyfit(x, y, 1)
    fun = lambda arg: pars[0] * arg + pars[1]
    return fun


def print_fit_results(p: np.ndarray, cov: np.ndarray, labels: list = None) -> \
        None:
    """
    Print fit results in a nice :class:`pandas.DataFrame` format

    :param p: fitted parameters of size `n`
    :param cov: covariance matrix of shape (`n` x `n`)
    :param labels: list of parameter's names
    """

    if labels is None:
        labels = []
        for i in range(len(p)):
            labels.append(f'p{i}')

    d = {'parameters': labels,
         'values': p,
         'errors': np.sqrt(np.diagonal(cov))
         }

    df = pd.DataFrame(data=d)
    pd.set_option('display.max_columns', 50)
    print(df)


# +----------------+ #
# | ARPES analysis | # ========================================================
# +----------------+ #


def kk_im2re(gamma: np.ndarray, vF: float = 1) -> np.ndarray:
    """
    Transform imaginary part of the self-energy (scattering rates) to real
    parts using Kramers-Kronig relations in Fourier transform approach.

    :param gamma: imaginary part (scattering rates)
    :param vF: Fermi velocity
    :return: real part of the self-energy
    """

    im = 0.5 * vF * np.array(gamma) * 1j
    im = np.hstack((im, np.flip(im[1:])))
    im[0] *= 0.5
    im_t = np.fft.ifft(im)
    n = im.size // 2 + 1
    # impose causality
    im_t[n:] = np.zeros(n - 1)
    re = 2 * np.fft.fftshift(np.fft.fft(im_t))
    return re


def kk_re2im(re: np.ndarray) -> np.ndarray:
    """
    Transform real part of the self-energy (energy differences) to imaginary
    part using Kramers-Kronig relations in Fourier transform approach.

    :param re: real part of the self-energy
    :return: imaginary part
    """

    re = np.array(re)
    re = np.hstack((np.array(re), np.flip(re[1:])))
    re[0] *= 0.5
    re_t = np.fft.ifft(re)
    n = re.size // 2 + 1
    # impose causality
    re_t[n:] = np.zeros(n - 1)
    re = 2 * np.fft.fftshift(np.fft.fft(re_t))
    return re


def find_vF(gamma: np.ndarray, re_disp: np.ndarray, vF0: float = 3,
            method: str = 'Nelder-Mead') -> float:
    """
    Iteratively find a value of the Fermi velocity by minimizing the difference
    between real parts of self energy calculated by Kramers-Kronig
    transformation of the scattering rates and obtained directly from the
    experiment.

    :param gamma: scattering rates
    :param re_disp: real part of the self-energy; difference between real and
                    bare band dispersion
    :param vF0: initial guess Fermi velocity
    :param method: minimization method, see :meth:`scipy.optimize.minimize`
                   for more details
    :return: minimized value
    """

    # function to minimize: difference between experimentally and
    # KK-transformed real parts
    def disp_kk_diff(vF, gamma, re_disp):
        re_kk = kk_im2re(gamma, vF=vF).real
        return np.sum(np.abs(re_kk[:re_disp.size] - re_disp))

    res = minimize(disp_kk_diff, x0=vF0, args=(gamma, re_disp,), method=method)
    return res


def find_vF_v2(omega: np.ndarray, km: np.ndarray, kF: float, gamma: np.ndarray,
               vF0: float = 3, method: str = 'Nelder-Mead') -> float:
    """
    Iteratively find a value of the Fermi velocity by minimizing the difference
    between real parts of self energy calculated by Kramers-Kronig
    transformation of the scattering rates and obtained directly from the
    experiment.

    :param omega: energy axis
    :param km: momenta positions of the fitted MDCs
    :param kF: Fermi momentum
    :param gamma: scattering rates
    :param re_disp: real part of the self-energy; difference between real and
                    bare band dispersion
    :param vF0: initial guess Fermi velocity
    :param method: minimization method, see :meth:`scipy.optimize.minimize`
                   for more details
    :return: minimized value
    """

    # function to minimize: difference between experimental dispersion and
    # KK-transformed real parts
    def disp_kk_diff(vF, omega, km, kF, gamma):
        re_kk = kk_im2re(gamma, vF=vF).real
        re_disp = omega - vF * (-km + kF)
        # re_disp = omega - vF * (km - kF)
        # return np.sum(np.abs((re_kk[:re_disp.size] - re_disp)))
        return np.sum((re_kk[:re_disp.size] - re_disp)**2)

    res = minimize(disp_kk_diff, x0=vF0, args=(omega, km, kF, gamma,),
                   method=method)
    return res


def get_chi(ek: Callable, ek_kwargs: dict, kx: np.ndarray, qx: np.ndarray,
            ky: Union[np.ndarray, None] = None,
            kz: Union[np.ndarray, float] = 0.,
            qy: Union[np.ndarray, None] = None,
            qz: Union[np.ndarray, None] = None,
            in_plnae: bool = True, crop: bool = False) -> np.ndarray:
    """
    Calculate real part of the Lindhard susceptibility at the limit omega -> 0.

    .. note::
        Algorithm implemented for a single-band case, based on the approach
        described by `Inosov et al.
        <https://iopscience.iop.org/article/10.1088/1367-2630/10/12/125027>`_.

    :param ek: function returning electronic dispersion, *e.g.*, using
               tight-binding model
    :param ek_kwargs: keyword arguments for the dispersion function
    :param kx: momentum axis along x direction
    :param qx: scattering q-vectors along x direction
    :param ky: momentum axis along y direction; if :class:`None`, equal to `kx`
    :param kz: momentum axis along y direction.
    :param qy: scattering q-vectors along y direction
    :param qz: scattering q-vectors along z direction
    :param in_plnae: determine direction of the calculated susceptibility;
                     if :py:obj:`True` - compute in-plane scattering
                     (qx, qy, qz = 0), if :py:obj:`False` - compute diagonal
                     scattering (qx, qy=qx, qz)
    :param crop: if :py:obj:`True` - compute only one quarter of the map. Helps
                 to save a lot of time taking advantage of the symmetry
                 conditions.
    :return: 2D Lindhard susceptibility map
    """

    if ky is None:
        ky = kx
    chi = np.zeros((qx.size, qx.size))
    if crop:
        x_range = range(chi.shape[0] // 2, chi.shape[0])
        y_range = range(chi.shape[1] // 2, chi.shape[1])
    else:
        x_range, y_range = range(chi.shape[0]), range(chi.shape[1])

    for i in x_range:
        for j in y_range:
            if in_plnae:
                qxi, qyi = qx[i], qy[j]
                ek_0 = ek(kx - 0.0, ky=ky - 0.0, kz=kz - 0.0, **ek_kwargs)
                ek_q = ek(kx - qxi, ky=ky - qyi, kz=kz - 0.0, **ek_kwargs)
            else:
                qxi, qzi = qx[i], qz[j]
                ek_0 = ek(kx - 0.0, ky=ky - 0.0, kz=kz - 0.0, **ek_kwargs)
                ek_q = ek(kx - qxi, ky=ky - qxi, kz=kz - qzi, **ek_kwargs)

            integrand = fermi_dirac(ek_q) - fermi_dirac(ek_0)
            den = ek_0 - ek_q
            den = np.where(den == 0., 1e-10, den)
            integrand /= den

            if crop:
                chi[i, j], chi[-i - 1, j], chi[i, -j - 1], chi[
                    -i - 1, -j - 1] = np.sum(integrand), np.sum(
                    integrand), np.sum(integrand), np.sum(integrand)
            else:
                chi[i, j] = np.sum(integrand)

    return chi / chi.size


def single_q(qi: float, Qi: float) -> float:
    """
    Get a single vector scattered by a density wave ordering vector Q.

    :param qi: q-vector axis
    :param Qi: density wave Q vector along particular direction
    :return: one-direction component of the Lorentzian-like density wave
             modulation
    """

    return np.power((qi + Qi * np.pi), 2)


def get_1D_modulation(qx: np.ndarray, permutations: list,
                      Q: float = 0.5, N: float = 10,
                      a: float = np.pi) -> np.ndarray:
    """
    1D Lorentzian modulation centered at the modulation vector Q.

    :param qx: q-axis along the x direction, q-axis along the y direction
               is assumed to be the same
    :param permutations: all symmetry equivalent permutations of the
                         considered scattering vector Q
    :param Q: density wave vector
    :param N: modulation span in a number of unit cells
    :param a: lattice constant
    :return: Lorentzian-like density wave modulation
    """

    res = np.zeros(qx.size)
    gamma = np.power(N * a, -1)
    fac = gamma / np.pi

    for pi in permutations:
        Qpi = pi * Q
        res += fac * np.power(np.power(gamma, 2) + single_q(qx, Qpi), -1)

    return np.array(res, dtype=complex)


def get_2D_modulation(qx: np.ndarray, permutations: list,
                      Q: np.ndarray = np.array([0.5, 0.5]),
                      N: float = 10, a: float = np.pi) -> np.ndarray:
    """
    2D Lorentzian modulation centered at the modulation vector Q.

    :param qx: q-axis along the x direction, q-axis along the y direction
               is assumed to be the same
    :param permutations: all symmetry equivalent permutations of the
                         considered scattering vector Q
    :param Q: density wave vector
    :param N: modulation span in a number of unit cells
    :param a: lattice constant
    :return: Lorentzian-like density wave modulation
    """

    res = np.zeros((qx.size, qx.size))
    gamma = np.power(N * a, -1)
    fac = gamma / (2 * np.pi)

    for i, qxi in enumerate(qx):
        tmp_qx = np.ones_like(qx, dtype=float) * qxi
        for pi in permutations:
            Qpi = pi * Q
            res[i, :] += fac * np.power(np.power(gamma, 2) +
                                        single_q(tmp_qx,  Qpi[0]) +
                                        single_q(qx,  Qpi[1]), -1.5)

    return np.array(res, dtype=complex)


def get_3D_modulation(qx: np.ndarray, qz: np.ndarray, permutations: list,
                      Q: np.ndarray = np.array([0.5, 0.5, 0.5]),
                      N: int = 10, a: float = np.pi) -> np.ndarray:
    """
    3D Lorentzian modulation centered at the modulation vector Q.

    :param qx: q-axis along the x direction, q-axis along the y direction
               is assumed to be the same
    :param qz: q-axis along the z direction
    :param permutations: all symmetry equivalent permutations of the
                         considered scattering vector Q
    :param Q: density wave vector
    :param N: modulation span in a number of unit cells
    :param a: lattice constant
    :return: Lorentzian-like density wave modulation
    """

    res = np.zeros((qx.size, qx.size, qx.size))
    gamma = np.power(N * a, -1)
    fac = np.power(np.pi * gamma, -2)

    for i, qxi in enumerate(qx):
        for j, qyi in enumerate(qx):
            tmp_qx, tmp_qy = np.ones_like(qx, dtype=float) * qxi, \
                             np.ones_like(qx, dtype=float) * qyi
            for pi in permutations:
                Qpi = pi * Q
                res[i, j, :] += fac * np.power(np.power(gamma, 2) +
                                               single_q(tmp_qx, Qpi[0]) +
                                               single_q(tmp_qy, Qpi[1]) +
                                               single_q(qz, Qpi[2]), -2)

    return np.array(res, dtype=complex)


def get_self_energy_of_FDW(Q: tuple, K: tuple, ek: Callable,
                              ek_kwargs: dict, omega: np.ndarray,
                              Pq: np.ndarray, eta: float = 0.1) -> np.ndarray:
    """
    Calculate self-energy of the fluctuating density wave order in a 3D case.

    .. note::
        Algorithm based on the approach described by `Hashimoto et al.
        <https://doi.org/10.1038/nmat4116>`_.

    :param Q: 3-element tuple of the (qx, qy, qz) axes, each being a 1D
              :class:`np.ndarray`
    :param K: 3-element tuple of the (kx, ky, kz) axes, each being a 1D
              :class:`np.ndarray`
    :param ek: function returning electronic dispersion, *e.g.*, using
               tight-binding model
    :param ek_kwargs: keyword arguments for the dispersion function
    :param omega: real frequencies (energies), must be the same shape as `ek`
    :param Pq: 3D Lorentzian describing the modulation
    :param eta: imaginary factor, responsible for broadening of the spectra
    :return: self energy of the density wave modulation
    """

    dim = len(Q)
    if dim == 1:
        kx = K
        qx = Q
        res = np.zeros_like(qx.size, dtype=complex)
        x_range, y_range, z_range = range(res.shape[0]), range(1), range(1)
    elif dim == 2:
        kx, ky = K
        qx, qy = Q
        res = np.zeros_like((qx.size, qy.size), dtype=complex)
        x_range, y_range, z_range = range(res.shape[0]), \
                                    range(res.shape[1]), \
                                    range(1)
    elif dim == 3:
        kx, ky, kz = K
        qx, qy, qz = Q
        res = np.zeros_like((qx.size, qy.size, qz.size), dtype=complex)
        x_range, y_range, z_range = range(res.shape[0]), \
                                    range(res.shape[1]), \
                                    range(res.shape[2])
    else:
        print("Wrong dimensions.")
        return

    for xi in x_range:
        for yi in y_range:
            for zi in z_range:
                # the notation here is reversed: summation should go over all
                # qs, for each k. But as long as these two are identical it
                # doesn't matter
                if dim == 1:
                    qxi = qx[xi]
                    ek = ek(kx + qxi, **ek_kwargs)
                    ek = np.array(np.where(ek == 0, 1e-10, ek), dtype=complex)
                    res = np.sum(Pq / ((omega + eta * 1.j) - ek))
                elif dim == 2:
                    qxi, qyi = qx[xi], qy[yi]
                    ek = ek(kx + qxi, ky=kx + qyi, **ek_kwargs)
                    ek = np.array(np.where(ek == 0, 1e-10, ek), dtype=complex)
                    res[xi, yi] = np.sum(Pq / ((omega + eta * 1.j) - ek))
                elif dim == 3:
                    qxi, qyi, qzi = qx[xi], qy[yi], qz[zi]
                    ek = ek(kx + qxi, ky=kx + qyi, kz=kz + qzi, **ek_kwargs)
                    ek = np.array(np.where(ek == 0, 1e-10, ek), dtype=complex)
                    res[xi, yi, zi] = np.sum(Pq / ((omega + eta * 1.j) - ek))

    return res / res.size


def get_A(omega: np.ndarray, ek: np.ndarray, eta: float = 0.075,
          sigma: Union[tuple, None] = None) -> np.ndarray:
    """
    Calculate spectral function A(k, omega) based on known dispersion.

    :param omega: real frequencies (energies), must be the same shape as `ek`
    :param ek: electronic dispersion obtained using, *e.g.*,
               tight-binding model
    :param eta: imaginary factor, responsible for broadening of the spectra
    :param sigma: self-energy of electronic excitations. If provided, include
                  contributions from various electronic excitations in the
                  spectral function.
    :return: spectral function A(k, omega)
    """

    Sigma = np.zeros_like(ek)
    if type(sigma) == tuple:
        for sigma_i in sigma:
            Sigma += sigma_i

    G = (omega + eta * 1.j) - ek - Sigma
    return (-1 / np.pi) * np.imag(np.power(G, -1))


# +--------------------------------------------------+ #
# | Resolution fitting functions and PGM calibration | # ======================
# +--------------------------------------------------+ #


def pgm_calibration(hv: np.ndarray, error_offset: float, dtheta: float,
                    dbeta: float, cff: float = 2.25, k: float = 1,
                    lines_per_mm: float = 300) -> np.ndarray:
    """
    Function for PGM motors calibration.

    .. note::
        Follows procedure described by
        `Weiss et al. <https://doi.org/10.1016/S0168-9002(01)00375-8>`_


    :param hv: measured photon energies
    :param error_offset: 
    :param dtheta: correction for theta angle
    :param dbeta: correction for beta angle
    :param cff: cff parameter
    :param k: 
    :param lines_per_mm: lines per mm, parameter characterizing PGM
                         monochromator
    :return: 
    """
    
    # convert from 'lines per mm' to 'lines per nm'
    N = lines_per_mm * (1e-6)

    # convert energy in eV to wavelength in nm
    wl = const.convert_eV_nm(hv)
    # convert degrees to radians
    dt = np.deg2rad(dtheta)
    db = np.deg2rad(dbeta)

    coeff = cff ** 2 - 1
    alpha = (coeff / (wl * N * k)) ** 2
    alpha = np.arcsin((np.sqrt(cff ** 2 + alpha) - 1) *
                      ((wl * N * k) / coeff))

    beta = -np.arccos(cff * np.cos(alpha))
    theta = 0.5 * (alpha - beta)

    # wavelength from grating equation, taking into account offsets
    actualEnergy = 2 * np.cos(theta + dt) * \
                   np.sin(theta + dt + beta + db) / (N * k)

    return const.convert_eV_nm(actualEnergy) - hv + error_offset


def fit_PGM_calibration(data: np.ndarray, hv: np.ndarray, 
                        error_offset: float = -0.06, dtheta: float = 0.001,
                        dbeta: float = -0.001, cff: float = 2.25, k: float = 1,
                        lines_per_mm: int = 300) -> tuple:
    """
    Run PGM calubration procedure (see :func:`pgm_calibration` for more
    details) and find angles corrections for monochromator.

    :param data: 
    :param hv: measured photon energies
    :param error_offset: 
    :param dtheta: correction for theta angle
    :param dbeta: correction for beta angle
    :param cff: cff parameter
    :param k: 
    :param lines_per_mm: lines per mm, parameter characterizing PGM 
                         monochromator
    :return: results of the fitting in format:

        - ``res[0]`` - :class:`np.ndarray`; fitted parameters,
        - ``res[1]`` - :class:`np.ndarray`; parameters' covariance matrix
    """
    # Initial guess and bounds for parameters
    p0 = [error_offset, dtheta, dbeta]
    lower = [-100, -1, -1]
    upper = [100, 1, 1]

    # wrapper to fix cff, n and lines_per_mm
    def fit_fun(hv, error_offset, dtheta, dbeta):
        return pgm_calibration(hv, error_offset, dtheta, dbeta, cff=cff, k=k,
                               lines_per_mm=lines_per_mm)

    p, cov = curve_fit(fit_fun, hv, data, p0=p0, bounds=(lower, upper))

    return p, cov


# +-----------------+ #
# | Gap analysis    | # =======================================================
# +-----------------+ #


def dec_fermi_div(edc: np.ndarray, erg: np.ndarray, res: float, Ef: float,
                  fd_cutoff: float, T: float = 5) -> np.ndarray:
    """
    Divide measured EDC by Fermi-Dirac function.
    
    :param edc: energy distribution curve, intensities
    :param erg: energy axis
    :param res: instrumental resolution
    :param Ef: Fermi energy
    :param fd_cutoff: energy cut off 
    :param T: temperature [K]
    :return: divided energy distribution function
    """
    
    # deconvolve resolution
    fd = fermi_dirac(erg, Ef, T=T)
    fd = normalize(ndimage.gaussian_filter(fd, res))
    co_idx = indexof(fd_cutoff, erg)
    edc = normalize(edc)
    edc[:co_idx] = edc[:co_idx] / fd[:co_idx]
    return edc


def deconvolve_resolution(edc: np.ndarray, energy: np.ndarray, 
                          resolution: float) -> tuple:
    """
    Deconvolve instrumental resolution from EDC by dividing the 
    Fourier transforms.
    
    :param edc: energy distribution curve, intensities
    :param energy: ebergy axis
    :param resolution: instrumental resolution
    :return: results in format:

        - ``res[0]`` - :class:`np.ndarray`; deconvolved profile,
        - ``res[1]`` - :class:`np.ndarray`; resolution mask
    """
    
    sigma = resolution / (2 * np.sqrt(2 * np.log(2)))
    de = get_step(energy)
    res_mask = sig.gaussian(edc.size, sigma / de)
    deconv = np.abs(ifftshift(ifft(fft(edc) / fft(res_mask))))
    # move first element at the end. stupid, but works
    tmp = deconv[0]
    deconv[:-1] = deconv[1:]
    deconv[-1] = tmp
    return deconv, res_mask


def detect_step(signal: np.ndarray, n_box: int = 15, n_smooth: int = 3) -> int:
    """
    Detect the biggest, clearest step in a signal by smoothing it and looking
    at the maximum of the first derivative.

    :param signal: signal to detect step in
    :param n_box: box size for smoothing prior to calculating derivative
    :param n_smooth: number of smoothing runs prior to calculating derivative
    :return: index at which step occurs
    """

    smoothened = smooth(signal, n_box=n_box, recursion_level=n_smooth)
    grad = np.gradient(smoothened)
    step_index = np.argmax(np.abs(grad))
    return step_index


def fermi_dirac(E: np.ndarray, mu: float = 0, T: float = 4.2) -> np.ndarray:
    r"""
    Fermi-Dirac distribution with chemical potential *mu* at temperature *T*
    for energy *E*. The Fermi Dirac distribution is given by:

    .. math::
        n(E) = \frac{1}{\exp{\big(\frac{E - \mu}{k_B T}\big)} + 1}

    :param E: energies
    :param mu: chemical potential
    :param T: temperature
    :return: Fermi-Dirac distribution
    """

    kT = const.k_B * T / const.eV
    res = 1 / (np.exp((E - mu) / kT) + 1)
    return res


def fermi_fit_func(E: np.ndarray, E_F: float, sigma: float, a0: float,
                   b0: float, a1: float, b1: float, T: float = 5) \
        -> np.ndarray:
    """
    Fermi Dirac distribution with an additional linear inelastic background
    and convoluted with a Gaussian for the instrument resolution.

    :param E: energy values [eV]
    :param E_F: position of Fermi energy [eV]
    :param sigma: experimental resolution in units of *E* step size
    :param a0: slope of the linear background below the *E_F*
    :param b0: offset of the linear background below *E_F*.
    :param a1: slope of the linear background above the E_F
    :param b1: offset of the linear background above *E_F*.
    :param T: temperature [K]
    :return: Fermi-Dirac distribution at given temperature
    """

    # Basic Fermi Dirac distribution at given T
    E = E
    y = fermi_dirac(E, E_F, T)
    dE = np.abs(E[0] - E[1])

    # Add a linear contribution to the 'above' and 'below-E_F' part
    y += (a0 * E + b0) * step_function(E, step_x=E_F, flip=True)  # below part
    y += (a1 * E + b1) * step_function(E, step_x=E_F + dE)  # above part

    # Convolve with instrument resolution
    if sigma > 0:
        y = ndimage.gaussian_filter(y, sigma)  # , mode='nearest')

    return y


def find_mid_old(xdata: np.ndarray, ydata: np.ndarray, xrange: list = None) \
        -> list:
    """
    Find step point at the middle of step intensity 
    
    :param xdata: `x` scale
    :param ydata: `y` scale
    :param xrange: list of points to crop the `x` scale
    :return: coordinated of the middle point
    """

    if xrange is None:
        x0, x1 = indexof(-0.1, xdata), indexof(0.1, xdata)
    else:
        x0, x1 = indexof(xrange[0], xdata), indexof(xrange[1], xdata)
    ydata = smooth(ydata, recursion_level=3)
    y_mid = 0.5 * (ydata[x0:x1].max() - ydata[x0:x1].min())
    x_mid = xdata[x0:x1][indexof(y_mid, ydata[x0:x1])]
    return [x_mid, y_mid]


def fit_fermi_dirac(energies: np.ndarray, edc: np.ndarray, e_0: float,
                    T: float = 5, sigma0: float = 10, a0: float = 0,
                    b0: float = -0.1, a1: float = 0, b1: float = -0.1) \
        -> tuple:
    """
    Fit Fermi-Dirac distribution convoluted by a Gaussian (simulating the
    instrument resolution) plus a linear components to a given energy
    distribution curve.

    :param energies: energy values [eV]
    :param edc: energy distribution curve (intensities)
    :param e_0: starting guess for the Fermi energy [eV]
    :param T: temperature [K]
    :param sigma0: experimental resolution in units of *E* step size
    :param a0: slope of the linear background below the *E_F*
    :param b0: offset of the linear background below *E_F*.
    :param a1: slope of the linear background above the E_F
    :param b1: offset of the linear background above *E_F*.
    :return: results of the fitting in format:

        - ``res[0]`` - :class:`np.ndarray`; fitted_profiles,
        - ``res[1]`` - :py:obj:`callable`; Fermi-Dirac function,
        - ``res[2]`` - :class:`np.ndarray`; parameters' covariance matrix,
        - ``res[3]`` - :py:obj:`float`; fitted instrumental resolution,
        - ``res[4]`` - :py:obj:`float`; fitted instrumental resolution's error
    """

    # Initial guess and bounds for parameters
    p0 = [e_0, sigma0, a0, b0, a1, b1]
    de = 1
    lower = [e_0 - de, 0, -100, -10, -10, -20]
    upper = [e_0 + de, 100, 100, 10, 10, 20]

    def fit_func(E, E_F, sigma, a0, b0, a1, b1):
        """ Wrapper around fermi_fit_func that fixes T. """
        return fermi_fit_func(E, E_F, sigma, a0, b0, a1, b1, T=T)

    # Carry out the fit
    p, cov = curve_fit(fit_func, energies, edc, p0=p0, bounds=(lower, upper))
    resolution = 2 * np.sqrt(2 * np.log(2)) * p[1] * \
                 np.abs(energies[1] - energies[0])
    resolution_err = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(cov[1][1]) * \
                     np.abs(energies[1] - energies[0])

    res_func = lambda x: fit_func(x, *p)
    return p, res_func, cov, resolution, resolution_err


def step_function(x: np.ndarray, step_x: float = 0, flip: bool = False) \
        -> np.ndarray:
    """
    np.ufunc wrapper for step_function_core. Confer corresponding
    documentation.

    :param x: arguments
    :param step_x: `x` value of the step
    :param flip: if :py:obj:`True`, flip function values around the step
    :return: step profile
    """

    res = \
        np.frompyfunc(lambda x: step_function_core(x, step_x, flip), 1, 1)(x)
    return res.astype(float)


def step_function_core(x: np.ndarray, x0: float = 0, flip: bool = False) \
        -> np.ndarray:
    r"""
    Basic step function, with step occuring at x0. Gives values:

    .. math::
                        0  \textrm{  } \textrm{    for  } \textrm{  } x < x_0

                f(x) =  0.5  \textrm{  } \textrm{  for  } \textrm{  } x = x_0

                        1  \textrm{  } \textrm{    for  } \textrm{  } x > x_0

    :param x: arguments
    :param x0: `x` value of the step
    :param flip: if :py:obj:`True`, flip function values around the step
    :return: step profile
    """

    sign = -1 if flip else 1
    if sign * x < sign * x0:
        result = 0
    elif x == x0:
        result = 0.5
    elif sign * x > sign * x0:
        result = 1
    return result


def symmetrize_edc(data: np.ndarray, energies: np.ndarray) -> tuple:
    """
    Symmetrize EDC around the Fermi level, sum intensities in the overlapping 
    region.
     
    :param data: energy distribution function, intensities
    :param energies: energy axis (must be in binding)
    :return: result in a format:

        - ``res[0]`` - :class:`np.ndarray`; symmetrized energy distribution
          curve
        - ``res[1]`` - :class:`np.ndarray`; symmetrized energies
    """
    
    Ef_idx = indexof(0, energies)
    sym_edc = np.zeros((2 * Ef_idx))
    sym_energies = np.zeros_like(sym_edc)
    dE = np.abs(energies[0] - energies[1])
    sym_edc[:data.size] = data
    sym_energies[:Ef_idx] = energies[:Ef_idx]
    for i in range(sym_energies.size):
        sym_energies[i] = energies[0] + i * dE
        if 0 < i < data.size:
            sym_edc[-i] += data[i]

    return sym_edc, sym_energies


def symmetrize_edc_around_Ef(data: np.ndarray, energies: np.ndarray) -> tuple:
    """
    Symmetrize EDC around the Fermi level, disregard signal above it.
     
    :param data: energy distribution function, intensities
    :param energies: energy axis (must be in binding)
    :return: result in a format:

        - ``res[0]`` - :class:`np.ndarray`; symmetrized energy distribution
          curve
        - ``res[1]`` - :class:`np.ndarray`; symmetrized energies
    """
    
    Ef_idx = indexof(0, energies)
    sym_edc = np.hstack((data[:Ef_idx], np.flip(data[:Ef_idx])))
    sym_energies = np.hstack((energies[:Ef_idx], np.flip(-energies[:Ef_idx])))

    return sym_edc, sym_energies


def sum_edcs_around_k(data: np.ndarray, kx: np.ndarray, ky: np.ndarray,
                      ik: int = 0) -> np.ndarray:
    """
    Sum EDCs around a given k-point, in a box specified by ``ik``.
    The box is binning over ``ik`` in all directions.

    :param data: dataset
    :param kx: `kx` coordinate of the k-point
    :param ky: `ky` coordinate of the k-point
    :param ik: box size of summation
    :return: summed EDC
    """

    dkx = np.abs(data.xscale[0] - data.xscale[1])
    min_kx = indexof(kx - ik * dkx, data.xscale)
    max_kx = indexof(kx + ik * dkx, data.xscale)
    min_ky = indexof(ky - ik * dkx, data.yscale)
    max_ky = indexof(ky + ik * dkx, data.yscale)

    edc = np.sum(np.sum(data.data[min_kx:max_kx, min_ky:max_ky, :], axis=0),
                 axis=0)
    return edc


# +--------------+ #
# | Utilities    | # ==========================================================
# +--------------+ #


def indexof(value: float, array: np.ndarray) -> int:
    """
    Find first index at which given value occurs in searched array.

    :param value: value
    :param array: array hosting value
    :return: index of occurrence
    """

    return np.argmin(np.abs(array - value))


def get_step(axis: np.ndarray) -> float:
    """
    Get step of the axis. Returns difference between first and second value.

    :param axis: axis of uniformly (**!**) distributed values
    :return: step value
    """

    return np.abs(axis[0] - axis[1])


def smooth(x: np.ndarray, n_box: int = 5, recursion_level: int = 1) \
        -> np.ndarray:
    """
    Implement a linear midpoint smoother: Move an imaginary 'box' of size
    ``n_box`` over the data points `x` and replace every point with the mean
    value of the box centered at that point.
    Can be called recursively to apply the smoothing `n` times in a row
    by setting ``recursion_level`` to `n`.

    At the endpoints, the arrays are assumed to continue by repeating their
    value at the start/end as to minimize endpoint effects. I.e. the array
    [1,1,2,3,5,8,13] becomes [1,1,1,1,2,3,5,8,13,13,13] for a box with
    ``n_box`` = 5.

    :param x: data to smooth
    :param n_box: size of the smoothing box (i.e. number of points
                  around the central point over which to take the mean).
                  Should be an odd number - otherwise the next lower odd
                  number is taken
    :param recursion_level: the number of times the smoothing is applied
    :return: smoothed data points of same shape as input
    """

    # Ensure odd n_box
    if n_box % 2 == 0:
        n_box -= 1

    # Make the box. Sum(box) should equal 1 to keep the normalization (?)
    box = np.ones(n_box) / n_box

    # Append some points to reduce endpoint effects
    n_append = int(0.5 * (n_box - 1))
    left = [x[0]]
    right = [x[-1]]
    y = np.array(n_append * left + list(x) + n_append * right)

    # Let numpy do the work
    smoothened = np.convolve(y, box, mode='valid')

    # Do it again (enter next recursion level) or return the result
    if recursion_level == 1:
        return smoothened
    else:
        return smooth(smoothened, n_box, recursion_level - 1)


def smooth_2d(x: np.ndarray, n_box: int = 5, recursion_level: int = 1) \
        -> np.ndarray:
    """
    Smooth 2-dimensional dataset, uses the same principal as in :func:`smooth`
    with a box of size (``n_box`` x ``n_box``).

    :param x: data set to smooth
    :param n_box: size of the square (``n_box`` x ``n_box``) smoothing box
    :param recursion_level: the number of times the smoothing is applied
    :return: smoothed data points of same shape as input
    """

    if n_box % 2 == 0:
        n_box -= 1

    # Make the box. Sum(box) should equal 1 to keep the normalization (?)
    box = np.ones((n_box, n_box)) / n_box ** 2

    # Append some points to reduce endpoint effects
    n_append = int(0.5 * (n_box - 1))
    y = np.ones((2 * n_append + x.shape[0], 2 * n_append + x.shape[1]))
    y[n_append:-n_append, n_append:-n_append] = x
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if (0 < i < y.shape[0]) and (j < n_append):
                if i < n_append:
                    y[i, j] = x[0, 0]
                elif i >= (y.shape[0] - n_append):
                    y[i, j] = x[-1, 0]
                else:
                    y[i, j] = x[i - n_append, 0]
            elif (i < n_append) and (n_append < j < (x.shape[1] + n_append)):
                y[i, j] = x[0, j - n_append]
            elif (0 < i < y.shape[0]) and (j < (n_append + x.shape[1])):
                if i < n_append:
                    y[i, j] = x[0, -1]
                elif i >= (y.shape[0] - n_append):
                    y[i, j] = x[-1, -1]
                else:
                    y[i, j] = x[i - n_append, -1]
            elif (i > x.shape[0] + n_append) and \
                    (n_append < j < (x.shape[1] + n_append - 1)):
                y[i, j] = x[-1, j - n_append]
    y[n_append:-n_append, n_append:-n_append] = x

    # Let numpy do the work
    smoothened = convolve2d(y, box, mode='valid')
    # smoothened = _smooth_2d(y, box)

    # Do it again (enter next recursion level) or return the result
    if recursion_level == 1:
        return smoothened
    else:
        return smooth_2d(smoothened, n_box, recursion_level - 1)


def normalize(data: np.ndarray, axis: int = 2) -> np.ndarray:
    """
    Normalize data along the given axis. Recognizes different dimensions.

    :param data: data set to normalize
    :param axis: index of axis to normalize along
    :return: normalized data set
    """

    if len(data.shape) == 1:
        if data.max() == 0:
            normalized = 0
        else:
            normalized = data / data.max()
    elif len(data.shape) == 2:
        normalized = np.zeros_like(data)
        if axis == 0:
            for i in range(data.shape[axis]):
                normalized[i, :] = normalize(data[i, :])
        elif axis == 1:
            for i in range(data.shape[axis]):
                normalized[:, i] = normalize(data[:, i])
    elif len(data.shape) == 3:
        normalized = np.zeros_like(data)
        if axis == 2:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    normalized[i, j, :] = normalize(data[i, j, :])
        elif axis == 1:
            for i in range(data.shape[0]):
                for j in range(data.shape[2]):
                    normalized[i, :, j] = normalize(data[i, :, j])
        elif axis == 0:
            for i in range(data.shape[1]):
                for j in range(data.shape[2]):
                    normalized[:, i, j] = normalize(data[:, i, j])
    else:
        print('Not implemented dimensions')
        return
    return normalized


def normalize_to_sum(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Normalize data sets to the sum of the intensity along the specified
    direction.

    :param data: data set to normalize
    :param axis: index of axis to normalize along
    :return: normalized data set
    """

    norm_data = np.zeros_like(data)
    if axis == 0:
        for i in range(data.shape[axis]):
            if np.sum(data[i, :, :]) == 0:
                norm_data[i, :, :] = data[i, :, :]
            else:
                norm_data[i, :, :] = data[i, :, :] / np.sum(data[i, :, :])
    elif axis == 1:
        for i in range(data.shape[axis]):
            if np.sum(data[:, i, :]) == 0:
                norm_data[:, i, :] = data[:, i, :]
            else:
                norm_data[:, i, :] = data[:, i, :] / np.sum(data[:, i, :])
    elif axis == 2:
        for i in range(data.shape[axis]):
            if np.sum(data[:, :, i]) == 0:
                norm_data[:, :, i] = data[i, :, :]
            else:
                norm_data[:, :, i] = data[:, :, i] / np.sum(data[:, :, i])
    return norm_data


def rotate(matrix: np.ndarray, alpha: float, deg: bool = True) -> np.ndarray:
    """
    Apply rotation on the matrix of coordinate's system.

    :param matrix: matrix of coordinates
    :param alpha: rotation angle
    :param deg: if :py:obj:`True`, rotation angle is in degrees
    :return: rotated matrix of coordinates
    """

    matrix = np.array(matrix)
    if deg:
        alpha = np.deg2rad(alpha)

    # specify rotation matrix
    r = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha),  np.cos(alpha)]])

    if len(matrix.shape) == 1:
        return r @ matrix
    elif len(matrix.shape) == 2:
        rotated = np.zeros_like(matrix)
        for vi in range(matrix.shape[0]):
            rotated[vi] = r @ matrix[vi]
        return rotated
    else:
        print('matrix size not supported.')
        return


def rotate_vector(x_coords: Union[float, np.ndarray],
                  y_coords: Union[float, np.ndarray], alpha: float,
                  deg: bool = True) -> tuple:
    res_x, res_y = np.zeros_like(x_coords), np.zeros_like(y_coords)
    for idx, (xi, yi) in enumerate(zip(x_coords, y_coords)):
        tmp_rotated = rotate(np.array([xi, yi]), alpha, deg=deg)
        res_x[idx] = tmp_rotated[0]
        res_y[idx] = tmp_rotated[1]
    return res_x, res_y


def shift_k_coordinates(kx: np.ndarray, ky: np.ndarray, qx: float = 0,
                        qy: float = 0) -> tuple:
    """
    Shift k-space coordinate system by a wavevector q = [``q_x``, ``q_y``].

    :param kx: initial `kx` axis (1D or :class:`np.meshgrid`)
    :param ky: initial `ky` axis (1D or :class:`np.meshgrid`)
    :param qx: wavevector shift along the `kx` direction
    :param qy: wavevector shift along the `ky` direction
    :return: (``kx``, ``ky``) shifted coordinate axes (1D or
             :class:`np.meshgrid`)
    """

    kxx = kx + np.ones_like(kx) * qx
    kyy = ky + np.ones_like(ky) * qy
    return kxx, kyy


def sum_XPS(data: list, crop: list = None, plot: bool = False) -> tuple:
    """
    Sum XPS spectra from separate scans.

    :param data: list of :class:`np.ndarray` to sum. ``len(data)`` corresponds
                 to number of separate scans to be summed
    :param crop: list [`E_i`, `E_f`] of energy values to crop between
    :param plot: if :py:obj:`True`, plot results
    :return: result in a format:

        - ``res[0]`` - :class:`np.ndarray`; summed spectrum
        - ``res[1]`` - :class:`np.ndarray`; energy scale
    """

    n = len(data)
    spectra = [np.sum(data_i.data[0, :, :], axis=0) for data_i in data]

    size_check = np.array([spectra_i.size for spectra_i in spectra])

    if all_equal(size_check):
        spectrum = np.sum(np.array(spectra), axis=0)
        energy = data[0].zscale
    else:
        spectrum = np.zeros((size_check.min()))
        narrow = np.where(size_check == min(size_check))[0][0]
        min_ergs = [data_i.zscale.min() for data_i in data]
        max_ergs = [data_i.zscale.max() for data_i in data]
        if narrow.size == 1:
            cond1 = np.all([data[narrow].zscale.min() >= min_ergs_i for
                            min_ergs_i in min_ergs])
            cond2 = np.all([data[narrow].zscale.max() >= max_ergs_i for
                            max_ergs_i in max_ergs])
            if cond1 and cond2:
                energy = data[narrow].zscale
                e0 = energy[0]
                for idx, sp in enumerate(spectra):
                    e_idx = indexof(e0, data[idx].zscale)
                    spectrum += sp[e_idx:(e_idx + energy.size)]

    if crop is not None:
        ei, ef = indexof(crop[0], energy), indexof(crop[1], energy)
        energy = energy[ei:ef]
        spectrum = spectrum[ei:ef]

    if plot:
        fig, axs = plt.subplots(n+1, 1, figsize=[10, 7])
        for idx in range(len(axs) - 1):
            axs[idx].plot(data[idx].zscale, spectra[idx])
            axs[idx].set_title(f'data{idx}')
        axs[-1].plot(energy, spectrum)
        axs[-1].set_title('summed spectra')

    return spectrum, energy


# +---------------------+ #
# | Image processing    | # ===================================================
# +---------------------+ #


def curvature_1d(data: np.ndarray, dx: float, a0: float = 0.0005,
                 nb: int = None, rl: int = None, xaxis: np.ndarray = None,
                 clip_co: float = 0.1) -> np.ndarray:
    """
    Calculate 1D curvature profile, similar to second derivative.

    .. note::
        Implements method described by `Zhang et al.
        <http://dx.doi.org/10.1063/1.3585113>`_

    :param data: original data, intensities
    :param dx: step in the units of experiment
    :param a0: free parameter
    :param nb: size of the smoothing box, for smoothing prior to calculating
               curvature profile
    :param rl: the number of times the smoothing is applied
    :param xaxis: energy axis. If given, zero values above the Fermi level
    :param clip_co: cutoff value of the curvature profile; useful to improve
                    clarity of the image
    :return: curvature profile
    """

    if (nb is not None) and (rl is not None):
        data = smooth(data, n_box=nb, recursion_level=rl)
    df = np.gradient(data, dx)
    d2f = np.gradient(df, dx)

    df_max = np.max(np.abs(df) ** 2)
    C_x = d2f / (np.sqrt(a0 + ((df / df_max) ** 2)) ** 3)
    if xaxis is not None:
        ef = indexof(0, xaxis)
        C_x[ef:] = 0

    # return -C_x
    # return -normalize(C_x)
    return -normalize(C_x).clip(max=clip_co)
    # return -np.where(C_x <= clip_co, clip_co, C_x)


def curvature_2d(data: np.ndarray, dx: float, dy: float, a0: float = 100,
                 nb: int = None, rl: int = None, eaxis: np.ndarray = None) \
        -> np.ndarray:
    """
    Calculate 2D curvature profile, similar to second derivative. Procedure is
    an image enhancement method to highlight small variations in a measured
    signal.

    .. note::
        Implements method described by `Zhang et al.
        <http://dx.doi.org/10.1063/1.3585113>`_

    :param data: original data, intensities image
    :param dx: step (along the 1st dimension) in the units of experiment
    :param dy: step (along the 2nd dimension) in the units of experiment
    :param a0: free parameter
    :param nb: size of the smoothing box (nb x nb), for smoothing prior to
               calculating curvature profile
    :param rl: the number of times the smoothing is applied
    :param eaxis: energy axis. If given, zero values above the Fermi level
    :return: curvature image
    """

    if (nb is not None) and (rl is not None):
        data = smooth_2d(data, n_box=nb, recursion_level=rl)

    dfdx, dfdy = np.gradient(data, dx, dy)
    d2fdx2 = np.gradient(dfdx, dx, axis=0)
    d2fdy2 = np.gradient(dfdy, dy, axis=1)
    d2fdxdy = np.gradient(dfdx, dy, axis=1)

    cx = a0 * (dx ** 2)
    cy = a0 * (dy ** 2)

    # mdfdx, mdfdy = np.max(np.abs(dfdx)), np.max(np.abs(dfdy))
    # cy = (dy / dx) * (mdfdx ** 2 + mdfdy ** 2) * a0
    # cx = (dx / dy) * (mdfdx ** 2 + mdfdy ** 2) * a0

    nom = (1 + cx * np.power(dfdx, 2)) * cy * d2fdy2 - \
          2 * cx * cy * dfdx * dfdy * d2fdxdy + \
          (1 + cy * np.power(dfdy, 2)) * cx * d2fdx2
    den = (1 + cx * np.power(dfdx, 2) + cy * np.power(dfdy, 2)) ** 1.5

    C_xy = nom / den

    if eaxis is not None:
        ef = indexof(0, eaxis)
        C_xy[ef:, :] = 0

    return np.abs(C_xy)


def find_gamma(FS: np.ndarray, x0: int, y0: int, method: str = 'Nelder-Mead',
               print_output: bool = False) -> scipy.optimize.OptimizeResult:
    """
    Find Gamma-point (center of rotation) of an image.

    .. note::
        Implements method described by Junck et al.
        (PubMed ID: `2362201 <https://pubmed.ncbi.nlm.nih.gov/2362201/>`_)

    :param FS: Fermi surface, 2D array of numbers
    :param x0: initial guess of center `x` coordinate
    :param y0: initial guess of center `y` coordinate
    :param method: minimization method, see :meth:`scipy.minimize` for mode
                   details
    :param print_output: if :py:obj:`True`, print results as a
                         :class:`pandas.DataFrame`
    :return: result of the minimization routine
    """

    # do the minimization and find optimized values
    res = minimize(rotate_around_xy, x0=[x0, y0], args=(FS,), method=method)

    # for printing results in nice table
    if print_output:
        if res.success:
            status = 'succes'
        else:
            status = 'failed'
        labels = ['status', 'r value', 'niter', 'x0', 'y0']
        entries = [status, '{:.4f}'.format(-res.fun), res.nit,
                   '{:.2f}'.format(res.x[0]), '{:.2f}'.format(res.x[1])]

        d = {'param': labels,
             'value': entries}
        df = pd.DataFrame(data=d)
        pd.set_option('display.max_columns', 50)
        print(df)

    return res


def imgs_corr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate the correlation coefficient between two matrices.

    .. note::
        For reference, see Junck et al. (PubMed ID:
        `2362201 <https://pubmed.ncbi.nlm.nih.gov/2362201/>`_)

    :param img1: original image
    :param img2: rotated image
    :return: correlation coefficient, parameter to minimize
    """

    # calculate averages
    avr_img1 = np.mean(img1)
    avr_img2 = np.mean(img2)

    # calculate rest of the coefficient
    num = 0.
    den1 = 0.
    den2 = 0.
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            num += (img1[i][j] - avr_img1) * (img2[i][j] - avr_img2)
            den1 += (img1[i][j] - avr_img1) ** 2
            den2 += (img2[i][j] - avr_img2) ** 2

    return num / np.sqrt(den1 * den2)


def rotate_around_xy(init_guess: list, data_org: Any) -> object:
    """
    Rotate matrix by 180 deg around (``x0``, ``y0``) and return only
    overlapping region.

    :param init_guess: [``x0``, ``y0``] list of center of rotation coordinates
    :param data_org: original full image (Fermi surface)
    :return: correlation coefficient between original and rotated images.
             See :func:`imgs_corr` for mode details
    """

    # get coordinates values
    x0 = init_guess[0]
    y0 = init_guess[1]

    # transpose if necessary and get dimensions
    transposed = False
    if data_org.shape[0] > data_org.shape[1]:
        data_org = data_org.T
        transposed = True
    x_org = data_org.shape[0]
    y_org = data_org.shape[1]

    # get lengths (half of them) of the overlapped cuts
    if x0 <= x_org // 2:
        x = x0
    else:
        x = x_org - x0

    if y0 <= y_org // 2:
        y = y0
    else:
        y = y_org - y0

    # specify boundaries
    x_start = int(np.max([0, x0 - x]))
    x_stop = int(np.min([x + x0, x_org]))
    y_start = int(np.max([0, y0 - y]))
    y_stop = int(np.min([y + y0, y_org]))

    data = data_org[x_start:x_stop, y_start:y_stop]

    # calculate correlation coefficient (minus, because we actually want to
    # maximize it)
    if transposed:
        return -imgs_corr(data.T, np.flip(data).T)
    else:
        return -imgs_corr(data, np.flip(data))


# +-----------------------+ #
# | K-space conversion    | # =================================================
# +-----------------------+ #


def k_fac(energy: Union[float, np.ndarray], Eb: float = 0, hv: float = 0,
          work_func: float = 4.5) -> float:
    """
    Calculate scaling factor for `k`-space conversion depending on kinetic
    energy of photoemitted electrons.

    :param energy: energy value [eV]
    :param Eb: binding energy [eV]
    :param hv: photon energy [eV]
    :param work_func: work function [eV]
    :return: scaling factor
    """

    me = const.m_e / const.eV / 1e20
    hbar = const.hbar_eV
    return np.sqrt(2 * me * (energy - Eb + hv - work_func)) / hbar


def angle2kspace(scan_ax: np.ndarray, ana_ax: np.ndarray,
                 d_scan_ax: float = 0, d_ana_ax: float = 0,
                 orientation: str = 'horizontal', a: float = np.pi,
                 energy: np.ndarray = np.array([0]),
                 **kwargs: dict) -> tuple:
    """
    Convert experimental axes o `k`-space.

    .. note::
        Follows procedure described by
        `Ishida et al. <https://doi.org/10.1063/1.5007226>`_

    :param scan_ax: angle axis along the scanned direction
    :param ana_ax: angle axis along the analyzer direction
    :param d_scan_ax: angle offset along the scanned direction
    :param d_ana_ax: angle offset along the analyzer direction
    :param orientation: orientation of the analyzer slit. Can be '`horizontal`'
                        or '`vertical`'
    :param a: in-plane lattice constant [Angstrom]. When given, convert
              momentum axis to a/2pi units
    :param energy: energy axis
    :param kwargs: kwargs for `k`-factor. See :func:`k_fac` for more details
    :return: result in a format:

        - ``res[0]`` - :class:`np.ndarray`; meshgrid of `kx` coordinates
        - ``res[1]`` - :class:`np.ndarray`; meshgrid of `ky`/energy
            coordinates. int = 1 for single momentum axis
    """

    # Angle to radian conversion and setting offsets
    scan_ax, ana_ax, energy = np.array(scan_ax), \
                               np.array(ana_ax), \
                               np.array(energy)
    d_scan_ax, d_ana_ax = np.deg2rad(d_scan_ax), np.deg2rad(d_ana_ax)
    scan_ax = np.deg2rad(scan_ax) - d_scan_ax
    ana_ax = np.deg2rad(ana_ax) - d_ana_ax

    nkx, nky, ne = scan_ax.size, ana_ax.size, energy.size

    # single momentum axis for specified binding energy
    if (nkx == 1) and (ne == 1):
        ky = np.zeros(nky)
        k0 = k_fac(energy, **kwargs)
        k0 *= (a / np.pi)
        if orientation == 'horizontal':
            ky = np.sin(ana_ax)
        elif orientation == 'vertical':
            ky = np.sin(ana_ax - d_ana_ax) * np.cos(d_ana_ax) + \
                 np.cos(ana_ax - d_ana_ax) * np.cos(d_scan_ax) * \
                 np.sin(d_ana_ax)
        return k0 * ky, 1

    # momentum vs energy, e.g. for band maps
    elif (nkx == 1) and (ne != 1):
        ky = np.zeros((ne, nky))
        erg = np.zeros_like(ky)
        if orientation == 'horizontal':
            for ei in range(ne):
                k0i = k_fac(energy[ei], **kwargs)
                k0i *= (a / np.pi)
                ky[ei] = k0i * np.sin(ana_ax)
                erg[ei] = energy[ei] * np.ones(nky)
        elif orientation == 'vertical':
            for ei in range(ne):
                k0i = k_fac(energy[ei], **kwargs)
                k0i *= (a / np.pi)
                ky[ei] = k0i * np.sin(ana_ax - d_ana_ax) * \
                         np.cos(d_ana_ax) + np.cos(ana_ax - d_ana_ax) * \
                         np.cos(d_scan_ax) * np.sin(d_ana_ax)
                erg[ei] = energy[ei] * np.ones(nky)
        return ky, erg

    # momentum vs momentum, e.g. for constant energy FSs
    elif (nkx != 1) and (ne == 1):
        kx = np.zeros((nkx, nky))
        ky = np.zeros_like(kx)
        k0 = k_fac(energy, **kwargs)
        k0 *= (a / np.pi)
        if orientation == 'horizontal':
            for kxi in range(nkx):
                # kx[kxi] = np.ones(nky) * np.sin(scan_ax[kxi])
                # ky[kxi] = np.cos(scan_ax[kxi]) * np.sin(ana_ax)
                kx[kxi] = np.sin(scan_ax[kxi]) * np.cos(ana_ax)
                ky[kxi] = np.sin(ana_ax)
        elif orientation == 'vertical':
            for kxi in range(nkx):
                # kx[kxi] = np.cos(ana_ax) * np.sin(scan_ax[kxi])
                # ky[kxi] = np.sin(ana_ax)
                kx[kxi] = np.cos(ana_ax - d_ana_ax) * np.sin(scan_ax[kxi])
                ky[kxi] = np.sin(ana_ax - d_ana_ax) * np.cos(d_ana_ax) + \
                          np.cos(ana_ax - d_ana_ax) * np.cos(d_scan_ax) * \
                          np.sin(d_ana_ax)
        return k0 * kx, k0 * ky

    # 3D set of momentum vs momentum coordinates, for all given
    # binding energies
    elif (nkx != 1) and (ne != 1):
        kx = np.zeros((ne, nkx, nky))
        ky = np.zeros_like(kx)
        for ei in range(ne):
            k0i = k_fac(energy[ei], **kwargs)
            k0i *= (a / np.pi)
            if orientation == 'horizontal':
                for kxi in range(nkx):
                    kx[ei, kxi, :] = k0i * np.sin(scan_ax[kxi]) * \
                                     np.cos(ana_ax)
                    ky[ei, kxi, :] = k0i * np.sin(ana_ax)
            elif orientation == 'vertical':
                for kxi in range(nkx):
                    kx[ei, kxi, :] = k0i * np.cos(ana_ax - d_ana_ax) * \
                                     np.sin(scan_ax[kxi])
                    ky[ei, kxi, :] = k0i * np.sin(ana_ax - d_ana_ax) * \
                                     np.cos(d_ana_ax) + \
                                     np.cos(ana_ax - d_ana_ax) * \
                                     np.cos(d_scan_ax) * np.sin(d_ana_ax)
        return kx, ky


def hv2kz(ang: np.ndarray, hvs: np.ndarray, work_func: float = 4.5,
          V0: float = 0, trans_kz: bool = False, c: float = np.pi,
          energy: np.ndarray = np.array([0]), **kwargs: dict) -> tuple:
    """
    Convert photon energy scan to `k`-space.

    .. note::
        Follows procedure described by
        `Ishida et al. <https://doi.org/10.1063/1.5007226>`_

    :param ang: angle axis along the analyzer direction
    :param hvs: photon energies
    :param work_func: work function [eV]
    :param V0: tin potential [eV]
    :param trans_kz: if :py:obj:`True`, transform photon energies to momentum
                     units
    :param c: out-of-plane lattice constant [Angstrom].
              When given, convert momentum axis to c/2pi units
    :param energy: energy axis
    :param kwargs: kwargs. See :func:`k_fac` and :func:`angle2kspace` for more
                   details
    :return: result in a format:

        - ``res[0]`` - :class:`np.ndarray`; meshgrid of `ky` coordinates
        - ``res[1]`` - :class:`np.ndarray`; meshgrid of `kz` coordinates
    """

    ang, hvs, energy = np.array(ang), np.array(hvs), np.array(energy)
    ky = []
    for hv in hvs:
        kyi, _ = angle2kspace(np.array([1]), ang, hv=hv, energy=energy,
                              **kwargs)
        ky.append(kyi)

    if 'd_ana_ax' in kwargs.keys():
        ana_ax_off = kwargs['d_ana_ax']
    else:
        ana_ax_off = 0

    ky = np.array(ky)
    kz = np.zeros_like(ky)
    me = const.m_e / const.eV / 1e20
    hbar = const.hbar_eV
    k0 = np.sqrt(2 * me) / hbar
    k0 *= (c / (1 * np.pi))
    ang = ang - ana_ax_off

    if trans_kz:
        for kz_i in range(kz.shape[0]):
            Ek = hvs[kz_i] + work_func
            for kz_j in range(kz.shape[1]):
                theta = np.deg2rad(ang[kz_j])
                k_ij = k0 * np.sqrt(Ek * (np.cos(theta) ** 2) + V0)
                kz[kz_i][kz_j] = k_ij
    else:
        for kzi, hv in enumerate(hvs):
            kz[kzi, :] = np.ones_like(ang) * hv

    return ky, kz


@njit(parallel=True)
def rescale_data(data: np.ndarray, org_scale: np.ndarray,
                 new_scale: np.ndarray) -> np.ndarray:
    """
    Rescales dataset, to make `k`-space converted image from photon energy scan
    possible to plot on rectangular lattice. That is required to plot data
    as a regular image instead of :class:`~matplotlib.pyplot.pcolormesh`
    object, which is incredibly slow.

    Method changes axes to span between lowest and highest values (specified
    by axis at the highest photon energy), with the smallest step (specified by
    axis at the lowest photon energy).

    :param data: original data
    :param org_scale: original `k`-space converted axis
    :param new_scale: new scale with fine step, over widest momentum range
    :return: rescaled data
    """

    new_data = np.zeros((data.shape[0], new_scale.size, data.shape[2]))
    for zi in prange(data.shape[2]):
        for xi in range(data.shape[0]):
            y_min, y_max = org_scale[xi].min(), org_scale[xi].max()
            for yi_idx, yi in enumerate(new_scale):
                y_org_idx = np.argmin(np.abs(org_scale[xi] - yi))
                if (yi >= y_min) and (yi <= y_max):
                    new_data[xi, yi_idx, zi] = data[xi, y_org_idx, zi]
                else:
                    continue
    return new_data


# +----------+ #
# | Misc     | # ==============================================================
# +----------+ #


def all_equal(iterable: Union[list, tuple]) -> bool:
    """
    Check if all elements in iterable are equal.

    :param iterable: list to check
    :return: logic test result
    """

    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def dynes_formula(omega: np.ndarray, n0: float, gamma: float, delta: float) \
        -> np.ndarray:
    r"""
    Dynes formula for tunneling density of states of superconductor,
    given by the formula:

    .. math::
        N(\omega) = N_0 \textrm{Re}\Bigg[
        \frac{\omega + i\Gamma}{\sqrt{(\omega + i\Gamma)^2 - \Delta^2}}\Bigg]

    :param omega: energy axis
    :param n0: normal-state density of states
    :param gamma: parameter quantifying for the pair-breaking processes
    :param delta: superconducting gap
    :return: superconductor's density of states
    """

    num = omega + gamma * 1.j
    n_omega = num / np.sqrt((num ** 2) + delta ** 2)
    return n0 * n_omega.real


def McMillan_Tc(omega_D: float = 1, lam: float = 1, mu: float = 1) -> float:
    r"""
    McMillan's formula for superconducting Tc:

    .. math::
        T_c = \frac{\omega_D}{1.45}\exp{\bigg(
        \frac{1.04(1 + \lambda)}{\lambda - \mu(1 + 0.62\lambda)}}\bigg)

    :param omega_D: Debay frequency (temperature) of the material
    :param lam: electron-boson coupling constant, lambda
    :param mu: Coulomb pseudopotential
    :return: superconducting critical temperature (Tc)
    """

    frac = 1.04 * (1 + lam) / (lam - mu * (1 + 0.62 * lam))
    Tc = (omega_D / 1.45) * np.exp(-frac)
    return Tc

