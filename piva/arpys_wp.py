# moje funkcje do obróbki danych, a przynajmniej taki jest zamysł

import math
from itertools import groupby

# from scipy import optimize as opt
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.signal as sig
from matplotlib import pyplot as plt
from matplotlib import colors, patches, interactive
from numpy.linalg import norm   # for shriley bckgrd
from scipy.special import voigt_profile
from scipy import ndimage
from scipy.optimize import curve_fit, minimize, fsolve
from scipy.signal import convolve2d
from scipy.optimize import OptimizeWarning
from scipy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift

import piva.data_loader as dl
import piva.my_constants as const

# +-------------------------------------+ #
# | Data fitting functions and routines | # =========================
# +-------------------------------------+ #

def gaussian(x, a=1, mu=0, sigma=1):
    """
    Implement a Gaussian bell curve f(x) given by the expression::

                                         2
                          1    / (x-mu) \
        f(x) = a * exp( - - * ( -------  ) )
                          2    \ sigma  /

    **Parameters**

    =====  =====================================================================
    x      array; variable at which to evaluate f(x)
    a      float; amplitude (maximum value of curve)
    mu     float; mean of curve (location of maximum)
    sigma  float; standard deviation (`width` of the curve)
    =====  =====================================================================

    **Returns**

    ===  =======================================================================
    res  array containing the value of the Gaussian at every point of input x
    ===  =======================================================================
    """
    return a * np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)


def lorentzian(x, a0, mu, gamma, resol=0):
    r"""
    .. math::
        a_0 \cdot \frac{\Gamma^2}{(x-x_\mathrm{max})^2 + \Gamma^2}

    **Parameters**

    =====  =====================================================================
    x      float or array; the variable.
    a0     float; normalization factor.
    xmax   float; peak position of the Lorentzian in *x* units.
    gamma  float; peak width in *x* units.
    =====  =====================================================================
    """
    # a0, xmax, gamma = pars[0], pars[1], pars[2]
    y = a0 * (1 / (2 * np.pi)) * gamma / ((x - mu) ** 2 + (.5 * gamma) ** 2)
    return ndimage.gaussian_filter(y, resol)


def dynes_formula(omega, n0, gamma, delta):
    num = omega + gamma * 1.j
    n_omega = num / np.sqrt((num ** 2) + delta ** 2)
    return n0 * n_omega.real


def asym_lorentzian(x, a0, mu, gamma, alpha=0, resol=0):
    if resol == 0:
        gamma *= 2 / (1 + np.exp(alpha * (x - mu)))
        return a0 * (1 / (2 * np.pi)) * gamma / ((x - mu) ** 2 + (.5 * gamma) ** 2)
    else:
        gamma *= 2 / (1 + np.exp(alpha * (x - mu)))
        y = a0 * (1 / (2 * np.pi)) * gamma / ((x - mu) ** 2 + (.5 * gamma) ** 2)
        return ndimage.gaussian_filter(y, sigma=resol)


def two_lorentzians(x, a0, mu0, gamma0, a1, mu1, gamma1):
    return lorentzian(x, a0, mu0, gamma0) + lorentzian(x, a1, mu1, gamma1)


def three_lorentzians(x, a0, mu0, gamma0, a1, mu1, gamma1, a2, mu2, gamma2):
    return lorentzian(x, a0, mu0, gamma0) + lorentzian(x, a1, mu1, gamma1) + lorentzian(x, a2, mu2, gamma2)


def four_lorentzians(x, a, mu, gamma):
    return lorentzian(x, a[0], mu[0], gamma[0]) + lorentzian(x, a[1], mu[1], gamma[1]) + \
           lorentzian(x, a[2], mu[2], gamma[2]) + lorentzian(x, a[3], mu[3], gamma[3])


def lorentzian_dublet(x, *p, delta=1, line='f'):
    """

    :param line:
    :param x:
    :param p:   [a0, mu, gamma, a1]
    :param delta:
    :return:
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
    return lorentzian(x, p[0], p[1], p[2]) + lorentzian(x, p[0] * coeff, p[1] - delta, p[2])


def fit_n_dublets(data, x, a0, mu, gamma, delta, constr=None, fit_delta=False, line='f'):
    pars = []
    pcovs = []
    errs = []
    pi = []
    for i in range(len(mu)):
        # pi.append([a0[i], mu[i], gamma[i], a1[i]])
        pi.append(a0[i])
        pi.append(mu[i])
        pi.append(gamma[i])

    if fit_delta:
        def fit_func(x, a0, mu, gamma, delta):
            return lorentzian_dublet(x, a0, mu, gamma, delta)
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
        d = {'a0': normalize(np.array(as0)),
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


def shirley_calculate(x, y, tol=1e-5, maxit=10):
    """ S = specs.shirley_calculate(x,y, tol=1e-5, maxit=10)
    Calculate the best auto-Shirley background S for a dataset (x,y). Finds the biggest peak
    and then uses the minimum value either side of this peak as the terminal points of the
    Shirley background.
    The tolerance sets the convergence criterion, maxit sets the maximum number
    of iterations.

    from:       https://github.com/kaneod/physics/blob/master/python/specs.py
    """

    # Make sure we've been passed arrays and not lists.
    x = np.array(x)
    y = np.array(y)

    # Sanity check: Do we actually have data to process here?
    if not (x.any() and y.any()):
        print("specs.shirley_calculate: One of the arrays x or y is empty. Returning zero background.")
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
        print("specs.shirley_calculate: Boundaries too high for algorithm: returning a zero background.")
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
                ysum += (x[j] - x[j + 1]) * 0.5 * (y[j] +
                                                   y[j + 1] - 2 * yr - B[j] - B[j + 1])
            Bnew[i] = k * ysum
        # If Bnew is close to B, exit.
        if norm(Bnew - B) < tol:
            B = Bnew.copy()
            break
        else:
            B = Bnew.copy()
        it += 1

    if it >= maxit:
        print("specs.shirley_calculate: Max iterations exceeded before convergence.")
    if is_reversed:
        return (yr + B)[::-1]
    else:
        return yr + B


def fit_bgr(bgr_poly_order, xdata, ydata, fit_range, peak_range):
    """ Function for fitting a curve with polynomial background of given order.

        **Parameters**
        ============================================================================
        bgr_poly_order      int; order of polynomial to fit background
        xdata               1D array;
        ydata               1D array;
        fit_range           2D vector; range for fitting background, in units of
                            'xdata'/'ydata' array indexes
        peak_range          2D vector; range for fitting peak, in units of
                            'xdata'/'ydata' array indexes
        ============================================================================

        **Returns**
        ============================================================================
        [fit_x, bgr_fit, bgr_peak, coefs_bgr]

        fit_x           1D array; range of 'xdata' determined by 'fit_range'
        bgr_fit         1D array; profile of fitted background within 'fit_range'
        bgr_peak        1D array; profile of fitted background within 'peak_range'
        coefs_bgr       1D array; coefficients in fitted polynomial
        ============================================================================
        """

    peak_x = xdata[peak_range[0]:peak_range[1]]
    fit_x = xdata[fit_range[0]:fit_range[1]]
    bgr_x = np.concatenate((xdata[fit_range[0]:peak_range[0]], xdata[peak_range[1]:fit_range[1]]), axis=0)
    bgr_y = np.concatenate((ydata[fit_range[0]:peak_range[0]], ydata[peak_range[1]:fit_range[1]]), axis=0)
    coefs_bgr = np.polyfit(bgr_x, bgr_y, bgr_poly_order)
    bgr_fit = np.poly1d(coefs_bgr)(fit_x)
    bgr_peak = np.poly1d(coefs_bgr)(peak_x)
    return [fit_x, bgr_fit, bgr_peak, coefs_bgr]


def fit_multiple_with_bgr(fun, bgr_poly_order, data, k_axis, erg_axis, data_range, k_limits, peak_limits, step=None,
                          n=5, smoothing=None,  plotting=False, y_offset=5, x_offset=0, p0=None):
    """
        Function for fitting stack of MDCs within a given cut, with polynomial
        background of given order. For description check also 'fit_multiple' and
        'lorentzian_with_poly_bgr'
    :param fun:                 callable; function to fit peak
    :param bgr_poly_order:      int; order of background polynomial
    :param data:                np.array; 2D data array
    :param k_axis:              np.array; 1D angle/momentum axis
    :param erg_axis:            np.array; 1D energy axis
    :param data_range:          np.array; 2-element array of data range to fit, in units of array indices
    :param k_limits:            list; [x_coords, y_coords] of two points to fit linear limit of peaks along energy
    :param peak_limits:         list; [x_coords, y_coords] of two points to fit linear limit of peaks along energy
    :param step:                int; defines step in energy between fitted MDCs, default=None
    :param n:                   int; defines number of MDCs to fit between 'data_range'; default=5
    :param smoothing:           boolean; smoothing
    :param plotting:            boolean; plot MDCs with fitted curves; default=False
    :param y_offset:            float; Only for plotting. space in energy between plotted MDCs; default=5
    :param x_offset:            float; Only for plotting. Offset in momentum between plotted MDCs; default=0
    :param p0:                  list; initial parameters for fit
    :return:
    """
    # stuff for storing results
    pars = []
    pcovs = []
    errs = []
    cuts = []
    fitted_profiles = []
    bgrs = []
    # taking inputs into more pleasent format
    data_range = [min(data_range), max(data_range)]

    # get fit range as a region between two linear bounds
    k_llim = get_linear(k_limits[0])
    k_ulim = get_linear(k_limits[1])
    peak_llim = get_linear(peak_limits[0])
    peak_ulim = get_linear(peak_limits[1])

    # cuts_to_fit = dl.start_step_n()
    if step is None:
        step = int(math.floor((data_range[1] - data_range[0]) / n))
    cuts_to_fit = np.arange(data_range[0], data_range[1], step)  # idxs of fitted MDCs in 'data'
    # print(cuts_to_fit)

    # do multiple fitting
    for idx in cuts_to_fit:
        ki = int(k_llim(idx))
        kf = int(k_ulim(idx))
        pi = int(peak_llim(idx))
        pf = int(peak_ulim(idx))
        cut = data[idx, ki:kf]
        if smoothing is not None:
            cut = smooth(cut, smoothing[0], smoothing[1])
        bgr_x = np.concatenate((k_axis[ki:pi], k_axis[pf:kf]), axis=0)
        bgr_y = np.concatenate((data[idx, ki:pi], data[idx, pf:kf]), axis=0)
        coefs_bgr = np.polyfit(bgr_x, bgr_y, bgr_poly_order)
        bgr = np.poly1d(coefs_bgr)(k_axis[ki:kf])
        if len(pars) == 0:
            if p0 is None:
                popt, pcov = curve_fit(fun, k_axis[ki:kf], cut - bgr)
            else:
                popt, pcov = curve_fit(fun, k_axis[ki:kf], cut - bgr, p0=p0)
        else:
            popt, pcov = curve_fit(fun, k_axis[ki:kf], cut - bgr, p0=pars[-1])

        # calculate errors and append fit results
        pars.append(popt)
        pcovs.append(pcov)
        errs.append(np.sqrt(np.diag(pcov)))
        # if plotting, save cuts
        if plotting:
            if smoothing is not None:
                cuts.append(smooth(data[idx, :], smoothing[0], smoothing[1]))
            else:
                cuts.append(data[idx, :])
            if len(pars[0]) == 3:
                fitted_profiles.append(fun(k_axis[ki:kf], popt[0], popt[1], popt[2]))
            elif len(pars[0]) == 4:
                fitted_profiles.append(fun(k_axis[ki:kf], popt[0], popt[1], popt[2], popt[3]))
            bgrs.append(bgr)

    if plotting:
        plt.figure("stack of mdcs", figsize=[15, 10])
        for i in range(len(cuts_to_fit)):
            idx = cuts_to_fit[i]
            ki = int(k_llim(idx))
            kf = int(k_ulim(idx))
            lbl = 'E = {:.3f} meV ({})'.format(erg_axis[cuts_to_fit[i]] * 1000, cuts_to_fit[i])
            plt.plot(k_axis + i * x_offset, cuts[i] + i * y_offset, label=lbl)
            plt.plot(k_axis[ki:kf] + i * x_offset, (fitted_profiles[i] + bgrs[i]) + i * y_offset)
            plt.plot(k_axis[ki:kf] + i * x_offset, bgrs[i] + i * y_offset)
            plt.legend()

    # create panda dataframe with results
    energies = []
    indices = []
    k_0s = []
    dk_0s = []
    gammas = []
    dgammas = []
    if len(pars[0]) == 3:
        for i in range(len(cuts_to_fit)):
            indices.append(cuts_to_fit[i])
            energies.append(erg_axis[cuts_to_fit[i]] * 1000)
            k_0s.append(pars[i][1])
            dk_0s.append(errs[i][1])
            gammas.append(pars[i][2])
            dgammas.append(errs[i][2])
        d = {'energy index': indices,
             'energy, meV': energies,
             'k_0, deg/A^-1': k_0s,
             'dk_0, deg/A^-1': dk_0s,
             'Gamma, deg/A^-1': gammas,
             'dGamma, deg/A^-1': dgammas
             }
    elif len(pars[0]) == 4:
        sigmas = []
        dsigmas = []
        for i in range(len(cuts_to_fit)):
            indices.append(cuts_to_fit[i])
            energies.append(erg_axis[cuts_to_fit[i]] * 1000)
            k_0s.append(pars[i][1])
            dk_0s.append(errs[i][1])
            gammas.append(pars[i][2])
            dgammas.append(errs[i][2])
            sigmas.append(pars[i][3])
            dsigmas.append(errs[i][3])
        d = {'energy index': indices,
             'energy, meV': energies,
             'k_0, deg/A^-1': k_0s,
             'dk_0, deg/A^-1': dk_0s,
             'Gamma, deg/A^-1': gammas,
             'dGamma, deg/A^-1': dgammas,
             'Sigma, deg/A^-1': sigmas,
             'dSigma, deg/A^-1': dsigmas
             }
    df = pd.DataFrame(data=d)

    return [cuts_to_fit, pars, errs, pcovs, df]


def fit_multiple(fun, data, xdata, data_range, upper_limits, lower_limits, step=100, n=5, plotting=False, y_offset=500,
                 x_offset=0):
    """ Function for fitting stack of MDCs within a given cut. With additional option for plotting.

        **Parameters**
        ============================================================================
        fun             function used for plotting. At this point works only for
                        3-parameter functions
        data            2D array; cut from which MDCs will be taken
        xdata           1D array; angle/momentum scale
        data_range      2D vector; range of 'data' (energy) to be fitted;
                        in units of array idx
        xdata_range     2D vector; range of 'xdata' (momentum) to be fitted;
                        in units of array idx
        step            int; defines step in energy between fitted MDCs, default=None
        n               int; defines number of MDCs to fit between 'data_range',
                        default=5
        plotting        boolean; plot MDCs with fitted curves; default=False
        y_offset        float; Only for plotting. space in energy between plotted MDCs;
                        default=5
        x_offset        float; Only for plotting. Offset in momentum between
                        plotted MDCs; default=0
        ============================================================================

        **Returns**
        ============================================================================
        [cuts_to_fit, pars, errs, pcovs]    2D array (4xn), n being
                                                number of fitted MDCs, where:
        cuts_to_fit     1D array (len=n); fitted MDC (along whole angle range)
        pars            1D array (len=n); parameters of fitted curves
        errs            1D array (len=n): errors of pars
        pcovs           1D array (len=n): estimated covariance of pars

        ============================================================================
        """
    # stuff for storing results
    pars = []
    pcovs = []
    errs = []
    if plotting:
        cuts = []
        fitted_profiles = []
    # taking inputs into more pleasent fromat
    data_range = [min(data_range), max(data_range)]
    # xi = xdata_range[0]
    # xf = xdata_range[1]

    # get fit range as a region between two linear bounds
    llim = get_linear(lower_limits)
    ulim = get_linear((upper_limits))

    if step is None:
        step = (data_range[1] - data_range[0]) // n
    cuts_to_fit = np.arange(data_range[0], data_range[1], step)
    # print(cuts_to_fit)
    # print(len(pars))

    # do multiple fitting
    for idx in cuts_to_fit:
        xi = int(llim(idx))
        xf = int(ulim(idx))
        cut = data[idx, xi:xf]
        if len(pars) == 0:
            popt, pcov = curve_fit(fun, xdata[xi:xf], cut)
        else:
            popt, pcov = curve_fit(fun, xdata[xi:xf], cut, p0=pars[-1])

        # calculate errors and append fit results
        pars.append(popt)
        pcovs.append(pcov)
        errs.append(np.sqrt(np.diag(pcov)))
        # if plotting, save cuts
        if plotting:
            cuts.append(data[idx, :])
            fitted_profiles.append(fun(xdata[xi:xf], popt[0], popt[1], popt[2]))

    if plotting:
        plt.figure("stack of mdcs", figsize=[10, 7])
        for i in range(len(cuts_to_fit)):
            idx = cuts_to_fit[i]
            xi = int(llim(idx))
            xf = int(ulim(idx))
            lbl = f"mdc at {cuts_to_fit[i]}"
            plt.plot(xdata + i * x_offset, cuts[i] + i * y_offset, label=lbl)
            plt.plot(xdata[xi:xf] + i * x_offset, fitted_profiles[i] + i * y_offset)
            plt.legend()

    return [cuts_to_fit, pars, errs, pcovs]


def lorentzian_with_poly_bgr(peak_fun, bgr_poly_order, xdata, ydata, fit_range, peak_range):
    """ Function for fitting a curve with polynomial background of given order.

        **Parameters**
        ============================================================================
        peak_fun            function for peak fitting. NOTE: right now only
                            3-parameter functions
        bgr_poly_order      int; order of polynomial to fit background
        xdata               1D array;
        ydata               1D array;
        fit_range           2D vector; range for fitting background, in units of
                            'xdata'/'ydata' array indexes
        peak_range          2D vector; range for fitting peak, in units of
                            'xdata'/'ydata' array indexes
        ============================================================================

        **Returns**
        ============================================================================
        [fit+bgr, np.poly1d(coefs_bgr)(fit_x), popt_peak, pcov_peak, coefs_bgr]

        fit+bgr                         1D array; profile of fitted peak with added background
        np.poly1d(coefs_bgr)(fit_x)     1D array; profile of fitted background
        popt_peak                       1D array; parameters of fitted curve
        err_peak                        1D array; errors of popt_peak
        pcov_peak                       1D array; estimated covariance of popt_peak
        coefs_bgr                       1D array; coefficients in fitted polynomial
        ============================================================================
        """

    peak_x = xdata[peak_range[0]:peak_range[1]]
    peak_y = ydata[peak_range[0]:peak_range[1]]
    fit_x = xdata[fit_range[0]:fit_range[1]]
    bgr_x = np.concatenate((xdata[fit_range[0]:peak_range[0]], xdata[peak_range[1]:fit_range[1]]), axis=0)
    bgr_y = np.concatenate((ydata[fit_range[0]:peak_range[0]], ydata[peak_range[1]:fit_range[1]]), axis=0)
    coefs_bgr = np.polyfit(bgr_x, bgr_y, bgr_poly_order)
    bgr = np.poly1d(coefs_bgr)(peak_x)
    popt_peak, pcov_peak = curve_fit(peak_fun, peak_x, peak_y - bgr)
    err_peak = np.sqrt(np.diag(pcov_peak))
    fit = peak_fun(fit_x, popt_peak[0], popt_peak[1], popt_peak[2])
    bgr = np.poly1d(coefs_bgr)(fit_x)
    return [fit + bgr, bgr, popt_peak, err_peak, pcov_peak, coefs_bgr]


def get_linear(points):
    x = points[0]
    y = points[1]

    pars = np.polyfit(x, y, 1)

    fun = lambda arg: pars[0] * arg + pars[1]

    return fun


def exp(x, a, k):
    return a * np.exp(k * x)


def voigt(x, a, mu, gamma, sigma):
    """
        Return the Voigt line shape at x with Lorentzian component FWHM gamma
        and Gaussian component FWHM sigma.

    """

    # return a * np.real(wofz((x - mu + 1j * gamma) / sigma / np.sqrt(2))) / sigma / np.sqrt(2 * np.pi)
    args = x - mu
    return a * voigt_profile(args, sigma, gamma)


def print_fit_results(p, cov, labels=None):
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


def subtract_bg_shirley(data, dim=0, profile=False, normindex=0):
    """ Use an iterative approach for the background of an EDC as described in
    DOI:10.1103/PhysRevB.5.4709. Mathematically, the value of the EDC after
    BG subtraction for energy E EDC'(E) can be expressed as follows::

                               E1
                               /
        EDC'(E) = EDC(E) - s * | EDC(e) de
                               /
                               E

    where EDC(E) is the value of the EDC at E before bg subtraction, E1 is a
    chosen energy value (in our case the last value in the EDC) up to which
    the subtraction is applied and s is chosen such that EDC'(E0)=EDC'(E1)
    with E0 being the starting value of the bg subtraction (in our case the
    first value in the EDC).

    In principle, this is an iterative method, so it should be applied
    repeatedly, until no appreciable change occurs through an iteration. In
    practice this convergence is reached in 4-5 iterations at most and even a
    single iteration may suffice.

    **Parameters**

    =======  ===================================================================
    data     np.array; input data with shape (m x n) or (1 x m x n) containing
             an E(k) cut
    dim      int; either 0 or 1. Determines whether the input is aranged as
             E(k) (n EDCs of length m, dim=0) or k(E) (m EDCs of length n,
             dim=1)
    profile  boolean; if True, a list of the background values for each MDC
             is returned additionally.
    =======  ===================================================================

    **Returns**

    =======  ===================================================================
    data     np.array; has the same dimensions as the input array.
    profile  1D-array; only returned as a tuple with data (`data, profile`)
             if argument `profile` was set to True. Contains the
             background profile, i.e. the background value for each MDC.
    =======  ===================================================================
    """
    # Prevent original data from being overwritten by retaining a copy
    data = data.copy()

    data, d, m, n = convert_data(data)

    if dim == 0:
        nk = n
        ne = m
        get_edc = lambda k: data[0, :, k]
    elif dim == 1:
        nk = m
        ne = n
        get_edc = lambda k: data[0, k]

    # Take shirley bg from the angle-averaged EDC
    average_edc = np.mean(data[0], dim + 1)

    # Calculate the "normalization" prefactor
    s = np.abs(average_edc[normindex] - average_edc[-1]) / average_edc.sum()

    # Prepare a function that sums the EDC from a given index upwards
    sum_from = np.frompyfunc(lambda e: average_edc[e:].sum(), 1, 1)
    indices = np.arange(ne)
    bg = s * sum_from(indices).astype(float)

    # Subtract the bg profile from each EDC
    for k in range(nk):
        edc = get_edc(k)
        # Update data in-place
        edc -= bg

    data = convert_data_back(data, d, m, n)
    if profile:
        return data, bg
    else:
        return data


def fit_TB_model(TB_fun, t, erg, kx, ky, coords_transf=None, model='li2018'):

    if coords_transf is not None:
        kx, ky = coords_transf(kx, ky)
    kdata = np.vstack((kx, ky))

    # wrapper around model
    def fit_fun(k, *t):
        return TB_fun(k, *t, model=model)

    popt, pcov = curve_fit(fit_fun, kdata, erg, p0=t)
    return popt


def TB_FS(TB_fun, k0, *t, model='li2018'):

    # kdata = np.vstack((kx, ky))

    def fun(kx, ky, *t):
        return TB_fun(kx, ky, *t, model=model)

    res = fsolve(fun, k0, args=t)

    return res


def shift_k_coordinates(kx, ky, qx=0, qy=0):
    kxx = kx + np.ones_like(kx) * qx
    kyy = ky + np.ones_like(ky) * qy
    return kxx, kyy


def fit_mdc(k_ax, mdc, fit_range, p0, bgr_fit):

    range0 = indexof(fit_range[0], k_ax)
    range1 = indexof(fit_range[1], k_ax)
    k_fit = k_ax[range0:range1]
    try:
        mdc_fit = mdc[range0:range1] - bgr_fit
    except AttributeError:
        mdc_fit = mdc[range0:range1]
        bgr_fit = np.zeros_like(mdc_fit)
    except TypeError:
        mdc_fit = mdc[range0:range1]
        bgr_fit = np.zeros_like(mdc_fit)
    except ValueError:
        return

    a, mu, gamma, alpha, resol = p0
    if alpha == 1:
        fit_alpha = False
    else:
        fit_alpha = True
    if resol == 0:
        fit_resol = False
    else:
        fit_resol = True
    fit_fun, p0 = set_fit_fun(p0, fit_alpha, fit_resol)

    try:
        p, cov = curve_fit(fit_fun, k_fit, mdc_fit, p0=p0)
    except RuntimeWarning:
        return None
    except OptimizeWarning:
        return None

    # prepare_fitting_results(fit_alpha, fit_beta)
    res_func = lambda x: fit_fun(x, *p)
    fit = res_func(k_fit) + bgr_fit
    return fit, p


def fit_mdc_bgr(k_ax, mdc, fit_range, bgr_range, poly_order):
    bgr0, bgr1 = fit_range[0], bgr_range[0]
    bgr2, bgr3 = bgr_range[1], fit_range[1]
    bgr0, bgr1 = indexof(bgr0, k_ax), indexof(bgr1, k_ax)
    bgr2, bgr3 = indexof(bgr2, k_ax), indexof(bgr3, k_ax)

    bgr_k = np.hstack((k_ax[bgr0:bgr1], k_ax[bgr2:bgr3]))
    bgr_mdc = np.hstack((mdc[bgr0:bgr1], mdc[bgr2:bgr3]))
    coefs_bgr = np.polyfit(bgr_k, bgr_mdc, poly_order)

    fit_k = k_ax[bgr0:bgr3]
    return np.poly1d(coefs_bgr)(fit_k), fit_k


def set_fit_fun(p00, fit_alpha, fit_resol):
    p0 = [p00[0], p00[1], p00[2]]
    if fit_alpha and fit_resol:
        fit_fun = lambda x, a0, mu, gamma, alpha, resol: asym_lorentzian(x, a0, mu, gamma, alpha=alpha, resol=resol)
        p0.append(p00[3])
        p0.append(p00[4])
    elif fit_alpha and not fit_resol:
        fit_fun = lambda x, a0, mu, gamma, alpha: asym_lorentzian(x, a0, mu, gamma, alpha=alpha)
        p0.append(p00[3])
    elif not fit_alpha and fit_resol:
        fit_fun = lambda x, a0, mu, gamma, resol: asym_lorentzian(x, a0, mu, gamma, resol=resol)
        p0.append(p00[4])
    else:
        fit_fun = lambda x, a0, mu, gamma: asym_lorentzian(x, a0, mu, gamma)
    return fit_fun, p0


# +--------------------------------------------------+ #
# | Resolution fitting functions and PGM calibration | # =========================
# +--------------------------------------------------+ #

def step_function_core(x, step_x=0, flip=False):
    """ Implement a perfect step function f(x) with step at `step_x`::

                / 0   if x < step_x
                |
        f(x) = {  0.5 if x = step_x
                |
                \ 1   if x > step_x

    **Parameters**

    ======  ====================================================================
    x       array; x domain of function
    step_x  float; position of the step
    flip    boolean; Flip the > and < signs in the definition
    ======  ====================================================================
    """
    sign = -1 if flip else 1
    if sign * x < sign * step_x:
        result = 0
    elif x == step_x:
        result = 0.5
    elif sign * x > sign * step_x:
        result = 1
    return result


def step_function(x, step_x=0, flip=False):
    """ np.ufunc wrapper for step_function_core. Confer corresponding
    documentation.
    """
    res = \
        np.frompyfunc(lambda x: step_function_core(x, step_x, flip), 1, 1)(x)
    return res.astype(float)


def step_ufunc(x, step_x=0, flip=False):
    """ np.ufunc wrapper for :func:`step_core
    <arpys.postprocessing.step_core>`. Confer corresponding documentation.
    """
    res = \
        np.frompyfunc(lambda x: step_core(x, step_x, flip), 1, 1)(x)
    return res.astype(float)


def step_core(x, step_x=0, flip=False):
    """ Implement a step function f(x) with step at `step_x`::

                / 0 if x < step_x
        f(x) = {
                \ 1 if x >= step_x

    :See also:
        :func:`step_function_core <arpys.postprocessing.step_function_core>`.
    """
    sign = -1 if flip else 1
    if sign * x < sign * step_x:
        result = 0
    elif sign * x >= sign * step_x:
        result = 1
    return result


def detect_step(signal, n_box=15, n_smooth=3):
    """ Try to detect the biggest, clearest step in a signal by smoothing
    it and looking at the maximum of the first derivative.
    """
    smoothened = smooth(signal, n_box=n_box, recursion_level=n_smooth)
    grad = np.gradient(smoothened)
    step_index = np.argmax(np.abs(grad))
    return step_index


def fermi_fit_func(E, E_F, sigma, a0, b0, a1, b1, T=5):
    """ Fermi Dirac distribution with an additional linear inelastic
    background and convoluted with a Gaussian for the instrument resolution.

    **Parameters**

    =====  =====================================================================
    E      1d-array; energy values in eV
    E_F    float; Fermi energy in eV
    sigma  float; instrument resolution in units of the energy step size in *E*.
    a      float; slope of the linear background.
    b      float; offset of the linear background at *E_F*.
    T      float; temperature.
    =====  =====================================================================
    """
    # Basic Fermi Dirac distribution at given T
    E = E
    y = fermi_dirac(E, E_F, T)
    dE = np.abs(E[0] - E[1])

    # Add a linear contribution to the 'above' and 'below-E_F' part
    y += (a0 * E + b0) * step_function(E, step_x=E_F, flip=True)  # below part
    y += (a1 * E + b1) * step_function(E, step_x=E_F+dE)  # above part

    # Convolve with instrument resolution
    if sigma > 0:
        y = ndimage.gaussian_filter(y, sigma)  # , mode='nearest')

    return y


def fit_fermi_dirac(energies, edc, e_0, T=5, sigma0=10, a0=0, b0=-0.1, a1=0, b1=-0.1):
    """ Try fitting a Fermi Dirac distribution convoluted by a Gaussian
    (simulating the instrument resolution) plus a linear component on the
    side with E<E_F to a given energy distribution curve.

    **Parameters**

    ========  ==================================================================
    energies  1D array of float; energy values.
    edc       1D array of float; corresponding intensity counts.
    e_0       float; starting guess for the Fermi energy. The fitting
              procedure is quite sensitive to this.
    T         float; (fixed) temperature.
    sigma0    float; starting guess for the standard deviation of the
              Gaussian in units of pixels (i.e. the step size in *energies*).
    a0        float; starting guess for the slope of the linear component.
    b0        float; starting guess for the linear offset.
    ========  ==================================================================

    **Returns**

    ========  ==================================================================
    p         list of float; contains the fit results for [E_F, sigma, a, b].
                NOTE: sigma in convoluted gaussian is converted to FWHM (since
                that corresponds to the actual resolution).
    res_func  callable; the fit function with the optimized parameters. With
              this you can just do res_func(E) to get the value of the
              Fermi-Dirac distribution at energy E.
    ========  ==================================================================
    """
    # Normalize the EDC to interval [0, 1]
    # edc = normalize(edc)

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
    resolution = 2 * np.sqrt(2 * np.log(2)) * p[1] * np.abs(energies[1] - energies[0])
    resolution_err = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(cov[1][1]) * np.abs(energies[1] - energies[0])

    res_func = lambda x: fit_func(x, *p)
    return p, res_func, cov, resolution, resolution_err


def fermi_dirac(E, mu=0, T=4.2):
    """ Return the Fermi Dirac distribution with chemical potential *mu* at
    temperature *T* for energy *E*. The Fermi Dirac distribution is given by::

                         1
        n(E) = ----------------------
                exp((E-mu)/(kT)) + 1

    and assumes values from 0 to 1.
    """
    kT = const.k_B * T / const.eV
    res = 1 / (np.exp((E - mu) / kT) + 1)
    return res


def fit_binned(data, energies, angles, nbinned=10, T=5, units=1000, excluded=[], plot=False, fname=None):
    """
    Bin EDCs from 'binned' channels, fit FD to each bin and fit constant to obtained resolution
    :param data:            np.array; intensities
    :param energies:        np.array; energy values
    :param angles:          np.array; angles along the detector
    :param nbinned:         int; number of EDCs to put in single bin
    :param T:               float; temperature for FD distribution (units: K)
    :param units:           float; change the units of energy (resolution and its error); defaul: from eV to meV
    :param excluded:        list; indices of bins to be excluded from fitting
    :param plot:            bool; ploting option
    :return:
            resolutions:        list; fitted resolutions from each bin
            ang:                list; angles along the detector indicating centers of each bin
            fitted_resol:       float; constant (resolution) fitted to all binns (units: meV)
            fitted_err:         float; error of 'fitted_resol' (units: meV)
            resolutions_err:    list; errors of 'resolutions'
            params              np.array; all fitted FD parameters from all bins
    """
    if len(data.shape) == 3:
        data = data[0, :, :].T
    # else:
    #     data = data.T
    edcs = []
    ang = []
    params = []
    resolutions = []
    resolutions_err = []
    Efs = []

    # bin EDCs
    idx = 0
    stop = 0
    while (stop + nbinned) <= data.shape[1]:
        start = idx * nbinned
        stop = start + nbinned
        mid = int(start + 0.5 * (stop - start))
        edcs.append(normalize(np.sum(data[:, start:stop], axis=1)))
        ang.append(angles[mid])
        idx += 1

    # to take into account last, not "full" bin
    # start = idx * nbinned
    # mid = int(start + 0.5 * (len(data) - start))
    # edcs.append(normalize(np.sum(data[start:-1, :], axis=0)))
    # ang.append(angles[mid])
    # edcs = np.array(edcs)
    # ang = np.array(ang)

    # fit FD to each bin
    angs_to_remove = []
    for idx, edc in enumerate(edcs):
        Ef = energies[detect_step(edc)]
        if np.isnan(edc).any():
            print('There is nan in bin {}'.format(idx))
        elif idx in excluded:
            angs_to_remove.append(ang[idx])
        else:
            p, rf, cov, resol, resol_err = fit_fermi_dirac(energies, edc, Ef, T=T)
            params.append([p, rf, cov, resol, resol_err])
            resolutions.append(resol * units)
            resolutions_err.append(resol_err * units)
            Efs.append(p[0])
    ang = [ele for ele in ang if ele not in angs_to_remove]
    # print(len(ang))

    # fit final resolution to results from each bin
    p, cov = fit_constant(ang, resolutions)
    const_resol = p[0]
    const_err = np.sqrt(cov[0][0])

    # fit constant Ef to results from each bin
    p, cov = fit_constant(ang, Efs)
    const_Ef = p[0]
    const_Ef_err = np.sqrt(cov[0][0])

    # check

    # plot
    if plot:
        plt.figure(figsize=[10, 7])
        plt.plot(ang, resolutions)
        plt.plot([ang[0], ang[-1]], [const_resol, const_resol], c='k')
        txt_res = 'resolution = {:.4} meV'.format(const_resol)
        txt_err = 'error = {:.2} meV'.format(const_err)
        if fname is not None:
            plt.text(0, const_resol - 0.1 * const_resol, fname, fontsize=15)
        plt.text(0, const_resol - 0.2 * const_resol, txt_res, fontsize=15)
        plt.text(0, const_resol - 0.3 * const_resol, txt_err, fontsize=15)
        plt.xlabel('deg')
        plt.ylabel('Resolution, meV')
        plt.ylim([2, np.max(resolutions) + 0.5])

    return [resolutions, ang, const_resol, const_err, resolutions_err, const_Ef, const_Ef_err, params, np.array(ang), \
            edcs]


def fit_constant(x, y):
    """
    wrapper for fitting linear with slope a=0
    :param x:
    :param y:
    :return:
    """

    def fit_fun(x, b):
        return 0 * x + b

    p, cov = curve_fit(fit_fun, x, y)

    return p, cov


def PGM_calibration(hv, error_offset, dtheta, dbeta, cff=2.25, k=1, lines_per_mm=300):
    """
    Function for PGM motors calibration. Based on:
        Absolute Energy Calibration for Plane Grating Monochromators, Nucl. Instrum. Meth. A 467-468, 482-484 (2001)
    :param data:
    :param hv:
    :param error_offset:
    :param dtheta:
    :param dbeta:
    :param cff:
    :param n:
    :param lines_per_mm:
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
    alpha = np.arcsin((np.sqrt(cff ** 2 + alpha) - 1) * ((wl * N * k) / coeff))

    beta = -np.arccos(cff * np.cos(alpha))
    theta = 0.5 * (alpha - beta)

    # wavelength from grating equation, taking into account offsets
    actualEnergy = 2 * np.cos(theta + dt) * np.sin(theta + dt + beta + db) / (N * k)

    return const.convert_eV_nm(actualEnergy) - hv + error_offset


def fit_PGM_calibration(data, hv, error_offset=-0.06, dtheta=0.001, dbeta=-0.001, cff=2.25, k=1, lines_per_mm=300):
    # Initial guess and bounds for parameters
    p0 = [error_offset, dtheta, dbeta]
    lower = [-100, -1, -1]
    upper = [100, 1, 1]

    # wrapper to fix cff, n and lines_per_mm
    def fit_fun(hv, error_offset, dtheta, dbeta):
        return PGM_calibration(hv, error_offset, dtheta, dbeta, cff=cff, k=k, lines_per_mm=lines_per_mm)

    p, cov = curve_fit(fit_fun, hv, data, p0=p0, bounds=(lower, upper))

    return p, cov


# +-----------------+ #
# | Gap analysis    | # ==============================================================
# +-----------------+ #


def dec_fermi_div(edc, erg, res, Ef, fd_cutoff, T=5):
    # deconvolve resolution
    fd = fermi_dirac(erg, T=T)
    fd = normalize(ndimage.gaussian_filter(fd, res))
    co_idx = indexof(fd_cutoff, erg)
    edc = normalize(edc)
    edc[:co_idx] = edc[:co_idx] / fd[:co_idx]
    return edc


def deconvolve_resolution(data, energy, resolution, Ef=0):
    sigma = resolution / (2 * np.sqrt(2 * np.log(2)))
    de = get_step(energy)
    res_mask = sig.gaussian(data.size, sigma / de)
    deconv = np.abs(ifftshift(ifft(fft(data) / fft(res_mask))))
    # deconv, _ = sig.deconvolve(data, res_mask)
    # move first element at the end. stupid, but works
    tmp = deconv[0]
    deconv[:-1] = deconv[1:]
    deconv[-1] = tmp
    return deconv, res_mask


def find_mid_old(xdata, ydata, xrange=None):
    """ Find middle point of EDC between given energy range

        **Parameters**
        ============================================================================
        xdata               1D array; energy scale
        ydata               1D array; intensities
        xrange              2D vector; range between which to look for, in energy units
        ============================================================================

        **Returns**
        ============================================================================
        [x_mid, y_mid]      2D vector; point's coordinates
        ============================================================================
        """
    if xrange is None:
        x0, x1 = indexof(-0.1, xdata), indexof(0.1, xdata)
    else:
        x0, x1 = indexof(xrange[0], xdata), indexof(xrange[1], xdata)
    ydata = smooth(ydata, recursion_level=3)
    y_mid = 0.5 * (ydata[x0:x1].max() - ydata[x0:x1].min())
    x_mid = xdata[x0:x1][indexof(y_mid, ydata[x0:x1])]
    return [x_mid, y_mid]


def find_midpoint(data, xscale):
    deriv = np.gradient(data)
    mid_idx = indexof(deriv.min(), deriv)
    return xscale[mid_idx], data[mid_idx]


def symmetrize_edc(data, energies):
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


def symmetrize_edc_around_Ef(data, energies):
    Ef_idx = indexof(0, energies)
    # sym_edc = np.zeros((2 * Ef_idx))
    # sym_energies = np.zeros_like(sym_edc)
    # dE = np.abs(energies[0] - energies[1])
    sym_edc = np.hstack((data[:Ef_idx], np.flip(data[:Ef_idx])))
    sym_energies = np.hstack((energies[:Ef_idx], np.flip(-energies[:Ef_idx])))

    return sym_edc, sym_energies


# +--------------+ #
# | Utilities    | # ==============================================================
# +--------------+ #


def add_attr(data, name, value):
    data.__dict__.update({name: value})


def add_kinetic_factor(data):
    if hasattr(data, 'ekin'):
        return
    else:
        if hasattr(data, 'wf'):
            pass
        else:
            dl.update_namespace(data, ('wf', 4.464))
        e_kin = data.hv - data.wf
        dl.update_namespace(data, ('ekin', e_kin))


def indexof(value, array):
    """
    Return the first index of the value in the array closest to the given
    `value`.

    Example::

        >>> a = np.array([1, 0, 0, 2, 1])
        >>> indexof(0, a)
        1
        >>> indexof(0.9, a)
        0
    """
    return np.argmin(np.abs(array - value))


def get_step(data):
    return np.abs(data[0] - data[1])


def convert_data(data):
    """ Helper function to convert data to the right shape. """
    # Find out whether we have a (m x n) (d=2) or a (1 x m x n) (d=3) array
    shape = data.shape
    d = len(shape)

    # Convert to shape (1 x m x n)
    if d == 2:
        m = shape[0]
        n = shape[1]
        data = data.reshape(1, m, n)
    elif d == 3:
        m = shape[1]
        n = shape[2]
    else:
        raise ValueError('Could not bring data with shape {} into right form.'.format(shape))
    return data, d, m, n


def convert_data_back(data, d, m, n):
    """ Helper function to convert data back to the original shape which is
    determined by the values of d, m and n (outputs of :func:`convert_data
    <arpys.postprocessing.convert_data>`).
    """
    if d == 2:
        data = data.reshape(m, n)
    return data


def smooth(x, n_box=5, recursion_level=1):
    """ Implement a linear midpoint smoother: Move an imaginary 'box' of size
    'n_box' over the data points 'x' and replace every point with the mean
    value of the box centered at that point.
    Can be called recursively to apply the smoothing n times in a row
    by setting 'recursion_level' to n.

    At the endpoints, the arrays are assumed to continue by repeating their
    value at the start/end as to minimize endpoint effects. I.e. the array
    [1,1,2,3,5,8,13] becomes [1,1,1,1,2,3,5,8,13,13,13] for a box with
    n_box=5.

    **Parameters**

    ===============  ===========================================================
    x                1D array-like; the data to smooth
    n_box            int; size of the smoothing box (i.e. number of points
                     around the central point over which to take the mean).
                     Should be an odd number - otherwise the next lower odd
                     number is taken.
    recursion_level  int; equals the number of times the smoothing is applied.
    ===============  ===========================================================

    **Returns**

    ===  =======================================================================
    res  np.array; smoothed data points of same shape as input.
    ===  =======================================================================
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
    # smoothened = np.convolve(x, box, mode='same')
    smoothened = np.convolve(y, box, mode='valid')

    # Do it again (enter next recursion level) or return the result
    if recursion_level == 1:
        return smoothened
    else:
        return smooth(smoothened, n_box, recursion_level - 1)


def smooth_2d(x, n_box=5, recursion_level=1):
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
            elif (i > x.shape[0] + n_append) and (n_append < j < (x.shape[1] + n_append - 1)):
                y[i, j] = x[-1, j - n_append]
    y[n_append:-n_append, n_append:-n_append] = x

    # Let numpy do the work
    # smoothened = np.convolve(x, box, mode='same')
    smoothened = convolve2d(y, box, mode='valid')

    # Do it again (enter next recursion level) or return the result
    if recursion_level == 1:
        return smoothened
    else:
        return smooth_2d(smoothened, n_box, recursion_level - 1)


def normalize(data):
    if len(data.shape) == 1:
        if data.max() == 0:
            normalized = 0
        else:
            normalized = data / data.max()
    elif len(data.shape) == 2:
        normalized = np.zeros_like(data)
        for i in range(data.shape[0]):
            normalized[i, :] = normalize(data[i, :])
    elif len(data.shape) == 3:
        normalized = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                normalized[i, j, :] = normalize(data[i, j, :])
    return normalized


def average_over_range(data, center, n):
    """
    Average 3D data set over 2n slices (predominantly FS) around center
    :param data:        np.array; 3D data set
    :param center:      int; index of the central slice
    :param n:           int; number of slices taken for average (in one direction!)
    :return:            np.array; averaged and normalized data
    """
    start = center - n
    stop = center + n
    avr = np.sum(data[start:stop, :, :], axis=0)
    return normalize(avr)


def sum_edcs_around_k(data, kx, ky, ik=0):
    dkx = np.abs(data.xscale[0] - data.xscale[1])
    min_kx = indexof(kx - ik * dkx, data.xscale)
    max_kx = indexof(kx + ik * dkx, data.xscale)
    min_ky = indexof(ky - ik * dkx, data.yscale)
    max_ky = indexof(ky + ik * dkx, data.yscale)

    edc = np.sum(np.sum(data.data[min_kx:max_kx, min_ky:max_ky, :], axis=0), axis=0)

    return edc


def sum_edcs_around(data, x, y, n=3):
    result = np.zeros((data.shape[0]))
    start_x, stop_x = x - n, x + n
    start_y, stop_y = y-n, y+n
    for xi in range(start_x, stop_x):
        for yi in range(start_y, stop_y):
            result += data[:, yi, xi]
    return normalize(result)


def sum_shifted_cuts(data):
    steps = []
    for xi in range(data.xscale.size):
        tmp = np.sum(data.data[xi, :, :], axis=0)
        steps.append(detect_step(tmp))
    steps = np.array(steps)
    steps_min, steps_max, steps_diff = steps.min(), steps.max(), steps.max() - steps.min()
    cut = np.zeros((data.yscale.size, data.zscale.size + steps_diff))
    nk, ne = data.data[0, :, :].shape
    for idx, step in enumerate(steps):
        shift = np.abs(step - steps_max)
        cut[:, shift:(shift + ne)] += data.data[idx, :, :]
    cut = cut[:, steps_diff:-steps_diff]
    return cut, data.zscale[:cut.shape[1]]


def sum_XPS(data, crop=None, plot=False):

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
            cond1 = np.all([data[narrow].zscale.min() >= min_ergs_i for min_ergs_i in min_ergs])
            cond2 = np.all([data[narrow].zscale.max() >= max_ergs_i for max_ergs_i in max_ergs])
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


def gradient_fill(x, y, fill_color=None, ax=None, origin='upper', lower_curve=None, ymin=None, upper_curve=None,
                  **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    if ax is None:
        ax = plt.gca()

    line, = ax.plot(x, y, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = colors.colorConverter.to_rgb(fill_color)
    z[:, :, :3] = rgb
    z[:, :, -1] = np.linspace(0, alpha, 100)[:, None]

    if ymin is None:
        xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    else:
        xmin, xmax, ymin, ymax = x.min(), x.max(), ymin, y.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax], origin=origin, zorder=zorder)

    if lower_curve is None and upper_curve is None:
        xy = np.column_stack([x, y])
        xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    elif lower_curve is not None and upper_curve is None:
        xy = np.column_stack([x, y])
        xy_lower = np.column_stack([np.flip(x), np.flip(lower_curve)])
        xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], xy_lower, [xmin, ymin]])
    else:
        xy = np.column_stack([x, upper_curve])
        xy_lower = np.column_stack([np.flip(x), np.flip(lower_curve)])
        xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], xy_lower, [xmin, ymin]])
    clip_path = patches.Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)
    return line, im


def scan_whole_FS_for_gaps(data1, data2, erg1, erg2, e_range=None, gap_cutoff=10, bin_edcs=False, n_bin=3):

    data1 = normalize(data1)
    data2 = normalize(data2)
    points_x = []
    points_y = []
    gaps = []

    # sanity check
    if not (data1.shape[1] == data2.shape[1] and data1.shape[2] == data2.shape[2]):
        print('Compared data sets have different dimensions: {} and {}'.format(data1.shape, data2.shape))
        return

    if e_range is None:
        e_range = [-0.3, 0.1]

    if bin_edcs:
        range_x = range(n_bin, data1.shape[1] - n_bin)
        range_y = range(n_bin, data1.shape[2] - n_bin)
    else:
        range_x = range(data1.shape[1])
        range_y = range(data1.shape[2])
    for x in range_x:
        for y in range_y:
            if bin_edcs:
                edc1 = sum_edcs_around(data1, y, x, n=n_bin)
                edc2 = sum_edcs_around(data2, y, x, n=n_bin)
            else:
                edc1 = data1[:, x, y]
                edc2 = data2[:, x, y]
            e_mid_1 = find_mid_old(erg1, edc1, e_range)
            e_mid_2 = find_mid_old(erg2, edc2, e_range)
            gap_size = np.abs(e_mid_1[0] - e_mid_2[0]) * 1000
            if gap_size > gap_cutoff:
                points_x.append(x)
                points_y.append(y)
                gaps.append(gap_size)

    print('{} points were found.'.format(len(points_x)))

    return [np.array(points_x), np.array(points_y), gaps]


# +-----------------------+ #
# | Plotting functions    | # ==============================================================
# +-----------------------+ #


def set_mpl_rcparams(fontsize=9, color='b1b6d5', print_params=False):
    mpl.rcParams['font.size'] = fontsize
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize
    # mpl.rcParams[]
    # mpl.rcParams['ytick.color'] = color
    # mpl.rcParams['xtick.color'] = color
    # mpl.rcParams['axes.titlecolor'] = color
    # mpl.rcParams['axes.labelcolor'] = color
    mpl.rcParams['figure.autolayout'] = True
    # mpl.rcParams["figure.facecolor"] = '333a47'
    mpl.rcParams['pcolor.shading'] = 'auto'
    if print_params:
        print(mpl.rcParams.keys())


def plot_n_bandmaps(data, momentum, energies, names, e_cutoff=None, gamma=1, fsize=12):

    n = len(data)
    if n < 4:
        nrow = 1
        ncol = n
    elif 4 <= n < 9:
        nrow = 2
        ncol = n // 2 + n % 2
    elif n == 9:
        nrow = 3
        ncol = 3
    elif 9 <= n < 13:
        nrow = 3
        ncol = 4
    elif 13 <= n < 16:
        nrow = 3
        ncol = 5
    elif 16 <= n < 19:
        nrow = 3
        ncol = 6
    else:
        ans = input('Function works up to 18 files so far, and your input has {}. '
                    'Plot first 18 or abort? (p/a)'.format(n))
        if ans == 'p':
            data = data[:13]
            momentum = momentum[:13]
            energies = energies[:13]
            names = names[:13]
            nrow = 3
            ncol = 6
        else:
            return

    # if qt:
    #     % matplotlib qt
    #     # mpl.rcParams['ytick.color'] = 'black'
        # mpl.rcParams['xtick.color'] = 'black'

    fig, axs = plt.subplots(nrow, ncol, figsize=[16, 15])
    idx = 0
    # names = ['Horizontal', 'Vertical']
    for row in range(nrow):
        for col in range(ncol):
            if idx < n:
                if e_cutoff is not None:
                    e_idx = indexof(e_cutoff, energies[idx])
                else:
                    e_idx = 0
                if nrow == 1:
                    k, e = np.meshgrid(momentum[idx], energies[idx][e_idx:])
                    axs[col].pcolormesh(k, e, data[idx][e_idx:, :].T, norm=colors.PowerNorm(gamma=gamma))
                    # axs[col].set_title('{}: {}'.format(idx, names[idx]), fontsize=fsize)
                    axs[col].set_title('{}'.format(names[idx]), fontsize=fsize)
                    axs[col].set_ylabel('$E_b, eV$', fontsize=fsize)
                    axs[col].set_xlabel('$k_x, \AA^{-1}$', fontsize=fsize)
                    # axs[col].set_xlabel('deg', fontsize=fsize)
                else:
                    k, e = np.meshgrid(momentum[idx], energies[idx][e_idx:])
                    axs[row, col].pcolormesh(k, e, data[idx][e_idx:, :].T, norm=colors.PowerNorm(gamma=gamma))
                    # axs[row, col].set_title('{}: {}'.format(idx, names[idx]), fontsize=fsize)
                    axs[row, col].set_title('{}'.format(names[idx]), fontsize=fsize)
                    axs[row, col].set_ylabel('$E_b, eV$', fontsize=fsize)
                    # axs[row, col].set_xlabel('$k_x, \AA^{-1}$',fontsize=fsize)
                    axs[row, col].set_xlabel('deg', fontsize=fsize)
                    # axs[row, col].text(-1, 0.05, 'T=11 K', fontsize=18, color='w')
                idx += 1


def plot_n_const_E_maps(data, kx, ky, energies, names, E=0, bin=3, e_cutoff=None, gamma=1, fsize=12, points=None):

    n = len(data)
    if n < 4:
        nrow = 1
        ncol = n
    elif 4 <= n < 9:
        nrow = 2
        ncol = n // 2 + n % 2
    elif n == 9:
        nrow = 3
        ncol = 3
    elif 9 <= n < 13:
        nrow = 3
        ncol = 4
    elif 13 <= n < 16:
        nrow = 3
        ncol = 5
    elif 16 <= n < 19:
        nrow = 3
        ncol = 6
    else:
        ans = input('Function works up to 18 files so far, and your input has {}. '
                    'Plot first 18 or abort? (p/a)'.format(n))
        if ans == 'p':
            data = data[:13]
            kx = kx[:13]
            ky = ky[:13]
            names = names[:13]
            nrow = 3
            ncol = 6
        else:
            return

    fig, axs = plt.subplots(nrow, ncol, figsize=[10, 7])
    idx = 0

    if points is not None:
        tmpx = np.array([kx[0][xx] for xx in points[1]])
        tmpy = np.array([ky[0][yy] for yy in points[0]])
        x, y = [], []
        for xx, yy in zip(tmpx, tmpy):
            if np.sqrt(xx ** 2 + yy ** 2) < 10:
                x.append(xx)
                y.append(yy)
    # names = ['T = 30 K', 'T = 40 K', 'T = 50 K']
    for row in range(nrow):
        for col in range(ncol):
            if idx < n:
                E_idx = indexof(E, energies[idx])
                map = np.sum(data[idx][(E_idx - bin):(E_idx + bin), :, :], axis=0)
                print(map.shape)
                kxx, kyy = np.meshgrid(kx[idx], ky[idx])
                if nrow == 1:
                    axs[col].pcolormesh(kxx, kyy, map, norm=colors.PowerNorm(gamma=gamma))
                    # axs[col].set_title('{}: {}'.format(idx, names[idx]), fontsize=fsize)
                    axs[col].set_title('{}'.format(names[idx]), fontsize=fsize)
                    if col == 0:
                        axs[col].set_ylabel('$k_y, \AA^{-1}$', fontsize=fsize)
                    axs[col].set_xlabel('$k_x, \AA^{-1}$', fontsize=fsize)
                    axs[col].set_aspect('equal')
                    if points is not None:
                        x = np.array([kx[xx] for xx in points[0]])
                        y = np.array([ky[yy] for yy in points[1]])
                        axs[col].scatter(x, y, c='red', s=5)
                else:
                    axs[row, col].pcolormesh(kxx, kyy, map, norm=colors.PowerNorm(gamma=gamma))
                    # axs[row, col].set_title('{}: {}'.format(idx, names[idx]), fontsize=fsize)
                    axs[row, col].set_title('{}'.format(names[idx]), fontsize=fsize)
                    axs[row, col].set_ylabel('$k_y, \AA^{-1}$', fontsize=fsize)
                    axs[row, col].set_xlabel('$k_x, \AA^{-1}$',fontsize=fsize)
                    axs[row, col].set_aspect('equal')
                    # axs[row, col].text(-1, 0.05, 'T=11 K', fontsize=18, color='w')
                    if not (points is None) and idx != 1:
                        axs[row, col].scatter(x, y, c='white')
                idx += 1


def plot_kz_in_kspace(data, ang, hvs, work_func=4.5, E0=0, trans_kz=False, gamma=1, plot=False, **kwargs):

    data, ang, hvs = np.array(data), np.array(ang), np.array(hvs)
    kx_init = []
    for hv in hvs:
        kx_init.append(a2k_single_axis(ang, hv, **kwargs))
    kx_init = np.array(kx_init)

    if trans_kz:
        k0 = np.sqrt(2 * const.m_e * const.eV) * 1e-10 / const.hbar
        V0 = E0 + work_func
        kxx = kx_init
        kzz = np.zeros_like(kxx)
        print(kxx.shape)
        for kz_i in range(kzz.shape[0]):
            Ek = hvs[kz_i] + work_func
            for kz_j in range(kzz.shape[1]):
                theta = np.deg2rad(ang[kz_j])
                k_ij = k0 * np.sqrt(Ek * (np.cos(theta) ** 2) + V0)
                kzz[kz_i][kz_j] = k_ij
        kz_map = data.T
    else:
        kx_max, kx_min = kx_init.max(), kx_init.min()
        dkx_min = min(np.abs(kx_init[0][0] - kx_init[0][1]), np.abs(kx_init[-1][0] - kx_init[-1][1]))
        kx = np.arange(kx_min, kx_max, dkx_min)

        kz_map = np.zeros((kx.size, hvs.size))
        print(kz_map.shape, data.shape, kx.shape)

        for kzi in range(hvs.size):
            for idx, kxi in enumerate(kx):
                kxi_max, kxi_min = kx_init[kzi].max(), kx_init[kzi].min()
                if kxi_min < kxi < kxi_max:
                    data_idx = indexof(kxi, kx_init[kzi])
                    kz_map[idx, kzi] = data[data_idx, kzi]
        kxx, kzz = np.meshgrid(kx, hvs)
        kz_map = kz_map.T

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=[10, 7])
        ax.pcolormesh(kxx, kzz, kz_map, norm=colors.PowerNorm(gamma=gamma), **kwargs)
        ax.set_xlabel('$k_x, \AA^{-1}$', **kwargs)
        if trans_kz:
            ax.set_ylabel('$k_z, \AA^{-1}$', **kwargs)
            ax.set_title('$k_z$ dependence along $\Gamma$-M, @E$_F$', **kwargs)
        else:
            ax.set_ylabel('$hv, eV$', **kwargs)
            ax.set_title('photon energy dependence along $\Gamma$-M, @E$_F$', **kwargs)


# +---------------------+ #
# | Image processing    | # ==============================================================
# +---------------------+ #

def find_gamma(FS, x0, y0, method='Nelder-Mead', print_output=False):
    """
    Wrapper for doing minimization and finding Gamma in Fermi surface.
    :param FS:              np.array; FS 2D data
    :param x0:              int; initial guess for x-axis coordinate
    :param y0:              int; initial guess for y-axis coordinate
    :param method:          string; method used for scipy minimization; default: simplex, which was tested and should be enough.
                            For other methods check documentation for <scipy.optimize.minimize>.
    :param print_output:    bool; printing option
    :return:
            res.x   np.array; [x0, y0] minimized values
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
        entries = [status, '{:.4f}'.format(-res.fun), res.nit, '{:.2f}'.format(res.x[0]), '{:.2f}'.format(res.x[1])]

        d = {'param': labels,
             'value': entries}
        df = pd.DataFrame(data=d)
        pd.set_option('display.max_columns', 50)
        print(df)

    return res


def rotate_around_xy(init_guess, data_org):
    """
    Rotate matrix by 180 deg around (x0, y0) and return only overlapping region, original and flipped
    :param init_guess:      list; [x0, y0] center of rotation coordinates
    :param data_org:        np.array; original full FS
    :return:
            r:              float; corelation coeffitient of the cut of the data, and flipped one
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

    # specify boundries
    x_start = int(np.max([0, x0 - x]))
    x_stop = int(np.min([x + x0, x_org]))
    y_start = int(np.max([0, y0 - y]))
    y_stop = int(np.min([y + y0, y_org]))

    data = data_org[x_start:x_stop, y_start:y_stop]

    # calculate correlation coeffitient (minus, because we actually want to maximize it)
    if transposed:
        return -imgs_corr(data.T, np.flip(data).T)
    else:
        return -imgs_corr(data, np.flip(data))


def imgs_corr(img1, img2):
    """
    Calculate the correlation coefficient between two matrices based on (PubMed ID: 2362201)
    :param img1:    np.array; original image
    :param img2:    np.array; rotated image
    :return:
            r:      float; correlation coefficient [-1 - total garbage; 1 - perfect match]
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


def curvature_1d(data, a0=0.0005, nb=None, rl=None, xaxis=None):

    if (nb is not None) and (rl is not None):
        data = smooth(data, n_box=nb, recursion_level=rl)
    df = np.gradient(data)
    d2f = np.gradient(df)
    # df = df
    # d2f = d2f

    df_max = np.max(np.abs(df) ** 2)
    C_x = d2f / (np.sqrt(a0 + ((df / df_max) ** 2)) ** 3)
    if xaxis is not None:
        ef = indexof(0, xaxis)
        C_x[ef:] = 0

    return normalize(np.abs(C_x))


def curvature_2d(data, dx, dy, a0=100, nb=None, rl=None, eaxis=None):

    if (nb is not None) and (rl is not None):
        data = smooth_2d(data, n_box=nb, recursion_level=rl)

    dfdx = np.gradient(data, axis=0)
    dfdy = np.gradient(data, axis=1)
    d2fdx2 = np.gradient(dfdx, axis=0)
    d2fdy2 = np.gradient(dfdy, axis=1)
    d2fdxdy = np.gradient(dfdx, axis=1)

    cx = a0 * (dx ** 2)
    cy = a0 * (dy ** 2)

    nom = (1 + cx * (dfdx ** 2)) * cy * d2fdy2 - \
          2 * cx * cy * dfdx * dfdy * d2fdxdy + \
          (1 + cy * (dfdy ** 2)) * cx * d2fdx2
    den = (1 + cx * (dfdx ** 2) + cy * (dfdy ** 2)) ** 1.5

    C_xy = nom / den

    if eaxis is not None:
        ef = indexof(0, eaxis)
        C_xy[ef:, :] = 0

    return np.abs(C_xy)


def exclude_points(cut, points, r, xscale, yscale, intensity_cutoff=0):
    kx = []
    ky = []
    tmp = np.zeros_like(cut)
    for p in points:
        for i in range(cut.shape[0]):
            for j in range(cut.shape[1]):
                if (np.sqrt((xscale[j] - xscale[p[0]])**2 + (yscale[i] - yscale[p[1]])**2) < r) and (cut[i][j] > intensity_cutoff):
                    kx.append(xscale[j])
                    ky.append(yscale[i])
                    tmp[i][j] = cut[i][j]

    return np.array(kx), np.array(ky), tmp


def get_points(data, xscale, yscale, r, intensity_cutoff):
    """
    This function assumes center in (0, 0)
    :param data:
    :param xscale:
    :param yscale:
    :param r:
    :param intensity_cutoff:
    :return:
    """
    X = []
    Y = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if (np.sqrt(xscale[j]**2 + yscale[i]**2) < r) and (data[i][j] > intensity_cutoff):
                X.append(xscale[j])
                Y.append(yscale[i])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def fit_ellipse(data, xscale, yscale, r, intensity_cutoff, printing=True):
    """
    Fit ellipse to the 2D data set.
    The routine is based on: https://github.com/ndvanforeest/fit_ellipse.git
    :param data:                np.array; 2D data set. For better performance make sure that each EDC is normalized;
                                otherwise 'intensity_cutoff' won't make any difference.
    :param xscale:              np.array; 1D array of x coordinates of whatever units
    :param yscale:              np.array; 1D array of x coordinates of whatever units
    :param r:                   float; cutoff radius of of data. Centre is assumed to be in zero in units specified in
                                'xscale' and 'yscale'. Data above 'r' will be excluded
    :param intensity_cutoff:    float; treat intensities below 'intensity_cutoff' as noise and exclude them from fit
    :param printing:            bool; printing option
    :return:
            xx:                 np.array; 1D array of ellipse's x coordinates
            yy:                 np.array; 1D array of ellipse's y coordinates
    """
    # prepare points for fit. Be careful, your destiny depends highly on your choices here
    x, y = get_points(data, xscale, yscale, r, intensity_cutoff)

    def __fit_ellipse(x, y):
        x, y = x[:, np.newaxis], y[:, np.newaxis]
        D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
        S, C = np.dot(D.T, D), np.zeros([6, 6])
        C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
        U, s, V = np.linalg.svd(np.dot(np.linalg.inv(S), C))
        a = U[:, 0]
        return a

    def ellipse_center(a):
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        num = b * b - a * c
        x0 = (c * d - b * f) / num
        y0 = (a * f - b * d) / num
        return np.array([x0, y0])

    def ellipse_axis_length(a):
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
        down1 = (b * b - a * c) * (
                (c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
        )
        down2 = (b * b - a * c) * (
                (a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
        )
        res1 = np.sqrt(up / down1)
        res2 = np.sqrt(up / down2)
        return np.array([res1, res2])

    def ellipse_angle_of_rotation(a):
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        return math.atan2(2 * b, (a - c)) / 2

    a = __fit_ellipse(x, y)
    centre = ellipse_center(a)
    phi = ellipse_angle_of_rotation(a)
    M, m = ellipse_axis_length(a)
    # assert that the major axix M > minor axis m
    if m > M:
        M, m = m, M
    # ensure the angle is betwen 0 and 2*pi
    phi -= 2 * np.pi * int(phi / (2 * np.pi))
    # print some useful info
    if printing:
        # print('The ellipse is given by {:.3}x^2 + {:.3}xy + {:.3}y^2 + {:.3}x + '
        #       '{:.3}y + {:.3} = 0'.format(a[0], a[1], a[2], a[3], a[4], a[5]))
        print('major axis = {:.5} 1/A'.format(M))
        print('minor axis = {:.5} 1/A'.format(m))
        # print('centre = ({:.3}) ({:.3})'.format(centre[0], centre[1]))
        print('Ellipse area = {:.5} A^-2'.format(np.pi * M * m))

    # get data for plotting
    R = np.arange(0, 2 * np.pi, 0.001)
    xx = centre[0] + M * np.cos(R) * np.cos(phi) - m * np.sin(R) * np.sin(phi)
    yy = centre[1] + M * np.cos(R) * np.sin(phi) + m * np.sin(R) * np.cos(phi)

    return xx, yy


def order_points(x0, y0):
    """
    Sort random set of points to create list of polygon's coordinates (closed shape)
    :param x0:
    :param y0:
    :return:
    """
    x = np.array(x0)
    y = np.array(y0)
    x_res = np.array([x[0]])
    y_res = np.array([y[0]])
    if not (x.size == y.size):
        print('x and y must be same size!')
        return
    while x.size > 0:
        dist = 100
        idx = len(list(x0))
        for i in range(x.size):
            tmp = np.sqrt((x_res[-1] - x[i])**2 + (y_res[-1] - y[i])**2)
            if tmp < dist:
                dist = tmp
                idx = i
        x_res = np.append(x_res, x[idx])
        y_res = np.append(y_res, y[idx])
        x = np.delete(x, idx)
        y = np.delete(y, idx)

    return x_res, y_res


def shape_area(x, y):
    """
    Get arbitrary polygon area using Shoelace formula. Vertices are described by their Cartesian
    coordinates in the plane.
    :param x:   np.array; 1D, x coorginates
    :param y:   np.array; 1D, y coordinates
    :return:    float; area
    """
    S = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return S


# +-----------------------+ #
# | K-space conversion    | # ==============================================================
# +-----------------------+ #


def k_fac(energy, Eb=0, hv=0, work_func=4.5):
    me = const.m_e / const.eV / 1e20
    hbar = const.hbar_eV
    return np.sqrt(2 * me * (energy - Eb + hv - work_func)) / hbar


def angle2kscape(scan_ax, anal_ax, d_scan_ax=0, d_anal_ax=0, orientation='horizontal', a=np.pi, energy=np.array([0]),
                 **kwargs):
    # Angle to radian conversion and setting offsets
    scan_ax, anal_ax, energy = np.array(scan_ax), np.array(anal_ax), np.array(energy)
    d_scan_ax = -np.deg2rad(d_scan_ax)
    d_anal_ax = -np.deg2rad(d_anal_ax)
    scan_ax = np.deg2rad(scan_ax) + d_scan_ax
    anal_ax = np.deg2rad(anal_ax) + d_anal_ax

    nkx, nky, ne = scan_ax.size, anal_ax.size, energy.size

    # single momentum axis for specified binding energy
    if (nkx == 1) and (ne == 1):
        ky = np.zeros(nky)
        k0 = k_fac(energy, **kwargs)
        k0 *= (a / np.pi)
        if orientation == 'horizontal':
            ky = np.cos(d_scan_ax) * np.sin(anal_ax)
        elif orientation == 'vertical':
            ky = np.sin(anal_ax)
        return k0 * ky, 1

    # momentum vs energy, e.g. for band maps
    elif (nkx == 1) and (ne != 1):
        ky = np.zeros((ne, nky))
        erg = np.zeros_like(ky)
        if orientation == 'horizontal':
            for ei in range(ne):
                k0i = k_fac(energy[ei], **kwargs)
                k0i *= (a / np.pi)
                ky[ei] = k0i * np.cos(d_scan_ax) * np.sin(anal_ax)
                erg[ei] = energy[ei] * np.ones(nky)
        elif orientation == 'vertical':
            for ei in range(ne):
                k0i = k_fac(energy[ei], **kwargs)
                k0i *= (a / np.pi)
                ky[ei] = k0i * np.sin(anal_ax)
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
                kx[kxi] = np.ones(nky) * np.sin(scan_ax[kxi])
                ky[kxi] = np.cos(scan_ax[kxi]) * np.sin(anal_ax)
        elif orientation == 'vertical':
            for kxi in range(nkx):
                kx[kxi] = np.cos(anal_ax) * np.sin(scan_ax[kxi])
                ky[kxi] = np.sin(anal_ax)
        return k0 * kx, k0 * ky

    # 3D set of momentum vs momentum coordinates, for all given binding energies
    elif (nkx != 1) and (ne != 1):
        kx = np.zeros((ne, nkx, nky))
        ky = np.zeros_like(kx)
        for ei in range(ne):
            k0i = k_fac(energy[ei], **kwargs)
            k0i *= (a / np.pi)
            if orientation == 'horizontal':
                for kxi in range(nkx):
                    kx[ei, kxi, :] = k0i * np.ones(nky) * np.sin(scan_ax[kxi])
                    ky[ei, kxi, :] = k0i * np.cos(scan_ax[kxi]) * np.sin(anal_ax)
            elif orientation == 'vertical':
                for kxi in range(nkx):
                    kx[ei, kxi, :] = k0i * np.cos(anal_ax) * np.sin(scan_ax[kxi])
                    ky[ei, kxi, :] = k0i * np.sin(anal_ax)
        return kx, ky


def hv2kz(ang, hvs, work_func=4.5, V0=0, trans_kz=False, c=np.pi, energy=np.array([0]), **kwargs):

    ang, hvs, energy = np.array(ang), np.array(hvs), np.array(energy)
    ky = []
    for hv in hvs:
        kyi, _ = angle2kscape(np.array([1]), ang, hv=hv, energy=energy, **kwargs)
        ky.append(kyi)

    ky = np.array(ky)
    kz = np.zeros_like(ky)
    me = const.m_e / const.eV / 1e20
    hbar = const.hbar_eV
    k0 = np.sqrt(2 * me) / hbar
    k0 *= (c / np.pi)

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


############################## deprecated ##############################


def a2k(scan_ax, anal_ax, hv, d_scan_ax=0, d_anal_ax=0, orientation='horizontal', work_func=4.5, a=np.pi,
        Eb=0, polar=0, azimuth=0, tilt=0):
    print('*** deprecated ***')
    """
    Convert angles of the experimental geometry to k-space coordinates.
    Confer the sheet "ARPES angle to k-space conversion" [doc/a2k.pdf] for
    detailed explanations.

    **Parameters**

    ===========  ===============================================================
    alpha        array of length *nkx*; angles in degrees along the
                 independent rotation (often called "theta" or "polar" at
                 beamlines.
    beta         array of length *nky*; angles in degrees along the dependent
                 rotation (not the azimuth, often called "tilt").
    hv           float; used photon energy in eV.
    dalpha       float; offset to *alpha* in degrees.
    dbeta        float; offset to *alpha* in degrees.
    orientation  str; determines the analyzer slit orientation, which can be
                 horizontal (default) or vertical. The first letter of the
                 given string must be either 'h' or 'v'.
    work_func    float; value of the work function in eV.
    ===========  ===============================================================

    **Returns**

    ==  ========================================================================
    KX  array of shape (nkx, nky); mesh of k values in parallel direction in
        units of inverse Angstrom.
    KY  array of shape (nkx, nky); mesh of k values in perpendicular
        direction in units of inverse Angstrom.
    ==  ========================================================================
    """
    # constant factor contribution
    k0 = np.sqrt(2 * const.m_e * const.eV) * 1e-10 / const.hbar
    # k0 = np.sqrt(2 * const.m_e) / const.hbar
    # energy contribution
    k0 *= np.sqrt(hv - work_func + Eb)
    # lattice constant contribution
    k0 *= (np.pi / a)
    # Angle to radian conversion and setting offset
    d_scan_ax = -np.deg2rad(d_scan_ax)
    d_anal_ax = -np.deg2rad(d_anal_ax)
    polar = np.deg2rad(polar)
    azimuth = np.deg2rad(azimuth)
    tilt = np.deg2rad(tilt)
    scan_ax = np.deg2rad(scan_ax) + d_scan_ax
    anal_ax = np.deg2rad(anal_ax) + d_anal_ax

    # Prepare containers
    nkx = len(scan_ax)
    nky = len(anal_ax)
    KX = np.empty((nkx, nky))
    KY = np.empty((nkx, nky))

    slit = orientation.lower()[0]
    if slit == 'h':
        for i in range(nkx):
            t_rot = rot_mat(polar=polar, azimuth=azimuth, tilt=scan_ax[i])
            KX[i] = t_rot[0, 1] * np.sin(anal_ax) + t_rot[0, 2] * np.cos(anal_ax)
            KY[i] = t_rot[1, 1] * np.sin(anal_ax) + t_rot[1, 2] * np.cos(anal_ax)
        # for i in range(nkx):
        #     KX[i] = np.sin(anal_ax) * np.cos(scan_ax[i])
        #     KY[i] = np.sin(scan_ax[i])
    elif slit == 'v':
        for i in range(nkx):
            t_rot = rot_mat(polar=scan_ax[i], azimuth=azimuth, tilt=tilt)
            KY[i] = -t_rot[0, 0] * np.sin(anal_ax) + t_rot[0, 2] * np.cos(anal_ax)
            KX[i] = -t_rot[1, 0] * np.sin(anal_ax) + t_rot[1, 2] * np.cos(anal_ax)
        # KX = KX.T
        # cos_da = np.cos(d_scan_ax)
        # sin_theta_cos_beta = np.sin(d_scan_ax) * np.cos(anal_ax)
        # for i in range(nkx):
        #     KX[i] = sin_theta_cos_beta + cos_da * np.cos(scan_ax[i]) * np.sin(anal_ax)
        #     KY[i] = cos_da * np.sin(scan_ax[i])
    elif slit == 'd':
        t_rot = rot_mat(polar=polar, azimuth=azimuth, tilt=tilt)
        for i in range(nkx):
            eta = np.sqrt(anal_ax ** 2 + scan_ax[i] ** 2)
            s_eta = np.sin(eta) / eta
            KX[i] = -t_rot[0, 0] * scan_ax[i] * s_eta + t_rot[0, 1] * anal_ax * s_eta + t_rot[0, 2] * np.cos(eta)
            KY[i] = -t_rot[1, 0] * scan_ax[i] * s_eta + t_rot[1, 1] * anal_ax * s_eta + t_rot[1, 2] * np.cos(eta)
    else:
        raise ValueError('Orientation "{}" not understood.'.format(orientation))
    return k0 * KX, k0 * KY


def a2k_piva(scan_ax, anal_ax, hv, d_scan_ax=0, d_anal_ax=0, orientation='horizontal', work_func=4.5, a=np.pi,
        Eb=0, polar=0, azimuth=0, tilt=0):
    """
    Convert angles of the experimental geometry to k-space coordinates.
    Confer the sheet "ARPES angle to k-space conversion" [doc/a2k.pdf] for
    detailed explanations.

    **Parameters**

    ===========  ===============================================================
    alpha        array of length *nkx*; angles in degrees along the
                 independent rotation (often called "theta" or "polar" at
                 beamlines.
    beta         array of length *nky*; angles in degrees along the dependent
                 rotation (not the azimuth, often called "tilt").
    hv           float; used photon energy in eV.
    dalpha       float; offset to *alpha* in degrees.
    dbeta        float; offset to *alpha* in degrees.
    orientation  str; determines the analyzer slit orientation, which can be
                 horizontal (default) or vertical. The first letter of the
                 given string must be either 'h' or 'v'.
    work_func    float; value of the work function in eV.
    ===========  ===============================================================

    **Returns**

    ==  ========================================================================
    KX  array of shape (nkx, nky); mesh of k values in parallel direction in
        units of inverse Angstrom.
    KY  array of shape (nkx, nky); mesh of k values in perpendicular
        direction in units of inverse Angstrom.
    ==  ========================================================================
    """
    print('*** deprecated ***')
    # constant factor contribution
    k0 = np.sqrt(2 * const.m_e * const.eV) * 1e-10 / const.hbar
    # k0 = np.sqrt(2 * const.m_e) / const.hbar
    # energy contribution
    k0 *= np.sqrt(hv - work_func + Eb)
    # lattice constant contribution
    k0 *= (np.pi / a)
    # Angle to radian conversion and setting offset
    d_scan_ax = -np.deg2rad(d_scan_ax)
    d_anal_ax = -np.deg2rad(d_anal_ax)
    polar = np.deg2rad(polar)
    azimuth = np.deg2rad(azimuth)
    tilt = np.deg2rad(tilt)
    scan_ax = np.deg2rad(scan_ax) + d_scan_ax
    anal_ax = np.deg2rad(anal_ax) + d_anal_ax

    # Prepare containers
    nkx = len(scan_ax)
    nky = len(anal_ax)
    KX = np.empty((nkx, nky))
    KY = np.empty((nkx, nky))

    slit = orientation.lower()[0]
    if slit == 'h':
        for i in range(nkx):
            t_rot = rot_mat(polar=polar, azimuth=azimuth, tilt=scan_ax[i])
            KX[i] = t_rot[0, 1] * np.sin(anal_ax) + t_rot[0, 2] * np.cos(anal_ax)
            KY[i] = t_rot[1, 1] * np.sin(anal_ax) + t_rot[1, 2] * np.cos(anal_ax)
        # for i in range(nkx):
        #     KX[i] = np.sin(anal_ax) * np.cos(scan_ax[i])
        #     KY[i] = np.sin(scan_ax[i])
    elif slit == 'v':
        for i in range(nkx):
            t_rot = rot_mat(polar=scan_ax[i], azimuth=azimuth, tilt=tilt)
            KX[i] = -t_rot[0, 0] * np.sin(anal_ax) + t_rot[0, 2] * np.cos(anal_ax)
            KY[i] = -t_rot[1, 0] * np.sin(anal_ax) + t_rot[1, 2] * np.cos(anal_ax)
        # cos_da = np.cos(d_scan_ax)
        # sin_theta_cos_beta = np.sin(d_scan_ax) * np.cos(anal_ax)
        # for i in range(nkx):
        #     KX[i] = sin_theta_cos_beta + cos_da * np.cos(scan_ax[i]) * np.sin(anal_ax)
        #     KY[i] = cos_da * np.sin(scan_ax[i])
    elif slit == 'd':
        t_rot = rot_mat(polar=polar, azimuth=azimuth, tilt=tilt)
        for i in range(nkx):
            eta = np.sqrt(anal_ax ** 2 + scan_ax[i] ** 2)
            s_eta = np.sin(eta) / eta
            KX[i] = -t_rot[0, 0] * scan_ax[i] * s_eta + t_rot[0, 1] * anal_ax * s_eta + t_rot[0, 2] * np.cos(eta)
            KY[i] = -t_rot[1, 0] * scan_ax[i] * s_eta + t_rot[1, 1] * anal_ax * s_eta + t_rot[1, 2] * np.cos(eta)
    else:
        raise ValueError('Orientation "{}" not supported.'.format(orientation))
    return k0 * KX, k0 * KY


def a2k_single_axis(alpha, hv, beta=0, dalpha=0, work_func=4.5, a=np.pi, Eb=0, orientation='horizontal'):
    """
    Convert angles of the experimental geometry to k-space coordinates.
    Confer the sheet "ARPES angle to k-space conversion" [doc/a2k.pdf] for
    detailed explanations.

    **Parameters**

    ===========  ===============================================================
    alpha        array of length *nkx*; angles in degrees along the
                 independent rotation (often called "theta" or "polar" at
                 beamlines.
    hv           float; used photon energy in eV.
    dalpha       float; offset to *alpha* in degrees.
    work_func    float; value of the work function in eV.
    Eb           float; binding energy in eV
    ===========  ===============================================================

    **Returns**

    ==  ========================================================================
    KX  array of shape (nkx); mesh of k values in parallel direction in
        units of inverse Angstrom.
    ==  ========================================================================
    """
    print('*** deprecated ***')
    # constant factor contribution
    k0 = np.sqrt(2 * const.m_e * const.eV) * 1e-10 / const.hbar
    # energy contribution
    k0 *= np.sqrt(hv - work_func - Eb)
    # lattice constant contribution
    k0 *= (np.pi / a)
    # Angle to radian conversion and setting offset
    dalpha = -np.deg2rad(dalpha)
    alpha = np.deg2rad(alpha) + dalpha
    beta = np.deg2rad(beta)

    # Prepare containers
    KX = np.empty(len(alpha))

    slit = orientation.lower()[0]
    if slit == 'h':
        KX = np.sin(alpha) * np.cos(beta)
    elif slit == 'v':
        cos_da = np.cos(dalpha)
        sin_theta_cos_beta = np.sin(dalpha) * np.cos(beta)
        KX = sin_theta_cos_beta + cos_da * np.cos(alpha) * np.sin(beta)

    return k0 * KX


def hv2k(ang, hvs, work_func=4.5, V0=0, trans_kz=False, c=np.pi, **kwargs):

    print('*** deprecated ***')
    ang, hvs = np.array(ang), np.array(hvs)
    kx = []
    for hv in hvs:
        kx.append(a2k_single_axis(ang, hv, **kwargs))
    kx = np.array(kx)
    kz = np.zeros_like(kx)

    if trans_kz:
        k0 = np.sqrt(2 * const.m_e * const.eV) * 1e-10 / const.hbar
        k0 *= (c / np.pi)
        for kz_i in range(kz.shape[0]):
            Ek = hvs[kz_i] + work_func
            for kz_j in range(kz.shape[1]):
                theta = np.deg2rad(ang[kz_j])
                k_ij = k0 * np.sqrt(Ek * (np.cos(theta) ** 2) + V0)
                kz[kz_i][kz_j] = k_ij
    else:
        for kzi, hv in enumerate(hvs):
            kz[kzi, :] = np.ones_like(ang) * hv

    return kx, kz


def rot_mat(polar=0, azimuth=0, tilt=0):

    print('*** deprecated ***')

    def c(angle):
        return np.cos(angle)

    def s(angle):
        return np.sin(angle)

    p = polar
    a = azimuth
    tlt = tilt
    T_rot = [[c(tlt) * c(a),  -c(p) * s(a) + s(p) * s(tlt) * c(a),  s(p) * s(a) + c(p) * s(tlt) * c(a)],
             [c(tlt) * s(a),  c(p) * c(a) + s(p) * s(tlt) * s(a),   -s(p) * c(a) + c(p) * s(tlt) * s(a)],
             [-s(tlt),        s(p) * c(tlt),                        c(p) * c(tlt)]]
    return np.array(T_rot)


# +----------+ #
# | Misc     | # ==============================================================
# +----------+ #


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


