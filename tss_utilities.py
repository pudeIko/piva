# functions and other shit for data processing specific for TaSe{2-x}S{x} samples

import numpy as np

# hopping parameters for both bands in meV, taken from li2018_sm.pdf
# lattice consts in A
a_TaSe2 = 3.4552        # doi: 10.1103/PhysRevB.59.6063
a_TaS2 = 3.314           # doi: 10.1107/S0108270190000014


def get_lattice_const(x=0):
    return ((2 - x) * a_TaSe2 + x * a_TaS2) / 2


def get_initial_t(model='li2018'):
    if model == 'li2018':
        t_barrel = np.array([-45.1, 157.8, 203.2, 25.7, 0.02, 0.48])
        t_dogbone = np.array([501.3, -15.9, 557.1, -72.0, -13.9, 12.2])
    elif model == 'rossnagel2005':
        t_barrel = np.array([-0.0373, 0.276, 0.2868, 0.007]) * 1000
        t_dogbone = np.array([0.4108, -0.0726, 0.4534, -0.12]) * 1000
    elif model == 'inosov2008':
        t_barrel = np.array([-0.064, 0.167, 0.211, 0.005, 0.003]) * 1000
        t_dogbone = np.array([0.369, 0.074, 0.425, -0.049, 0.018]) * 1000
    else:
        print('Wrong model given.')
        return
    return t_dogbone, t_barrel


def coords_trans(kx, ky, x=2, flip=False):
    # for TaSe2 kx and ky flipped comparing to li2018_sm.pdf
    a = get_lattice_const(x=x)
    if flip:
        Xi = 0.5 * ky * a
        Eta = np.sqrt(3) * 0.5 * kx * a
        return Xi, Eta
    else:
        Xi = 0.5 * kx * a
        Eta = np.sqrt(3) * 0.5 * ky * a
        return Xi, Eta


def TB_Ek(kx, ky, *t, x=2, flip=False, model='li2018'):

    kxx, kyy = coords_trans(kx, ky, x=x, flip=flip)
    if flip:
        E_k = np.zeros((ky.size, kx.size))
    else:
        E_k = np.zeros((kx.size, ky.size))
    print(len(t))
    if model == 'li2018':
        Xi = kxx
        Eta = kyy
        for i, xi in enumerate(Xi):
            for j, eta in enumerate(Eta):
                E_k[i][j] = t[0] + t[1] * (2 * np.cos(xi) * np.cos(eta) + np.cos(2 * xi)) + \
                            t[2] * (2 * np.cos(3 * xi) * np.cos(eta) + np.cos(2 * eta)) + \
                            t[3] * (2 * np.cos(2 * xi) * np.cos(2 * eta) + np.cos(4 * xi)) + \
                            t[4] * (np.cos(xi) * np.cos(3 * eta) + np.cos(5 * xi) * np.cos(eta) +
                                    np.cos(4 * xi) * np.cos(2 * eta)) + \
                            t[5] * (np.cos(3 * xi) * np.cos(3 * eta) + np.cos(6 * xi))
                if E_k[i][j] > 50:
                    E_k[i][j] = 50
    elif model == 'rossnagel2005':
        Xi = kxx
        Eta = kyy
        for i, xi in enumerate(Xi):
            for j, eta in enumerate(Eta):
                E_k[i][j] = t[0] + t[1] * (2 * np.cos(xi) * np.cos(eta) + np.cos(2 * xi)) + \
                            t[2] * (2 * np.cos(3 * xi) * np.cos(eta) + np.cos(2 * eta)) + \
                            t[3] * (2 * np.cos(2 * xi) * np.cos(2 * eta) + np.cos(4 * xi))
                if E_k[i][j] > 50:
                    E_k[i][j] = 50
    elif model == 'inosov2008':
        if flip:
            Xi = ky * 0.5
            Eta = kx * np.sqrt(3) * 0.5
        else:
            Xi = kx * 0.5
            Eta = ky * np.sqrt(3) * 0.5
        for i, xi in enumerate(Xi):
            for j, eta in enumerate(Eta):
                E_k[i][j] = t[0] + t[1] * (2 * np.cos(xi) * np.cos(eta) + np.cos(2 * xi)) + \
                            t[2] * (2 * np.cos(3 * xi) * np.cos(eta) + np.cos(2 * eta)) + \
                            t[3] * (2 * np.cos(2 * xi) * np.cos(2 * eta) + np.cos(4 * xi)) + \
                            t[4] * (2 * np.cos(6 * xi) * np.cos(2 * eta) + np.cos(4 * xi))
                if E_k[i][j] > 50:
                    E_k[i][j] = 50

    return E_k


def TB_Ek_fitting(K_points, *t, model='li2018'):
    Xi, Eta = K_points
    if model == 'li2018':
        E_k = t[0] + t[1] * (2 * np.cos(Xi) * np.cos(Eta) + np.cos(2 * Xi)) + \
              t[2] * (2 * np.cos(3 * Xi) * np.cos(Eta) + np.cos(2 * Eta)) + \
              t[3] * (2 * np.cos(2 * Xi) * np.cos(2 * Eta) + np.cos(4 * Xi)) + \
              t[4] * (np.cos(Xi) * np.cos(3 * Eta) + np.cos(5 * Xi) * np.cos(Eta) + np.cos(4 * Xi) * np.cos(2 * Eta)) +\
              t[5] * (np.cos(3 * Xi) * np.cos(3 * Eta) + np.cos(6 * Xi))
    elif model == 'rossnagel2005':
        E_k = t[0] + t[1] * (2 * np.cos(Xi) * np.cos(Eta) + np.cos(2 * Xi)) + \
                    t[2] * (2 * np.cos(3 * Xi) * np.cos(Eta) + np.cos(2 * Eta)) + \
                    t[3] * (2 * np.cos(2 * Xi) * np.cos(2 * Eta) + np.cos(4 * Xi))
    elif model == 'inosov2008':
        E_k = t[0] + t[1] * (2 * np.cos(Xi) * np.cos(Eta) + np.cos(2 * Xi)) + \
                    t[2] * (2 * np.cos(3 * Xi) * np.cos(Eta) + np.cos(2 * Eta)) + \
                    t[3] * (2 * np.cos(2 * Xi) * np.cos(2 * Eta) + np.cos(4 * Xi)) + \
                    t[4] * (2 * np.cos(6 * Xi) * np.cos(2 * Eta) + np.cos(4 * Xi))

    return E_k
