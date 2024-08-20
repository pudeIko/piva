#!/usr/bin/python

# Constants
# =============================================================================
# Avogadro constant [1/mol]
N_A = 6.022e23

# Bohr's Magneton [J/T]
mu_bohr = 9.274009994e-24

# Bohr radius [m]
a0 = 5.2917721067e-11

# Boltzmann constant [J/K]
k_B = 1.38064852e-23

# Dielectric constant in vacuum [C / V / m]
eps0 = 8.854e-12

# Electronvolt [J]
eV = 1.6021766208e-19

# Electron mass [kg]
m_e = 9.10938356e-31

# Planck's constant [J * s]
h = 6.626070040e-34

# reduced Planck's constant [J * s]
hbar = h / (2 * 3.14159265358979)

# reduced Planck's constant [eV * s]
hbar_eV = hbar / eV

# Rydberg constant [1 / m]
R_const = 10973731.56816

# Speed of light [m / s]
c = 299792458.

# Universal Gas constant [J / K / mol]
R = 8.3144598


# Dependent constants
# =============================================================================

# Electronvolt-nanometer conversion for light; 1239.84197
eV_nm_conversion = h * c / eV * 1e9

# Rydberg energy unit [eV]
Ry = 13.605703976


# Utilities
# =============================================================================


def convert_eV_nm(eV_or_nm: float) -> float:
    """
    Convert between electronvolt and nanometers for electromagnetic waves.
    The conversion follows from :math:`E = h*c/\lambda` and is simply::

        nm_or_eV = 1239.84193 / eV_or_nm

    :param eV_or_nm: value in [eV] or [nm] to be converted.
    :return: if [eV] were given, this is the corresponding value in
             [nm], and vice versa.
    """

    nm_or_eV = eV_nm_conversion / eV_or_nm
    return nm_or_eV


def Ry2eV(erg: float) -> float:
    """
    Convert rydberg energy units to electronvolt
    The conversion follows that 1 Ry = 13.6(...) eV

    :param erg: value in [eV] or [Ry] to be converted.
    :return: if [eV] were given, this is the corresponding value in [Ry],
             and vice versa.
    """

    return erg * Ry
