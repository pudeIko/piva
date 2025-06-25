"""
Automated test for working_procedurec.py module.
"""

import numpy as np
from scipy.optimize import curve_fit
import os
import pkg_resources
from pkg_resources import resource_filename
import piva.working_procedures as wp
import piva.data_loaders as dl
import piva.constants as const
import pickle

TEST_DATA_PATH = os.path.join(resource_filename("piva", "tests"), "data")
TEST_DATA_FILE = os.path.join(TEST_DATA_PATH, "wp_test_data.p")


class TestWP:
    """
    General class implementing individual steps.
    """

    def load_data(self):
        """
        Load data for running tests.
        """

        with open(TEST_DATA_FILE, "rb") as f:
            data = pickle.load(f)
        self.map = data["map"]
        self.cut = data["cut"]
        self.edc = data["edc"]
        self.mdc = data["mdc"]
        self.xps = data["xps"]

    def check_const(self):
        """
        Check functions in constants module.
        """

        const.convert_eV_nm(1.5)
        const.Ry2eV(1.5)

    def check_fitting_functions(self):
        """
        Test fitting functions.
        """

        # get one peak from MDC
        down_sample = 10
        peak_x = self.mdc.data[400:][::down_sample]
        peak_y = self.mdc.x[400:][::down_sample]

        # simple fitting curves
        popt, cov = curve_fit(wp.gaussian, peak_x, peak_y)
        popt, cov = curve_fit(wp.two_gaussians, peak_x, peak_y)
        popt, cov = curve_fit(wp.lorentzian, peak_x, peak_y)
        # popt, cov = curve_fit(wp.two_lorentzians, peak_x, peak_y)
        popt, cov = curve_fit(wp.asym_lorentzian, peak_x, peak_y)
        wp.get_linear([[0, 1], [0, 1]])
        wp.print_fit_results(popt, cov, ["a", "b", "c", "d", "e"])

        # xps fitting
        ta, ta_erg = wp.sum_XPS([self.xps, self.xps], crop=[-28, -21], plot=False)
        ta, ta_erg = ta[::down_sample], ta_erg[::down_sample]
        ta_bck = wp.shirley_calculate(ta_erg, ta)
        ta = ta - ta_bck
        _ = wp.fit_n_dublets(ta, ta_erg, [1, 1], [-23, -23], [0.1, 0.1], 1.9)

    def check_arpes_analysis_tools(self):
        """
        Test ARPES utilities.
        """

        erg = np.linspace(-3.0, 0.0, 15)
        # Synthetic Lorentzian widths
        gammas = np.array(
            [
                0.42,
                0.39,
                0.36,
                0.33,
                0.30,
                0.27,
                0.25,
                0.23,
                0.21,
                0.19,
                0.18,
                0.16,
                0.15,
                0.14,
                0.13,
            ]
        )

        # Self-energy analysis
        re = wp.kk_im2re(gammas)
        _ = wp.kk_re2im(re)
        _ = wp.find_vF(gammas, re)
        _ = wp.find_vF_v2(erg, np.linspace(0.1, 0.7, 15), 0.7, gammas)

        # Lindhard function
        def ek_tb(kx, ky=None, kz=0.0, t=1.0, mu=0.0):
            return -2 * t * (np.cos(kx) + np.cos(ky)) - mu

        ek_kwargs = {"t": 1.0, "mu": 0.5}
        kx, qx = np.linspace(-np.pi, np.pi, 5), np.linspace(0.0, 0.4 * np.pi, 5)
        ky, kz, qy, qz = np.copy(kx), np.copy(kx), np.copy(qx), np.copy(qx)
        _ = wp.get_chi(
            ek=ek_tb,
            ek_kwargs=ek_kwargs,
            kx=kx,
            qx=qx,
            ky=ky,
            qy=qy,
            kz=kz,
            qz=qz,
            in_plnae=False,
            crop=False,
        )

        # Density wave simulations
        res1D = wp.get_1D_modulation(qx, [1, -1])
        res2D = wp.get_2D_modulation(qx, [np.array([1, 0]), np.array([0, 1])])
        res3D = wp.get_3D_modulation(
            qx, qz, [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        )

        ____ = wp.get_self_energy_of_FDW(
            (qx), (kx), ek_tb, ek_kwargs, np.linspace(-0.5, 0, 5), res1D
        )
        se2D = wp.get_self_energy_of_FDW(
            (qx, qy), (kx, ky), ek_tb, ek_kwargs, np.linspace(-0.5, 0, 5), res2D
        )
        ____ = wp.get_self_energy_of_FDW(
            (qx, qy, qz), (kx, ky, kz), ek_tb, ek_kwargs, np.linspace(-0.5, 0, 5), res3D
        )
        _ = wp.get_A(np.zeros((5, 5)), ek_tb(kx, ky=kx), eta=0.075, sigma=se2D)

    def check_beamline_calibration_utilities(self):
        """
        Test beamline calibration utilities.
        """

        hv = np.linspace(50, 1000, 100)
        data = wp.pgm_calibration(hv, 0.1, 0.05, -0.03, 2.25, 1, 300)
        data += np.random.normal(0, 0.005, size=hv.shape)
        _, _ = wp.fit_PGM_calibration(data, hv)

    def check_gap_analysis(self):
        """
        Test gap analysis utilities.
        """

        # extract step
        edc, erg = self.edc.data[150:250], self.edc.x[150:250] - 75.0 + 4.3
        _ = wp.dec_fermi_div(edc, erg, 10, 0.0, -0.5)
        _ = wp.deconvolve_resolution(edc, erg, 10.0)
        _ = wp.detect_step(edc)
        _ = wp.fit_fermi_dirac(erg, edc, 0.0)
        _ = wp.find_mid_old(erg, edc)
        _ = wp.symmetrize_edc_around_Ef(edc, erg)
        self.ds = dl.load_data(os.path.join(TEST_DATA_PATH, "test_map.p"))
        _ = wp.sum_edcs_around_k(self.ds, 0, 0, 5)

    def check_utilities(self):
        """
        Test remaining utilities.
        """

        edc, _ = self.edc.data[150:250], self.edc.x[150:250] - 75.0 + 4.3
        k_tmp = np.linspace(0, 1, 10)
        _ = wp.smooth(edc)
        for axi in range(3):
            _ = wp.normalize_to_sum(self.ds.data, axis=axi)
        _ = wp.rotate(map, 0.5, deg=False)
        _ = wp.rotate(self.map, 30)
        _ = wp.rotate_vector(np.linspace(0, 1, 10), np.linspace(2, 3, 10), 10)
        _ = wp.shift_k_coordinates(k_tmp, k_tmp)

    def check_image_processing_utilities(self):
        """
        Test image processing procedures.
        """

        _ = wp.curvature_1d(
            self.edc.data,
            wp.get_step(self.edc.x),
            nb=5,
            rl=2,
            xaxis=self.edc.x - 75 + 4.3,
        )

        size, sigma = 32, 0.3
        x, y = np.linspace(-1, 1, size), np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        blurred_circle = np.exp(-((X**2 + Y**2) / (2 * sigma**2)))
        _ = wp.find_gamma(blurred_circle, size // 2, size // 2, print_output=True)

    def check_k_space_conversion(self):
        """
        Test k-space conversion functions.
        """

        _ = wp.angle2kspace(np.array([70]), self.map.x)
        _ = wp.angle2kspace(np.array([70]), self.map.x, orientation="vertical")

        _ = wp.angle2kspace(self.map.x, self.map.y)
        _ = wp.angle2kspace(self.map.x, self.map.y, orientation="vertical")

        _ = wp.angle2kspace(self.map.x, self.map.y, energy=np.linspace(-1, 0, 5))
        _ = wp.angle2kspace(
            self.map.x, self.map.y, orientation="vertical", energy=np.linspace(-1, 0, 5)
        )

        _ = wp.hv2kz(self.map.x, np.linspace(30, 100, 10))
        _ = wp.hv2kz(self.map.x, np.linspace(30, 100, 10), trans_kz=True)

    def check_misc(self):
        """
        Test misc.
        """

        _ = wp.McMillan_Tc()
        _ = wp.dynes_formula(self.edc.x - 75 + 4.3, 1.0, 0.5, 0.01)
        _ = wp.all_equal((2, 2, 3, 2))

    def test_wp(self) -> None:
        """
        Run the test.
        """

        # load test data and check methods in constants
        self.load_data()
        self.check_const()

        # run tests
        self.check_fitting_functions()
        self.check_arpes_analysis_tools()
        self.check_beamline_calibration_utilities()
        self.check_gap_analysis()
        self.check_utilities()
        self.check_image_processing_utilities()
        self.check_k_space_conversion()
        self.check_misc()


if __name__ == "__main__":
    import pytest
    from pkg_resources import resource_filename

    path = os.path.join(pkg_resources.resource_filename("piva", "tests"), "wp_test.py")
    pytest.main(["-v", "-s", path])
