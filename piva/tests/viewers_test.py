"""
Automated test for :class:`~data_viewer_2d.DataViewer2D` and
:class:`~data_viewer_3d.DataViewer3D` testing loading and displaying data and
most of the functionalities.
"""

from piva.data_browser import DataBrowser

# from piva.main import db
from piva.data_viewer_3d import DataViewer3D
from piva.data_viewer_2d import DataViewer2D
from piva.fitters import MDCFitter, EDCFitter
from piva.plot_tool import PlotTool
from piva.working_procedures import get_step
from pyqtgraph.Qt.QtCore import Qt
from PyQt5.QtTest import QTest

# from PyQt5.QtWidgets import QMessageBox
import numpy as np
from pkg_resources import resource_filename
import os
from typing import Any
from unittest.mock import patch

# from piva.utilities_panel import InfoWindow
import piva.config

piva.config.settings.IS_TESTING = True

CHECK_3D_Viewer_ = True
CHECK_2D_Viewer_ = True
CHECK_EDC_FITTER = True
CHECK_METADTA___ = True
CHECK_MDC_FITTER = True
CHECK_PLOT_TOOL_ = True
CHECK_LINKING___ = True
CHECK_K_SPC_CONV = True
CHECK_4D_Viewer_ = True

EXAMPLE_CUT = os.path.join(resource_filename("piva", "tests"), "data")
EXAMPLE_CUT = os.path.join(EXAMPLE_CUT, "test_map.p")
N_SLIDER_E, N_SLIDER_K, N_BINS_E = 20, 26, 3
N_E, N_X, N_Y = 20, 14, 3
K0_IDX = 83
CMAP, CMAP_IDX = "inferno", 27
N_GAMMA = 10
LONG_WT, SHORT_WT = 300, 1


class TestViewers:
    """
    General class implementing individual steps.
    """

    def open_browser(self, qtbot: Any) -> None:
        """
        Open :class:`~data_browser.DataBrowser` window.

        :param qtbot: object emulating a user
        """

        # initialize browser and open data_viewer_3d
        browser = DataBrowser()
        qtbot.wait(QTest.qWaitForWindowExposed(browser))
        qtbot.add_widget(browser)
        browser.mb_open_dir(
            chosen_dir=os.path.join(os.getcwd(), "piva", "tests", "data")
        )
        qtbot.wait(LONG_WT)
        qtbot.keyClick(browser.file_explorer, Qt.Key_Down)
        qtbot.keyClick(browser.file_explorer, Qt.Key_Down)
        qtbot.wait(LONG_WT)
        self.browser = browser
        browser.open_dv(EXAMPLE_CUT)
        self._3dv = browser.data_viewers[EXAMPLE_CUT]
        qtbot.wait(QTest.qWaitForWindowExposed(self._3dv))
        assert isinstance(self._3dv, DataViewer3D)
        self.up = self._3dv.util_panel
        qtbot.wait(LONG_WT)

    def check_3Dv_sliders(self, qtbot: Any) -> None:
        """
        Check response of the sliders in :class:`data_viewer_3d.DataViewer3D`.

        :param qtbot: object emulating a user
        """

        # move sliders
        qtbot.mouseClick(self.up.bin_x, Qt.LeftButton)
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up.bin_y, Qt.LeftButton)

        qtbot.mouseClick(self.up.bin_z, Qt.LeftButton)
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up.bin_z, Qt.LeftButton)

        qtbot.mouseClick(self.up.bin_zx, Qt.LeftButton)

        qtbot.mouseClick(self.up.bin_zx, Qt.LeftButton)

        e0, ex0, ey0 = (
            self.up.energy_main.value(),
            self.up.energy_hor.value(),
            self.up.energy_vert.value(),
        )
        x0_idx, y0_idx = self.up.momentum_vert.value(), self.up.momentum_hor.value()

        change_spinBox(qtbot, self.up.momentum_vert, N_X, Qt.Key_Down)
        assert self.up.momentum_vert.value() is (x0_idx - N_X)
        change_spinBox(
            qtbot, self.up.momentum_hor, N_Y, Qt.Key_Down, time=SHORT_WT // 5
        )
        assert self.up.momentum_hor.value() == (y0_idx - N_Y)
        qtbot.wait(LONG_WT)

        for _ in range(N_E):
            qtbot.keyPress(self.up.energy_main, Qt.Key_Down)
            qtbot.keyPress(self.up.energy_hor, Qt.Key_Down)
            qtbot.keyPress(self.up.energy_vert, Qt.Key_Down)
            qtbot.wait(SHORT_WT // 5)
        assert self.up.energy_main.value() == (e0 - N_E)
        assert self.up.energy_hor.value() == (ex0 - N_E)
        assert self.up.energy_vert.value() == (ey0 - N_E)
        qtbot.wait(LONG_WT)

    def check_3Dv_axes_tab(self, qtbot: Any) -> None:
        """
        Check behavior of the features in **Axes tab** in
        :class:`data_viewer_3d.DataViewer3D`.

        :param qtbot: object emulating a user
        """

        # move to AxesTab, change energy scales
        self.up.tabs.setCurrentIndex(2)
        assert self.up.tabs.currentIndex() == 2
        qtbot.wait(LONG_WT)
        qtbot.keyClicks(self.up.axes_energy_scale, "kinetic")
        assert self.up.axes_energy_scale.currentIndex() == 1
        qtbot.wait(LONG_WT * 2)
        qtbot.keyClicks(self.up.axes_energy_scale, "binding")
        qtbot.wait(LONG_WT * 2)
        qtbot.mouseClick(self.up.axes_copy_values, Qt.LeftButton)
        qtbot.wait(LONG_WT)

    def check_3Dv_orient_tab(self, qtbot: Any) -> None:
        """
        Check behavior of the features in **Orientate tab** in
        :class:`data_viewer_3d.DataViewer3D`.

        :param qtbot: object emulating a user
        """

        # move to OrientTab and find gamma
        self.up.tabs.setCurrentIndex(3)
        assert self.up.tabs.currentIndex() == 3
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up.orientate_copy_coords, Qt.LeftButton)
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up.orientate_find_gamma, Qt.LeftButton)

        qtbot.mouseClick(self.up.orientate_hor_line, Qt.LeftButton)
        qtbot.mouseClick(self.up.orientate_ver_line, Qt.LeftButton)

        change_spinBox(
            qtbot, self.up.orientate_angle, 181, Qt.Key_Up, time=SHORT_WT * 2
        )
        qtbot.wait(LONG_WT * 2)
        qtbot.mouseClick(self.up.orientate_hor_line, Qt.LeftButton)
        qtbot.mouseClick(self.up.orientate_ver_line, Qt.LeftButton)
        assert self.up.orientate_hor_line.isChecked() is False
        assert self.up.orientate_ver_line.isChecked() is False

        qtbot.mouseClick(self.up.orientate_info_button, Qt.LeftButton)
        qtbot.wait(QTest.qWaitForWindowExposed(self._3dv.info_box))
        qtbot.wait(LONG_WT)
        self._3dv.info_box.close()

    def check_3Dv_image_tab(self, qtbot: Any) -> None:
        """
        Check behavior of the features in **Image tab** in
        :class:`data_viewer_3d.DataViewer3D`.

        :param qtbot: object emulating a user
        """

        # move to ImageTab, normalize data and open 2Dviewer
        self.up.tabs.setCurrentIndex(1)
        assert self.up.tabs.currentIndex() == 1

        qtbot.keyClicks(self.up.image_cmaps, CMAP)
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up.image_invert_colors, Qt.LeftButton)
        qtbot.wait(LONG_WT)
        change_spinBox(qtbot, self.up.image_gamma, N_GAMMA, Qt.Key_Down)

        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up.image_normalize, Qt.LeftButton)
        qtbot.wait(LONG_WT * 2)
        qtbot.keyClicks(self.up.image_normalize_along, "energy")
        assert self.up.image_normalize_along.currentIndex() == 2
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up.image_normalize, Qt.LeftButton)
        qtbot.wait(LONG_WT)

    def open_2Dviewer(self, qtbot: Any) -> None:
        """
        Open a :class:`data_viewer_2d.DataViewer2D` window.

        :param qtbot: object emulating a user
        """

        self.up.image_2dv_cut_selector.setCurrentText("horizontal")
        qtbot.mouseClick(self.up.image_2dv_button, Qt.LeftButton)
        qtbot.wait(LONG_WT)
        self.up.image_2dv_cut_selector.setCurrentText("vertical")
        qtbot.mouseClick(self.up.image_2dv_button, Qt.LeftButton)
        for key in self.browser.data_viewers.keys():
            if ":" in key:
                new_window = key
        assert isinstance(self.browser.data_viewers[new_window], DataViewer2D)
        qtbot.wait(LONG_WT * 3)
        self._2dv = self.browser.data_viewers[new_window]
        self._2dv_title = new_window
        self.up_2dv = self._2dv.util_panel

    def check_2dv_sliders(self, qtbot: Any, linking: bool = False) -> None:
        """
        Check behavior of the :class:`data_viewer_2d.DataViewer2D` sliders.

        :param qtbot: object emulating a user
        :param linking:
        """

        # move sliders
        e_start_idx, k_start_idx = (
            self.up_2dv.energy_vert.value(),
            self.up_2dv.momentum_hor.value(),
        )
        change_spinBox(qtbot, self.up_2dv.energy_vert, N_SLIDER_E, Qt.Key_Down)
        assert self.up_2dv.energy_vert.value() is (e_start_idx - N_SLIDER_E)

        change_spinBox(qtbot, self.up_2dv.momentum_hor, N_SLIDER_K, Qt.Key_Down)
        assert self.up_2dv.momentum_hor.value() is (k_start_idx - N_SLIDER_K)
        qtbot.wait(LONG_WT)

        if not linking:
            # bin one and move it back
            qtbot.mouseClick(self.up_2dv.bin_z, Qt.LeftButton)
        change_spinBox(
            qtbot, self.up_2dv.bin_z_nbins, N_BINS_E, Qt.Key_Down, time=SHORT_WT * 20
        )
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up_2dv.bin_y, Qt.LeftButton)
        change_spinBox(
            qtbot, self.up_2dv.bin_y_nbins, N_BINS_E, Qt.Key_Down, time=SHORT_WT * 2
        )
        change_spinBox(qtbot, self.up_2dv.energy_vert, N_SLIDER_E, Qt.Key_Up)
        assert self.up_2dv.energy_vert.value() is e_start_idx
        qtbot.wait(LONG_WT)

    def check_2dv_image_tab(self, qtbot: Any) -> None:
        """
        Check behavior of the features in **Image tab** in
        :class:`data_viewer_2d.DataViewer2D`.

        :param qtbot: object emulating a user
        """

        # move to ImageTab and change colors
        self.up_2dv.tabs.setCurrentIndex(1)
        assert self.up_2dv.tabs.currentIndex() == 1
        qtbot.wait(LONG_WT)
        qtbot.keyClicks(self.up_2dv.image_cmaps, CMAP)
        assert self.up_2dv.image_cmaps.currentIndex() is CMAP_IDX
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up_2dv.image_invert_colors, Qt.LeftButton)
        assert self.up_2dv.image_invert_colors.isChecked() is True
        qtbot.wait(LONG_WT)
        change_spinBox(
            qtbot, self.up_2dv.image_gamma, N_GAMMA, Qt.Key_Down, time=(LONG_WT // 4)
        )
        assert np.abs(self.up_2dv.image_gamma.value() - 0.5) <= 1e-3
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up_2dv.image_smooth_button, Qt.LeftButton)
        qtbot.mouseClick(self.up_2dv.image_curvature_button, Qt.LeftButton)
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up_2dv.image_curvature_button, Qt.LeftButton)

    def check_2dv_normalization(self, qtbot: Any) -> None:
        """
        Check normalization in :class:`data_viewer_2d.DataViewer2D`.

        :param qtbot: object emulating a user
        """

        # normalize data set along different directions and bring colors
        qtbot.mouseClick(self.up_2dv.image_normalize, Qt.LeftButton)
        # assert self.up_2dv.image_normalize.isChecked() is True
        qtbot.wait(LONG_WT * 2)
        qtbot.keyClicks(self.up_2dv.image_normalize_along, "energy")
        assert self.up_2dv.image_normalize_along.currentIndex() == 1
        qtbot.wait(LONG_WT * 2)
        qtbot.mouseClick(self.up_2dv.image_normalize, Qt.LeftButton)
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up_2dv.image_invert_colors, Qt.LeftButton)
        qtbot.wait(LONG_WT)

    def check_2dv_axes_tab(self, qtbot: Any) -> None:
        """
        Check behavior of the features in **Axes tab** in
        :class:`data_viewer_2d.DataViewer2D`.

        :param qtbot: object emulating a user
        """

        # move to AxesTab, change energy scales,
        # do k-space conversion and reset it
        self.up_2dv.tabs.setCurrentIndex(2)
        assert self.up_2dv.tabs.currentIndex() == 2
        qtbot.wait(LONG_WT)
        qtbot.keyClicks(self.up_2dv.axes_energy_scale, "kinetic")
        assert self.up_2dv.axes_energy_scale.currentIndex() == 1
        qtbot.wait(LONG_WT * 2)
        qtbot.keyClicks(self.up_2dv.axes_energy_scale, "binding")
        qtbot.wait(LONG_WT * 2)
        qtbot.mouseClick(self.up_2dv.axes_gamma_x, Qt.LeftButton)
        qtbot.keyPress(self.up_2dv.axes_gamma_x, Qt.Key_Delete)
        # QTest.keyClicks(up.axes_gamma_x, str(K0_IDX))
        fill_text(qtbot, self.up_2dv.axes_gamma_x, str(K0_IDX))
        qtbot.keyPress(self.up_2dv.axes_gamma_x, Qt.Key_Return)
        assert self.up_2dv.axes_gamma_x.value() == K0_IDX
        qtbot.wait(LONG_WT * 2)
        qtbot.keyClicks(self.up_2dv.axes_slit_orient, "vertical")
        qtbot.wait(LONG_WT * 2)
        self.up_2dv.axes_do_kspace_conv.click()
        assert isinstance(self._2dv.k_axis, np.ndarray)
        qtbot.wait(LONG_WT * 2)

    def check_2dv_file_tab(self, qtbot: Any) -> None:
        """
        Check behavior of the features in **File tab** in
        :class:`data_viewer_2d.DataViewer2D`.

         :param qtbot: object emulating a user
        """

        # move to FileTab, show provenance and metadata windows,
        # add/remove entries
        self.up_2dv.tabs.setCurrentIndex(3)
        assert self.up_2dv.tabs.currentIndex() == 3
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up_2dv.file_show_dp_button, Qt.LeftButton)
        qtbot.wait(LONG_WT * 3)
        self.up_2dv.dp_box.close()

        # reset k-space conversion
        self.up_2dv.axes_reset_conv.click()
        assert self._2dv.k_axis is None

    def open_edc_fitter(self, qtbot: Any) -> None:
        """
        Open :class:`fitters.EDCFitter` window.

        :param qtbot: object emulating a user
        """

        qtbot.mouseClick(self.up_2dv.file_edc_fitter_button, Qt.LeftButton)
        for key in self._2dv.data_viewers.keys():
            if "edc" in key:
                new_window = key
        assert isinstance(self._2dv.data_viewers[new_window], EDCFitter)
        qtbot.wait(LONG_WT * 3)
        self.edc_viewer = self._2dv.data_viewers[new_window]
        self.edc_viewer_title = self.edc_viewer.title

    def check_edc_fitter_range_box_and_binning(self, qtbot: Any) -> None:
        """
        Check behavior of the range box and binning :class:`fitters.EDCFitter`.

        :param qtbot: object emulating a user
        """

        # change range
        range_start = self.edc_viewer.image_edc_range_start.value()
        change_spinBox(
            qtbot, self.edc_viewer.image_edc_range_start, N_SLIDER_E, Qt.Key_Up
        )
        assert self.edc_viewer.image_edc_range_start.value() == (
            range_start + N_SLIDER_E * get_step(self.edc_viewer.data_set.zscale)
        )
        qtbot.wait(LONG_WT)
        # change binning
        qtbot.mouseClick(self.edc_viewer.image_bin, Qt.LeftButton)
        bin_start = self.edc_viewer.image_bin_n.value()
        change_spinBox(qtbot, self.edc_viewer.image_bin_n, 5, Qt.Key_Up)
        assert self.edc_viewer.image_bin_n.value() == (bin_start + 5)
        qtbot.wait(LONG_WT)
        # symmetrize
        qtbot.mouseClick(self.edc_viewer.symmetrize_box, Qt.LeftButton)

    def check_edc_fitter_slider(self, qtbot: Any) -> None:
        """
        Check behavior of the :class:`fitters.EDCFitter` sliders.

        :param qtbot: object emulating a user
        """

        # move sliders and close
        e_start_idx, k_start_idx = (
            self.edc_viewer.image_x_pos.value(),
            self.edc_viewer.image_y_pos.value(),
        )
        change_spinBox(qtbot, self.edc_viewer.image_y_pos, N_SLIDER_K, Qt.Key_Down)
        assert self.edc_viewer.image_y_pos.value() == (k_start_idx - N_SLIDER_K)

        change_spinBox(qtbot, self.edc_viewer.image_x_pos, N_SLIDER_E - 10, Qt.Key_Up)
        assert self.edc_viewer.image_x_pos.value() == (e_start_idx + N_SLIDER_E - 10)
        qtbot.wait(LONG_WT * 3)

        qtbot.mouseClick(self.edc_viewer.image_close_button, Qt.LeftButton)
        assert (self.edc_viewer_title in self._2dv.data_viewers.keys()) is False
        qtbot.wait(LONG_WT)

    def open_2dv_for_metadata_and_summing(self, qtbot: Any):
        """
        Open new 2D viewer for checking data management and MDC fitter, and
        check these functionalities.

        :param qtbot: object emulating a user
        """
        msm_fname = os.path.join(
            os.path.join(resource_filename("piva", "tests"), "data"), "pickle-cut.p"
        )
        self.browser.open_dv(msm_fname)
        self._2dv_msm = self.browser.data_viewers[msm_fname]
        self.up_2dv_msm = self._2dv_msm.util_panel

        # check saving options for 2D viewer
        with patch("PyQt5.QtWidgets.QFileDialog.getExistingDirectory", return_value=""):
            qtbot.mouseClick(self._2dv_msm.util_panel.save_button, Qt.LeftButton)

        # move to FileTab, add and remove some metadata
        self.up_2dv_msm.tabs.setCurrentIndex(3)
        assert self.up_2dv_msm.tabs.currentIndex() == 3
        qtbot.wait(LONG_WT)
        self.up_2dv_msm.file_md_name.setText("test_metadata")
        qtbot.wait(LONG_WT)
        self.up_2dv_msm.file_md_value.setText("123.45")
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up_2dv_msm.file_add_md_button, Qt.LeftButton)
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up_2dv_msm.file_remove_md_button, Qt.LeftButton)
        qtbot.wait(LONG_WT)

        # sum with the same scan, and reset summation
        self.up_2dv_msm.file_sum_datasets_fname.setText("pickle-cut.p")
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up_2dv_msm.file_sum_datasets_sum_button, Qt.LeftButton)
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up_2dv_msm.file_sum_datasets_reset_button, Qt.LeftButton)
        qtbot.wait(LONG_WT)

    def open_mdc_fitter(self, qtbot: Any) -> None:
        """
        Open :class:`fitters.MDCFitter` window.

        :param qtbot: object emulating a user
        """

        qtbot.mouseClick(self._2dv_msm.util_panel.file_mdc_fitter_button, Qt.LeftButton)
        for key in self._2dv_msm.data_viewers.keys():
            if "mdc" in key:
                new_window = key
        assert isinstance(self._2dv_msm.data_viewers[new_window], MDCFitter)
        qtbot.wait(LONG_WT * 3)
        self.mdc_viewer = self._2dv_msm.data_viewers[new_window]
        self.mdc_viewer_title = self.mdc_viewer.title
        qtbot.wait(LONG_WT)

    def check_mdc_fitter_slider(self, qtbot: Any) -> None:
        """
        Check behavior of the :class:`fitters.MDCFitter` sliders.

        :param qtbot: object emulating a user
        """

        # move sliders
        # e_start_idx = self.mdc_viewer.image_y_pos.value()
        change_spinBox(
            qtbot,
            self.mdc_viewer.image_y_pos,
            N_SLIDER_E + 4,
            Qt.Key_Down,
            time=SHORT_WT,
        )
        qtbot.wait(LONG_WT)

    def check_mdc_fitter_ranges_and_fitting(self, qtbot: Any) -> None:
        """
        Check behavior of the ranges and fitting :class:`fitters.MDCFitter`.

        :param qtbot: object emulating a user
        """

        # move to FittingTab and change ranges
        self.mdc_viewer.tabs.setCurrentIndex(1)
        assert self.mdc_viewer.tabs.currentIndex() == 1
        qtbot.wait(LONG_WT)
        # move momentum sliders
        k_start_idx = self.mdc_viewer.image_x_pos.value()
        change_spinBox(qtbot, self.mdc_viewer.image_x_pos, 50, Qt.Key_Down)
        assert self.mdc_viewer.image_x_pos.value() == (k_start_idx - 50)
        # change ranges:
        self.mdc_viewer.fitting_range_stop.setValue(0)
        qtbot.wait(LONG_WT)
        self.mdc_viewer.fitting_bgr_range_first.setValue(-15)
        qtbot.wait(LONG_WT)
        self.mdc_viewer.fitting_bgr_range_second.setValue(-3)
        qtbot.wait(LONG_WT)
        qtbot.keyPress(self.mdc_viewer.fitting_bgr_poly_order, Qt.Key_Up)
        qtbot.mouseClick(self.mdc_viewer.fitting_bgr_poly_button, Qt.LeftButton)
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.mdc_viewer.fitting_button, Qt.LeftButton)
        qtbot.wait(LONG_WT * 3)

        # close
        qtbot.mouseClick(self.mdc_viewer.image_close_button, Qt.LeftButton)
        assert (self.mdc_viewer_title in self._2dv.data_viewers.keys()) is False
        qtbot.wait(LONG_WT)

    def open_plotting_tool(self) -> None:
        """
        Open :class:`plot_tool.PlotTool` window.
        """

        # open plotting tool
        self.browser.open_single_plotting_tool()
        for key in self.browser.plotting_tools.keys():
            if "1" in key:
                self.plot_tool = self.browser.plotting_tools[key]
                self.plot_tool_title = key
        assert isinstance(self.plot_tool, PlotTool)

    def check_plotting_tool(self, qtbot: Any) -> None:
        """
        Check behavior of the basic functionalities of
        :class:`plot_tool.PlotTool`.

        :param qtbot: object emulating a user
        """

        # add first curve
        qtbot.keyClick(self.plot_tool.ds_dv, Qt.Key_Down)
        qtbot.wait(LONG_WT)
        qtbot.keyClick(self.plot_tool.ds_dv, Qt.Key_Return)
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.plot_tool.ds_add_button, Qt.LeftButton)
        qtbot.wait(LONG_WT)

        # add third curve
        qtbot.keyClick(self.plot_tool.ds_dv, Qt.Key_Down)
        qtbot.wait(LONG_WT)
        qtbot.keyClick(self.plot_tool.ds_dv, Qt.Key_Return)
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.plot_tool.ds_add_button, Qt.LeftButton)
        qtbot.mouseClick(self.plot_tool.ds_update_lists, Qt.LeftButton)
        qtbot.wait(LONG_WT * 3)

        # add custom curve and remove it
        qtbot.keyClick(self.plot_tool.ds_dv, Qt.Key_Up)
        qtbot.keyClick(self.plot_tool.ds_dv, Qt.Key_Up)
        qtbot.wait(LONG_WT)
        qtbot.keyClick(self.plot_tool.ds_dv, Qt.Key_Return)
        qtbot.wait(LONG_WT)
        self.plot_tool.ds_custom_name.setText("custom_name")
        qtbot.wait(LONG_WT)
        self.plot_tool.ds_custom_x.setText("-0.2 0")
        self.plot_tool.ds_custom_y.setText("0 10")
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.plot_tool.ds_add_button, Qt.LeftButton)
        qtbot.wait(LONG_WT)
        self.plot_tool.ds_custom_x.setText("-1 0")
        self.plot_tool.main_added.setCurrentText("custom_name")
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.plot_tool.ds_remove_button, Qt.LeftButton)

        # move to EditCurvesTab and normalize one curve
        self.plot_tool.tabs.setCurrentIndex(1)
        assert self.plot_tool.tabs.currentIndex() == 1
        qtbot.wait(LONG_WT)

        qtbot.keyClick(self.plot_tool.main_added, Qt.Key_Down)
        qtbot.wait(LONG_WT)
        qtbot.keyClick(self.plot_tool.main_added, Qt.Key_Return)
        qtbot.wait(LONG_WT)

        qtbot.mouseClick(self.plot_tool.ec_normalize, Qt.LeftButton)
        qtbot.wait(LONG_WT * 3)

        # change x and y offsets
        change_spinBox(qtbot, self.plot_tool.ec_offset_x, 2, Qt.Key_Up)
        change_spinBox(qtbot, self.plot_tool.ec_offset_y, 2, Qt.Key_Up)
        qtbot.wait(LONG_WT)

        # change color
        qtbot.mouseClick(self.plot_tool.ec_color, Qt.LeftButton)
        qtbot.wait(LONG_WT)

        # change line width
        change_spinBox(qtbot, self.plot_tool.ec_width, 5, Qt.Key_Up)
        qtbot.wait(LONG_WT)

        # change line style
        for _ in range(5):
            qtbot.keyClick(self.plot_tool.ec_style, Qt.Key_Down)
            qtbot.keyClick(self.plot_tool.ec_style, Qt.Key_Return)
            qtbot.wait(LONG_WT)

        # reset scaling
        qtbot.mouseClick(self.plot_tool.ec_reset, Qt.LeftButton)
        qtbot.wait(LONG_WT)

        # move to EditPlotTab and change colors
        self.plot_tool.tabs.setCurrentIndex(2)
        assert self.plot_tool.tabs.currentIndex() == 2
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.plot_tool.ep_bgr_color, Qt.LeftButton)
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.plot_tool.ep_axes_color, Qt.LeftButton)
        qtbot.wait(LONG_WT)

        # change ticks font size
        change_spinBox(qtbot, self.plot_tool.ep_ticks_size, 5, Qt.Key_Up)
        qtbot.wait(LONG_WT)

        # set labels and change font size
        self.plot_tool.ep_xlabel.setText("Energy; eV")
        qtbot.wait(LONG_WT)
        self.plot_tool.ep_ylabel.setText("Intensity; a.u.")
        qtbot.wait(LONG_WT)
        change_spinBox(qtbot, self.plot_tool.ep_labels_font_size, 5, Qt.Key_Up)
        qtbot.wait(LONG_WT)

        # move to markers and drop two
        self.plot_tool.tabs.setCurrentIndex(3)
        assert self.plot_tool.tabs.currentIndex() == 3
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.plot_tool.marker_1["button"], Qt.LeftButton)
        qtbot.wait(LONG_WT)
        change_spinBox(qtbot, self.plot_tool.marker_1["x"], 10, Qt.Key_Up)
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.plot_tool.marker_2["button"], Qt.LeftButton)
        qtbot.wait(LONG_WT)
        change_spinBox(qtbot, self.plot_tool.marker_2["x"], -10, Qt.Key_Up)
        qtbot.wait(LONG_WT)

        # move to annotate
        self.plot_tool.ann_name.setText("tmp")
        qtbot.wait(LONG_WT)
        self.plot_tool.ann_text.setText("LABEL")
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.plot_tool.ann_add_update, Qt.LeftButton)
        qtbot.wait(LONG_WT)
        change_spinBox(qtbot, self.plot_tool.ann_y, 10, Qt.Key_Up)
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.plot_tool.ann_add_update, Qt.LeftButton)
        qtbot.wait(LONG_WT)

        # add second annotation and remove the first one
        self.plot_tool.ann_name.setText("tmp2")
        qtbot.wait(LONG_WT)
        self.plot_tool.ann_text.setText("SECOND LABEL")
        qtbot.wait(LONG_WT)
        change_spinBox(qtbot, self.plot_tool.ann_x, 10, Qt.Key_Up)
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.plot_tool.ann_add_update, Qt.LeftButton)
        qtbot.wait(LONG_WT)
        self.plot_tool.ann_added.setCurrentIndex(1)
        self.plot_tool.ann_added.setCurrentIndex(0)
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.plot_tool.ann_remove, Qt.LeftButton)
        qtbot.wait(LONG_WT * 5)

        # save as image and remove it
        self.plot_tool.save_image()
        try:
            os.remove("./test_plottool_image.png")
        except FileNotFoundError:
            pass

        # save as a session and close currect PlotTool window
        self.plot_tool.save_session()
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.plot_tool.main_close_button, Qt.LeftButton)
        assert (self.plot_tool_title in self.browser.plotting_tools.keys()) is False
        qtbot.wait(LONG_WT)

    def check_loaded_plottool_session(self, qtbot: Any):
        """
        Check loading option of the PlotTool.

        :param qtbot: object emulating a user
        """
        # load saved session to the newely opened window,
        # close it and remove the file
        try:
            new_pt = self.browser.plotting_tools["Plotting tool - 2"]
        except KeyError:
            return
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(new_pt.main_load_button, Qt.LeftButton)
        qtbot.wait(LONG_WT * 3)
        qtbot.mouseClick(new_pt.main_close_button, Qt.LeftButton)
        qtbot.wait(LONG_WT)
        try:
            os.remove("./test_plottool_session.p")
        except FileNotFoundError:
            pass

    def open_second_2Dv(self, qtbot: Any) -> None:
        """
        Open second :class:`data_viewer_2d.DataViewer2D` window.

        :param qtbot: object emulating a user
        """

        k_idx = "40"
        self.up.momentum_vert.setValue(int(k_idx))
        qtbot.mouseClick(self.up.image_2dv_button, Qt.LeftButton)
        for key in self.browser.data_viewers.keys():
            if k_idx in key:
                new_window = key
                self._2dv_second_title = key
        assert isinstance(self.browser.data_viewers[new_window], DataViewer2D)
        self._2dv_second = self.browser.data_viewers[new_window]
        self.up_2dv_second = self._2dv_second.util_panel
        qtbot.wait(LONG_WT * 3)

    def check_linking(self, qtbot: Any) -> None:
        """
        Check linking functionality.

        :param qtbot: object emulating a user
        """

        # set the first viewer to parent and link with second
        up = self.up_2dv
        qtbot.mouseClick(up.link_windows_list, Qt.LeftButton)
        up.link_windows_list.setItemCheckState(0, Qt.Checked)
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(up.link_windows, Qt.LeftButton)
        qtbot.wait(LONG_WT * 3)

    def check_kspace_conversion(self, qtbot: Any) -> None:
        """
        Check *k*-space conversion.

        :param qtbot: object emulating a user
        """

        # move back to AxesTab, and do k-space conversion
        self.up.tabs.setCurrentIndex(2)
        assert self.up.tabs.currentIndex() == 2
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up.axes_copy_values, Qt.LeftButton)
        qtbot.wait(LONG_WT * 2)
        qtbot.mouseClick(self.up.axes_do_kspace_conv, Qt.LeftButton)
        qtbot.wait(LONG_WT * 2)

    def check_converted_viewer(self, qtbot: Any) -> None:
        """
        Check saving files.
        """

        for key in self.browser.data_viewers.keys():
            if "rescaled" in key:
                new_window = key
                self._3dv_conv_title = key
        assert isinstance(self.browser.data_viewers[new_window], DataViewer3D)
        self._3dv_conv = self.browser.data_viewers[new_window]
        self.up_3dv_conv = self._3dv_conv.util_panel

        # show metadata window
        qtbot.mouseClick(self.up_3dv_conv.file_show_md_button, Qt.LeftButton)
        qtbot.wait(QTest.qWaitForWindowExposed(self.up_3dv_conv.info_box))
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up_3dv_conv.info_box, Qt.LeftButton)
        change_spinBox(
            qtbot,
            self.up_3dv_conv.info_box.central_widget,
            30,
            Qt.Key_Down,
            time=SHORT_WT * 2,
        )
        qtbot.wait(LONG_WT)
        self.up_3dv_conv.info_box.close()

        # check saving options
        qtbot.mouseClick(self.up_3dv_conv.save_button, Qt.LeftButton)

    def check_BZ_contour(self, qtbot: Any) -> None:
        """
        Check appending Brillouin Zone contour.

        :param qtbot: object emulating a user
        """

        # change to ImageTab and show BZ contour
        self.up_3dv_conv.tabs.setCurrentIndex(1)
        assert self.up_3dv_conv.tabs.currentIndex() == 1
        qtbot.wait(LONG_WT * 2)
        assert self.up_3dv_conv.image_symmetry.value() == 4
        qtbot.wait(LONG_WT * 2)
        qtbot.mouseClick(self.up_3dv_conv.image_show_BZ, Qt.LeftButton)
        qtbot.wait(LONG_WT * 2)

        change_spinBox(qtbot, self.up_3dv_conv.image_rotate_BZ, 90, Qt.Key_Up)
        assert self.up_3dv_conv.image_rotate_BZ.value() == 45

    def open_4d_viewer(self):
        """
        Open 4D viewer.

        :param qtbot: object emulating a user
        """
        _4dv_fname = os.path.join(
            os.path.join(resource_filename("piva", "tests"), "data"), "pickle-raster.p"
        )
        self.browser.open_dv(_4dv_fname)
        self._4dv = self.browser.data_viewers[_4dv_fname]
        self.up_4dv = self._4dv.util_panel

    def check_4dv_volume_tab(self, qtbot: Any):
        """
        Check sliders of the 4D viewer.

        :param qtbot: object emulating a user
        """
        # move sliders
        qtbot.mouseClick(self.up_4dv.bin_y, Qt.LeftButton)
        qtbot.mouseClick(self.up_4dv.bin_z, Qt.LeftButton)
        qtbot.wait(LONG_WT)

        change_spinBox(qtbot, self.up_4dv.rx_vert, 1, Qt.Key_Down)
        qtbot.wait(LONG_WT)
        change_spinBox(qtbot, self.up_4dv.ry_hor, 1, Qt.Key_Up)
        qtbot.wait(LONG_WT)

        change_spinBox(qtbot, self.up_4dv.energy_vert, 10, Qt.Key_Down)
        qtbot.wait(LONG_WT)
        change_spinBox(qtbot, self.up_4dv.momentum_hor, 10, Qt.Key_Up)
        qtbot.wait(LONG_WT)

        qtbot.mouseClick(self.up_4dv.bin_y, Qt.LeftButton)
        qtbot.mouseClick(self.up_4dv.bin_z, Qt.LeftButton)
        qtbot.wait(LONG_WT)

    def check_4dv_image_tab(self, qtbot: Any):
        """
        Check image tab of the 4D viewer.

        :param qtbot: object emulating a user
        """
        # move to image and change colors and normalizations
        self.up_4dv.tabs.setCurrentIndex(1)
        assert self.up_4dv.tabs.currentIndex() == 1
        qtbot.wait(LONG_WT)

        qtbot.keyClicks(self.up_4dv.image_cmaps, CMAP)
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up_4dv.image_invert_colors, Qt.LeftButton)
        qtbot.wait(LONG_WT)
        change_spinBox(qtbot, self.up_4dv.image_gamma, N_GAMMA, Qt.Key_Down)

        qtbot.mouseClick(self.up_4dv.image_normalize, Qt.LeftButton)
        qtbot.wait(LONG_WT * 2)
        qtbot.keyClicks(self.up_4dv.image_normalize_along, "energy")
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up_4dv.image_normalize, Qt.LeftButton)
        qtbot.wait(LONG_WT)

        qtbot.keyClicks(self.up_4dv.image_normalize_along, "signal/noise")
        qtbot.wait(LONG_WT)
        qtbot.keyClicks(self.up_4dv.image_normalize_along, "sharpest edge")
        qtbot.wait(LONG_WT)

    def check_4dv_axes_tab(self, qtbot: Any):
        """
        Check axes tab and k-space convertion of the 4D viewer.

        :param qtbot: object emulating a user
        """
        # move to AxesTab, change energy scales,
        # do k-space conversion and reset it
        self.up_4dv.tabs.setCurrentIndex(2)
        assert self.up_4dv.tabs.currentIndex() == 2
        qtbot.wait(LONG_WT)
        qtbot.keyClicks(self.up_4dv.axes_energy_scale, "kinetic")
        assert self.up_4dv.axes_energy_scale.currentIndex() == 1
        qtbot.wait(LONG_WT)
        self.up_4dv.axes_energy_Ef.setValue(-0.03)
        self.up_4dv.axes_energy_wf.setValue(4.3)
        qtbot.keyClicks(self.up_4dv.axes_energy_scale, "binding")
        qtbot.wait(LONG_WT * 2)
        qtbot.mouseClick(self.up_4dv.axes_gamma_x, Qt.LeftButton)
        qtbot.keyPress(self.up_4dv.axes_gamma_x, Qt.Key_Delete)
        fill_text(qtbot, self.up_4dv.axes_gamma_x, str(100))
        qtbot.keyPress(self.up_4dv.axes_gamma_x, Qt.Key_Return)
        qtbot.wait(LONG_WT * 2)
        qtbot.keyClicks(self.up_4dv.axes_slit_orient, "vertical")
        qtbot.wait(LONG_WT * 2)
        qtbot.mouseClick(self.up_4dv.axes_do_kspace_conv, Qt.LeftButton)
        # self.up_4dv.axes_do_kspace_conv.click()
        qtbot.wait(LONG_WT * 2)
        qtbot.mouseClick(self.up_4dv.axes_reset_conv, Qt.LeftButton)

    def check_4dv_file_tab(self, qtbot: Any):
        """
        Check file tab of the 4D viewer.

        :param qtbot: object emulating a user
        """
        # move to FileTab, show provenance and metadata windows,
        # add/remove entries
        self.up_4dv.tabs.setCurrentIndex(3)
        assert self.up_4dv.tabs.currentIndex() == 3
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up_4dv.file_show_dp_button, Qt.LeftButton)
        qtbot.wait(LONG_WT)
        self.up_4dv.dp_box.close()
        qtbot.wait(LONG_WT)

        self.up_4dv.file_md_name.setText("test_metadata")
        qtbot.wait(LONG_WT)
        self.up_4dv.file_md_value.setText("123.45")
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up_4dv.file_add_md_button, Qt.LeftButton)
        qtbot.wait(LONG_WT)
        qtbot.mouseClick(self.up_4dv.file_remove_md_button, Qt.LeftButton)
        qtbot.wait(LONG_WT)

    def test_viewers(self, qtbot: Any) -> None:
        """
        Run the test.

        :param qtbot: object emulating a user
        """

        self.open_browser(qtbot)
        if CHECK_3D_Viewer_:
            self.check_3Dv_sliders(qtbot)
            self.check_3Dv_orient_tab(qtbot)
            self.check_3Dv_axes_tab(qtbot)
            self.check_3Dv_image_tab(qtbot)

        # open first 2Dviewer
        self.open_2Dviewer(qtbot)

        if CHECK_2D_Viewer_:
            self.check_2dv_sliders(qtbot)
            self.check_2dv_image_tab(qtbot)
            self.check_2dv_normalization(qtbot)
            self.check_2dv_axes_tab(qtbot)
            self.check_2dv_file_tab(qtbot)

        if CHECK_EDC_FITTER:
            self.open_edc_fitter(qtbot)
            self.check_edc_fitter_range_box_and_binning(qtbot)
            self.check_edc_fitter_slider(qtbot)

        if CHECK_METADTA___:
            self.open_2dv_for_metadata_and_summing(qtbot)

        if CHECK_MDC_FITTER:
            self.open_mdc_fitter(qtbot)
            self.check_mdc_fitter_slider(qtbot)
            self.check_mdc_fitter_ranges_and_fitting(qtbot)

        if CHECK_PLOT_TOOL_:
            self.open_plotting_tool()
            self.check_plotting_tool(qtbot)
            self.open_plotting_tool()
            self.check_loaded_plottool_session(qtbot)

        if CHECK_LINKING___:
            self.open_second_2Dv(qtbot)
            self.check_linking(qtbot)
            self.check_2dv_sliders(qtbot, linking=True)
            # close second 2Dviewer
            qtbot.mouseClick(self.up_2dv_second.close_button, Qt.LeftButton)
            assert (self._2dv_second_title in self.browser.data_viewers.keys()) is False
            qtbot.wait(LONG_WT)

        # close first 2Dviewer
        qtbot.mouseClick(self.up_2dv.close_button, Qt.LeftButton)
        assert (self._2dv_title in self.browser.data_viewers.keys()) is False
        qtbot.wait(LONG_WT)

        if CHECK_K_SPC_CONV:
            self.check_kspace_conversion(qtbot)
            self.check_converted_viewer(qtbot)
            self.check_BZ_contour(qtbot)

        if CHECK_4D_Viewer_:
            self.open_4d_viewer()
            self.check_4dv_volume_tab(qtbot)
            self.check_4dv_image_tab(qtbot)
            self.check_4dv_axes_tab(qtbot)
            self.check_4dv_file_tab(qtbot)

        qtbot.wait(LONG_WT * 3)
        to_close = list(self.browser.data_viewers.keys())
        for key in to_close:
            qtbot.mouseClick(
                self.browser.data_viewers[key].util_panel.close_button, Qt.LeftButton
            )
            qtbot.wait(LONG_WT)
        self.browser.close()
        qtbot.wait(LONG_WT * 3)


def change_spinBox(
    bot: Any, widget: Any, steps: int, key: Any, time: int = SHORT_WT
) -> None:
    """
    Useful function for changing spinBoxes in a sequence.

    :param bot: object emulating a user
    :param widget: widget to change values in a sequence
    :param steps: number of steps
    :param key: keyboard key, indicating direction of the change
    :param time: waiting time in [ms]
    """

    for _ in range(steps):
        bot.keyPress(widget, key)
        bot.wait(time)


def fill_text(bot: Any, widget: Any, text: str, time: int = SHORT_WT * 6) -> None:
    """
    Useful function for changing LineEdit widgets in a sequence.

    :param bot: object emulating a user
    :param widget: widget to change values in a sequence
    :param text: text to fill widget with
    :param time: waiting time in [ms]
    """

    for char in text:
        QTest.keyClicks(widget, char)
        bot.wait(time)


if __name__ == "__main__":
    import pytest
    from pkg_resources import resource_filename

    path = os.path.join(resource_filename("piva", "tests"), "viewers_test.py")
    pytest.main(["-v", "-s", path])
