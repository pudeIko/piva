from __future__ import annotations
import os
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any
import numpy as np
from pyqtgraph.Qt import QtWidgets
from PyQt5.QtWidgets import QMessageBox, QLabel

import piva.working_procedures as wp
import piva.data_loaders as dl
import piva.image_panels as ip
from piva.utilities_panel import UtilitiesPanel
from piva.cmaps import cmaps
import piva.data_viewer_2d as p2d
from piva.fitters import MDCFitter, EDCFitter

if TYPE_CHECKING:
    from piva.data_browser import DataBrowser

app_style = """
QMainWindow{background-color: rgb(64,64,64);}
QTabWidget{background-color: rgb(64,64,64);}
"""
DEFAULT_CMAP = 'viridis'
ORIENTLINES_LINECOLOR = (164, 37, 22, 255)


class DataHandler4D:
    """
    Object that keeps track of a set of 4D data and allows manipulations on it.
    """

    def __init__(self, main_window: DataViewer4D) -> None:
        """
        Initialize `DataHandler` for :class:`DataViewer2D`.

        :param main_window: related viewer displaying the data
        """

        self.main_window = main_window
        self.binning = False

        # Initialize instance variables
        # np.array that contains the 3D data
        self.data = None
        self.axes = np.array([[0, 1], [0, 1], [0, 1]])
        # Indices of *data* that are displayed in the main plot
        self.displayed_axes = (0, 1)

    def get_data(self) -> np.ndarray:
        """
        Convenient `getter` method.

        :return: 2D array with data
        """

        if self.main_window.util_panel.image_normalize.isChecked():
            return self.norm_data
        else:
            return self.data.get_value()[0, :, :]

    def set_bm_data(self, data: np.ndarray) -> None:
        """
        Set data for the :class:`DataViewer2D`.

        :param data: array with the data
        """

        self.data.set_value(data)

    def prepare_data(self, data: np.ndarray, axes: list = 3 * [None]) -> None:
        """
        Register loaded data and axes and pass them to the
        :class:`DataViewer2D`. Prepare normalized data for quick availability.

        :param data: loaded array of data
        :param axes: loaded list of axes
        """

        # self.data = ip.TracedVariable(data, name='data')
        self.data = ip.CustomTracedVariable(data, name='data')
        self.axes = np.array(axes, dtype="object")
        self.norm_data = wp.normalize(self.data.get_value()[0, :, :].T).T

        self.set_scan_data()
        self.set_raster_data()

        self.prepare_axes()

        self.main_window.set_axes()
        self.main_window.image_data = self.get_data()
        self.main_window.set_bm_image()
        self.main_window.set_raster_image()

    def prepare_axes(self) -> None:
        """
        Prepare loaded axes in order: [*scanned*, *analyzer*, *energy*].
        Here *scanned*  is a single-element array.
        """

        shapes = self.data.get_value().shape
        # Avoid undefined axes scales and replace them with len(1) sequences
        for i, axis in enumerate(self.axes):
            if axis is None:
                self.axes[i] = np.arange(shapes[i])

    def set_scan_data(self):
        """
        Set combined scan data for faster access.
        """

        nx, ny = self.main_window.scan.shape
        ne, nk = self.main_window.scan[0, 0].data[0, :, :].shape
        self.scan_data = np.zeros((nx, ny, ne, nk))
        x_ax, y_ax = [], []
        for xi in range(nx):
            for yi in range(ny):
                self.scan_data[xi, yi] = \
                    self.main_window.scan[xi, yi].data[0, :, :]
                x_ax.append(self.main_window.scan[xi, yi].x)
                y_ax.append(self.main_window.scan[xi, yi].y)
        self.x_ax = np.unique(np.array(x_ax))
        self.y_ax = np.unique(np.array(y_ax))

    def set_raster_data(self):
        """
        Set raster map as a sum individual spectra.
        """

        self.raster_data = np.sum(np.sum(self.scan_data, axis=2), axis=2)

    def update_scan_data(self):
        """
        Update scan_data array after changing normalization option.
        """

        up = self.main_window.util_panel
        nx, ny = self.main_window.scan.shape
        for xi in range(nx):
            for yi in range(ny):
                if up.image_normalize.isChecked():
                    norm_along = up.image_normalize_along.currentIndex()
                    tmp_data = wp.normalize(
                        self.main_window.scan[xi, yi].data[0, :, :],
                        axis=norm_along)
                else:
                    tmp_data = self.main_window.scan[xi, yi].data[0, :, :]
                self.scan_data[xi, yi] = tmp_data

    def update_raster_data(self):
        """
        Update raster map after changing representation method.
        """

        up = self.main_window.util_panel
        # get currently selected method: 0 - sum, 1 - signal/noise ratio,
        # 2 - the sharpest edge
        opt = up.image_raster_options.currentIndex()
        nx, ny = self.main_window.scan.shape
        ne, nk = self.main_window.scan[0, 0].data[0, :, :].shape
        tmp_scan_data = np.zeros((nx, ny, ne, nk))
        for xi in range(nx):
            for yi in range(ny):
                tmp_scan_data[xi, yi] = \
                    self.main_window.scan[xi, yi].data[0, :, :]
        if opt == 0:
            self.raster_data = np.sum(np.sum(tmp_scan_data, axis=2), axis=2)
        elif opt == 1:
            for xi in range(tmp_scan_data.shape[0]):
                for yi in range(tmp_scan_data.shape[1]):
                    flat = tmp_scan_data[xi, yi].flatten() + 1.
                    cut_off = int(flat.size * 0.05)
                    low = np.sort(flat)[:cut_off]
                    high = np.sort(flat)[-cut_off:]
                    self.raster_data[xi, yi] = \
                        np.average(high) / np.average(low)
        elif opt == 2:
            for xi in range(tmp_scan_data.shape[0]):
                for yi in range(tmp_scan_data.shape[1]):
                    edc = np.sum(tmp_scan_data[xi, yi], axis=0)
                    grad = np.gradient(edc)
                    self.raster_data[xi, yi] = grad.max()


class DataViewer4D(QtWidgets.QMainWindow):
    """
    Main window of the 4D data.
    """

    def __init__(self, data_browser: DataBrowser, data_set: dl.Dataset,
                 index: str = None, slice: bool = False) -> None:
        """
        Initialize main window.

        :param data_browser: `DataBrowser` that was used for opening
                             **DataViewer**, for keeping reference to all
                             other opened  **DataViewer**
        :param data_set: loaded dataset with available metadata.
        :param index: title of the window and index in the record of opened
                      **DataViewer**
        :param slice: if :py:obj:`True`, opened
                      :class:`DataViewer2D` is a slice extracted from
                      :class:`~_3Dviewer.DataViewer3D`, that affects behavior
                      of some methods
        """

        super(DataViewer4D, self).__init__()
        self.title = index.split('/')[-1]
        self.fname = index
        self.central_widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QGridLayout()

        # moved to get rid of warnings
        self.cmap = None
        self.image_kwargs = None
        self.cmap_name = None
        self.lut = None
        self.db = data_browser
        self.index = index
        # self.slice = slice
        self.slider_pos = [0, 0]
        self.new_energy_axis = None
        self.k_axis = None
        self.smooth = False
        self.curvature = False
        self.thread = {}
        self.data_viewers = {}

        # Initialize instance variables
        # Plot transparency alpha
        self.alpha = 1
        # Plot powerlaw normalization exponent gamma
        self.gamma = 1
        # Relative colormap maximum
        self.vmax = 1

        # self.data_set = deepcopy(data_set)
        # self.org_dataset = None
        self.data_handler = DataHandler4D(self)

        # raster xy map
        self.raster_plot = ip.ImagePlot(name='bm_plot', crosshair=True)
        # band map
        self.bm_plot = ip.ImagePlot(name='bm_plot', crosshair=True)
        # Create cut of cut_x
        self.plot_x = ip.CurvePlot(name='plot_x')
        # Create cut of cut_y
        self.plot_y = ip.CurvePlot(name='plot_y', orientation='vertical')
        # Create utilities panel
        self.util_panel = UtilitiesPanel(self, name='utilities_panel',
                                         dim=4)
        self.util_panel.positions_momentum_label.setText('Spectrum sliders')
        self.util_panel.bins_label.setText('Integrate spectra')

        self.setStyleSheet(app_style)
        self.set_cmap()

        self.setGeometry(50, 50, 1000, 800)
        self.setWindowTitle(self.title)
        self.initUI()

        # # Set the loaded data
        # if data_set is None:
        #     print('Data set not defined.')
        # else:
        D = data_set[0, 0]
        self.scan = data_set

        self.data_handler.prepare_data(D.data, [D.xscale, D.yscale, D.zscale])
        self.org_image_data = self.data_handler.get_data()

        self.util_panel.energy_vert.setRange(0, len(self.data_handler.axes[2]))
        self.util_panel.momentum_hor.setRange(
            0, len(self.data_handler.axes[1]))

        self.util_panel.rx_vert.setRange(0, self.data_handler.x_ax.size)
        self.util_panel.ry_hor.setRange(0, self.data_handler.y_ax.size)

        try:
            self.load_corrections(self.scan[0, 0])
        except AttributeError:
            pass

        self.set_sliders_initial_positions()

    # initialization methods
    def initUI(self) -> None:
        """
        Initialize widgets by connecting triggered signals to actions.
        """

        self.setWindowTitle(self.title)
        # Create a "central widget" and its layout
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)
        self.util_panel.bin_y_nbins.setValue(10)
        self.util_panel.bin_z_nbins.setValue(5)

        # connect signals to plot_x
        self.plot_x.register_traced_variable(self.bm_plot.pos[0])
        self.plot_x.pos.sig_value_changed.connect(self.update_plot_y)

        # connect signals to plot_y
        self.plot_y.register_traced_variable(self.bm_plot.pos[1])
        self.plot_y.pos.sig_value_changed.connect(self.update_plot_x)

        # binning boxes
        self.util_panel.bin_y.stateChanged.connect(self.update_binning_lines)
        self.util_panel.bin_y_nbins.valueChanged.connect(
            self.update_binning_lines)
        self.util_panel.bin_z.stateChanged.connect(self.update_binning_lines)
        self.util_panel.bin_z_nbins.valueChanged.connect(
            self.update_binning_lines)
        # spectra position boxes
        self.util_panel.energy_vert.valueChanged.connect(
            self.set_vert_slider)
        self.util_panel.momentum_hor.valueChanged.connect(
            self.set_hor_slider)

        # raster position boxes
        self.util_panel.rx_vert.valueChanged.connect(self.set_rx_slider)
        self.util_panel.ry_hor.valueChanged.connect(self.set_ry_slider)

        self.raster_plot.sliders.vpos.sig_value_changed.connect(
            self.set_bm_image)
        self.raster_plot.sliders.hpos.sig_value_changed.connect(
            self.set_bm_image)

        # Create utilities panel and connect signals
        self.util_panel.image_cmaps.currentIndexChanged.connect(self.set_cmap)
        self.util_panel.image_invert_colors.stateChanged.connect(self.set_cmap)
        self.util_panel.image_gamma.valueChanged.connect(self.set_gamma)
        self.util_panel.image_only_spectrum.stateChanged.connect(self.set_cmap)
        self.util_panel.image_normalize.stateChanged.connect(
            self.normalize_data)
        self.util_panel.image_normalize_to.currentIndexChanged.connect(
            self.normalize_data)
        self.util_panel.image_normalize_to.setDisabled(True)
        self.util_panel.image_normalize_along.currentIndexChanged.connect(
            self.normalize_data)
        self.util_panel.image_raster_options.currentIndexChanged.connect(
            self.change_raster_plot)
        self.util_panel.image_2dv_cut_selector.setDisabled(True)
        self.util_panel.image_2dv_button.clicked.connect(self.open_2dviewer)

        # buttons
        self.util_panel.file_mdc_fitter_button.clicked.connect(
            self.open_mdc_fitter)
        self.util_panel.file_edc_fitter_button.clicked.connect(
            self.open_edc_fitter)
        self.util_panel.close_button.clicked.connect(self.close)
        self.util_panel.save_button.clicked.connect(self.save_to_pickle)

        # energy and k-space concersion
        self.util_panel.axes_energy_Ef.valueChanged.connect(
            self.apply_energy_correction)
        self.util_panel.axes_energy_hv.valueChanged.connect(
            self.apply_energy_correction)
        self.util_panel.axes_energy_wf.valueChanged.connect(
            self.apply_energy_correction)
        self.util_panel.axes_energy_scale.currentIndexChanged.connect(
            self.apply_energy_correction)
        self.util_panel.axes_do_kspace_conv.clicked.connect(
            self.convert_to_kspace)
        self.util_panel.axes_reset_conv.clicked.connect(
            self.reset_kspace_conversion)

        self.util_panel.file_add_md_button.setEnabled(False)
        self.util_panel.file_remove_md_button.setEnabled(False)

        # Align all the gui elements
        self._align()
        self.show()

    def _align(self):
        """
        Align all GUI widgets in the window.
        """

        # subdivision
        sd = 1
        # Get a short handle
        l = self.layout
        # utilities bar
        l.addWidget(self.util_panel,    0, 0, 2, 5)
        # raster plot
        l.addWidget(self.raster_plot,    3, 0, 3, 2)
        # Main plot
        l.addWidget(self.bm_plot,     3, 2, 3, 2)
        # EDC
        l.addWidget(self.plot_x,        2, 2, 1, 2)
        # MDC
        l.addWidget(self.plot_y,        3, 4, 3, 1)

        dummy_lbl = QLabel('')
        l.addWidget(dummy_lbl,          2, 0, 1, 2)

        nrows = 6 * sd
        ncols = 5 * sd
        # Need to manually set all row- and columnspans as well as min-sizes
        for i in range(nrows):
            l.setRowMinimumHeight(i, 50)
            l.setRowStretch(i, 1)
        for i in range(ncols):
            l.setColumnMinimumWidth(i, 50)
            l.setColumnStretch(i, 1)

    def set_sliders_initial_positions(self) -> None:
        """
        Set sliders initial positions to either middle of axes or *zeros*.
        """

        # band plot
        if self.new_energy_axis is None:
            e_ax = self.data_handler.axes[2]
        else:
            e_ax = self.new_energy_axis
        if e_ax.min() > 0:
            mid_energy = int(len(e_ax) / 2)
        else:
            mid_energy = wp.indexof(-0.005, e_ax)

        if self.k_axis is None:
            mom_ax = self.data_handler.axes[1]
        else:
            mom_ax = self.k_axis
        if (mom_ax.min() < 0) and (mom_ax.max() > 0):
            mid_angle = wp.indexof(0, mom_ax)
        else:
            mid_angle = int(len(mom_ax) / 2)

        self.bm_plot.pos[1].set_value(mid_energy)
        self.bm_plot.pos[0].set_value(mid_angle)

        # raster plot
        x_mid = self.data_handler.x_ax.size // 2
        y_mid = self.data_handler.y_ax.size // 2
        self.raster_plot.pos[0].set_value(x_mid)
        self.raster_plot.pos[1].set_value(y_mid)

    def load_corrections(self, data_set: dl.Dataset) -> None:
        """
        Load saved energy corrections and axes transformed to *k*-space.

        :param data_set: object containing data and available metadata.
        """

        if not (data_set.Ef is None):
            self.util_panel.axes_energy_Ef.setValue(data_set.Ef)
        if not (data_set.hv is None):
            self.util_panel.axes_energy_hv.setValue(data_set.hv)
        if not (data_set.wf is None):
            self.util_panel.axes_energy_scale.setCurrentIndex(0)
            self.util_panel.axes_energy_wf.setValue(data_set.wf)
        if not (data_set.kyscale is None):
            self.k_axis = data_set.kyscale
            new_range = [self.k_axis[0], self.k_axis[-1]]
            self.bm_plot.plotItem.getAxis(
                self.bm_plot.main_yaxis).setRange(*new_range)
            self.plot_y.plotItem.getAxis(
                self.plot_y.main_xaxis).setRange(*new_range)
            self.util_panel.momentum_hor_value.setText('({:.4f})'.format(
                self.k_axis[self.bm_plot.pos[1].get_value()]))

    @staticmethod
    def swap_axes_aroud(D: np.ndarray) -> np.ndarray:
        """
        Swap axes to plot data with energy scale along *x* axis. This sort of
        breaks the general convention, but only in inside this class.

        :param D: array with originally shaped data
        :return: array with swapped axes
        """

        for xi in range(D.shape[0]):
            for yi in range(D.shape[1]):
                tmp = deepcopy(D[xi, yi])
                D[xi, yi].xscale = tmp.yscale
                D[xi, yi].yscale = tmp.zscale
                D[xi, yi].zscale = tmp.xscale
                D[xi, yi].data = np.ones((1, tmp.zscale.size, tmp.yscale.size))
                D[xi, yi].data[0, :, :] = np.rollaxis(tmp.data[0, :, :], 1)

        return D

    # volume/sliders methods
    def set_axes(self) -> None:
        """
        Set the `x`- and `y`-scales of the plots. The
        :class:`~imageplot.ImagePlot` object takes care of keeping the
        scales as they are, once they are set.
        """

        # bm plot
        xaxis = self.data_handler.axes[1]
        yaxis = self.data_handler.axes[2]
        self.bm_plot.set_xscale(range(0, xaxis.size))
        self.bm_plot.set_ticks(xaxis[0], xaxis[-1],
                                 self.bm_plot.main_xaxis)
        self.plot_x.set_ticks(xaxis[0], xaxis[-1], self.plot_x.main_xaxis)
        self.plot_x.set_secondary_axis(0, xaxis.size)
        self.bm_plot.set_yscale(range(0, yaxis.size))
        self.bm_plot.set_ticks(yaxis[0], yaxis[-1],
                                 self.bm_plot.main_yaxis)
        self.plot_y.set_ticks(yaxis[0], yaxis[-1], self.plot_y.main_xaxis)
        self.plot_y.set_secondary_axis(0, yaxis.size)
        self.bm_plot.fix_viewrange()

        # raster plot
        x_ax_r = self.data_handler.x_ax
        y_ax_r = self.data_handler.y_ax
        self.raster_plot.set_xscale(range(0, x_ax_r.size))
        self.raster_plot.set_ticks(x_ax_r[0], x_ax_r[-1],
                                   self.raster_plot.main_xaxis)
        self.raster_plot.set_yscale(range(0, y_ax_r.size))
        self.raster_plot.set_ticks(y_ax_r[0], y_ax_r[-1],
                                   self.raster_plot.main_yaxis)
        self.raster_plot.fix_viewrange()

    def update_plot_x(self) -> None:
        """
        Update extracted horizontal curve, corresponding to MDC.
        """

        # Get shorthands for plot
        xp = self.plot_x
        try:
            old = xp.listDataItems()[0]
            xp.removeItem(old)
        except IndexError:
            pass

        # Get the correct position indicator
        pos = self.bm_plot.pos[1]
        binning = self.util_panel.bin_y.isChecked()
        if binning:
            width = self.util_panel.bin_y_nbins.value()
        else:
            width = 0
        if pos.allowed_values is not None:
            i_x = int(min(pos.get_value(), pos.allowed_values.max() - 1))
        else:
            i_x = 0

        try:
            data = self.image_data
        except AttributeError:
            data = self.data_handler.get_data()

        if width == 0:
            y = data[:, i_x]
        else:
            start = i_x - width
            stop = i_x + width
            y = np.sum(data[:, start:stop], axis=1)
        x = np.arange(0, len(self.data_handler.axes[1]))
        self.edc = y
        xp.plot(x, y)
        self.util_panel.energy_vert.setValue(i_x)

        if self.new_energy_axis is None:
            self.util_panel.energy_vert_value.setText('({:.4f})'.format(
                self.data_handler.axes[2][i_x]))
        else:
            self.util_panel.energy_vert_value.setText('({:.4f})'.format(
                self.new_energy_axis[i_x]))

        if binning:
            self.plot_y.left_line.setValue(i_x - width)
            self.plot_y.right_line.setValue(i_x + width)

        # if (self.util_panel.link_windows_status.currentIndex() == 1) and \
        #         self.util_panel.get_linked_windows():
        #     linked_windows = self.util_panel.get_linked_windows()
        #     for dvi in self.db.data_viewers.keys():
        #         if self.db.data_viewers[dvi].title in linked_windows:
        #             pos_variable = self.db.data_viewers[dvi].bm_plot.pos[1]
        #             matching_idx = self.get_matching_momentum_idx(
        #                 i_x, self.db.data_viewers[dvi])
        #             pos_variable.set_value(matching_idx)

    def update_plot_y(self) -> None:
        """
        Update extracted horizontal curve, corresponding to EDC.
        """

        # Get shorthands for plot
        yp = self.plot_y
        try:
            old = yp.listDataItems()[0]
            yp.removeItem(old)
        except IndexError:
            pass

        # Get the correct position indicator
        pos = self.bm_plot.pos[0]
        binning = self.util_panel.bin_z.isChecked()
        if binning:
            width = self.util_panel.bin_z_nbins.value()
        else:
            width = 0
        if pos.allowed_values is not None:
            i_y = int(min(pos.get_value(), pos.allowed_values.max() - 1))
        else:
            i_y = 0
        try:
            data = self.image_data
        except AttributeError:
            data = self.data_handler.get_data()
        if width == 0:
            y = data[i_y, :]
        else:
            start = i_y - width
            stop = i_y + width
            y = np.sum(data[start:stop, :], axis=0)
        x = np.arange(0, len(self.data_handler.axes[2]))
        # x = self.data_handler.ang_ax
        self.mdc = y
        yp.plot(y, x)
        self.util_panel.momentum_hor.setValue(i_y)

        if self.k_axis is None:
            self.util_panel.momentum_hor_value.setText('({:.4f})'.format(
                self.data_handler.axes[1][i_y]))
        else:
            self.util_panel.momentum_hor_value.setText('({:.4f})'.format(
                self.k_axis[i_y]))

        if binning:
            self.plot_x.left_line.setValue(i_y - width)
            self.plot_x.right_line.setValue(i_y + width)

        # if (self.util_panel.link_windows_status.currentIndex() == 1) and \
        #         self.util_panel.get_linked_windows():
        #     linked_windows = self.util_panel.get_linked_windows()
        #     for dvi in self.db.data_viewers.keys():
        #         if self.db.data_viewers[dvi].title in linked_windows:
        #             pos_variable = self.db.data_viewers[dvi].bm_plot.pos[0]
        #             matching_idx = self.get_matching_energy_idx(
        #                 i_y, self.db.data_viewers[dvi])
        #             pos_variable.set_value(matching_idx)

    def update_binning_lines(self) -> None:
        """
        Set up/update binning lines indicating integration of the plotted
        curves.
        """

        # hor plot
        if self.util_panel.bin_y.isChecked():
            try:
                half_width = self.util_panel.bin_y_nbins.value()
                pos = self.bm_plot.pos[1].get_value()
                self.bm_plot.add_binning_lines(pos, half_width)
                self.plot_y.add_binning_lines(pos, half_width)
                ymin = 0 + half_width
                ymax = len(self.data_handler.axes[2]) - half_width
                new_range = np.arange(ymin, ymax)
                self.bm_plot.pos[1].set_allowed_values(new_range)
                self.plot_y.pos.set_allowed_values(new_range)
            except AttributeError:
                pass
        else:
            try:
                self.bm_plot.remove_binning_lines()
                self.plot_y.remove_binning_lines()
                org_range = np.arange(0, len(self.data_handler.axes[2]))
                self.bm_plot.pos[1].set_allowed_values(org_range)
                self.plot_y.pos.set_allowed_values(org_range)
            except AttributeError:
                pass

        # vert plot
        if self.util_panel.bin_z.isChecked():
            try:
                half_width = self.util_panel.bin_z_nbins.value()
                pos = self.bm_plot.pos[0].get_value()
                self.bm_plot.add_binning_lines(pos, half_width,
                                                 orientation='vertical')
                self.plot_x.add_binning_lines(pos, half_width)
                ymin = 0 + half_width
                ymax = len(self.data_handler.axes[1]) - half_width
                new_range = np.arange(ymin, ymax)
                self.bm_plot.pos[0].set_allowed_values(new_range)
                self.plot_x.pos.set_allowed_values(new_range)
            except AttributeError:
                pass
        else:
            try:
                self.bm_plot.remove_binning_lines(orientation='vertical')
                self.plot_x.remove_binning_lines()
                org_range = np.arange(0, len(self.data_handler.axes[2]))
                self.bm_plot.pos[0].set_allowed_values(org_range)
                self.plot_x.pos.set_allowed_values(org_range)
            except AttributeError:
                pass

    def set_vert_slider(self) -> None:
        """
        Set position of vertical sliders, moving along the energy direction.
        """

        energy = self.util_panel.energy_vert.value()
        self.bm_plot.pos[1].set_value(energy)
        self.update_binning_lines()

    def set_hor_slider(self) -> None:
        """
        Set position of horizontal sliders, moving along the momentum
        direction.
        """

        angle = self.util_panel.momentum_hor.value()
        self.bm_plot.pos[0].set_value(angle)
        self.update_binning_lines()

    def set_rx_slider(self) -> None:
        """
        Set position of vertical sliders, moving along the energy direction.
        """

        x = self.util_panel.rx_vert.value()
        self.raster_plot.pos[0].set_value(x)

    def set_ry_slider(self) -> None:
        """
        Set position of vertical sliders, moving along the energy direction.
        """

        y = self.util_panel.ry_hor.value()
        self.raster_plot.pos[1].set_value(y)

    def set_sliders_postions(self, positions: list) -> None:
        """
        Set positions of horizontal and vertical sliders.

        :param positions: [`horizontal_position`, `vertical_position`] list
                          of stored positions
        """

        self.bm_plot.pos[1].set_value(positions[0])
        self.bm_plot.pos[0].set_value(positions[1])

    def get_sliders_positions(self) -> list:
        """
        Get position of horizontal and vertical sliders. Useful for to storing
        them before updating the main image, as afterwards they are moved to
        initial positions.

        :return: [`horizontal_position`, `vertical_position`] of the sliders
        """

        main_hor = self.bm_plot.sliders.hpos.get_value()
        main_ver = self.bm_plot.sliders.vpos.get_value()
        return [main_hor, main_ver]

    # image methods
    def set_cmap(self) -> None:
        """
        Set colormap to one of the standard :mod:`matplotlib` cmaps.
        """

        try:
            cmap = self.util_panel.image_cmaps.currentText()
            if self.util_panel.image_invert_colors.isChecked() and ip.MY_CMAPS:
                cmap = cmap + '_r'
        except AttributeError:
            cmap = DEFAULT_CMAP

        try:
            self.cmap = cmaps[cmap]
        except KeyError:
            print('Invalid colormap name. Use one of: ')
            print(cmaps.keys())
        self.cmap_name = cmap
        # Since the cmap changed it forgot our settings for alpha and gamma
        self.cmap.set_alpha(self.alpha)
        self.cmap.set_gamma(self.util_panel.image_gamma.value())
        sliders_pos = self.get_sliders_positions()
        self.cmap_changed()
        self.set_sliders_postions(sliders_pos)
        self.update_binning_lines()

    def cmap_changed(self) -> None:
        """
        Recalculate the lookup table and redraw the plots such that the
        changes are immediately reflected.
        """

        self.lut = self.cmap.getLookupTable()
        self.redraw_plots()

    def redraw_plots(self) -> None:
        """
        Redraw plots to apply changes in data and color scales.
        """

        try:
            self.set_bm_image(displayed_axes=self.data_handler.displayed_axes)
            if not self.util_panel.image_only_spectrum.isChecked():
                self.set_raster_image()
        except AttributeError:
            # In case mainwindow is not defined yet
            pass

    def set_bm_image(self, *args: dict, **kwargs: dict) -> None:
        """
        Wraps underlying :meth:`image_panels.ImagePlot.set_image()` method.

        :param args: additional arguments for
                     :meth:`~image_panels.ImagePlot.set_image`
        :param kwargs: additional keyword arguments for
                       :meth:`~image_panels.ImagePlot.set_image`
        """

        x_pos = self.raster_plot.pos[0].get_value()
        y_pos = self.raster_plot.pos[1].get_value()
        image = self.data_handler.scan_data[x_pos, y_pos, :, :]
        tmp_image = np.zeros((1, image.shape[0], image.shape[1]))
        tmp_image[0, :, :] = image

        self.bm_plot.set_image(image, *args, lut=self.lut, **kwargs)
        self.data_handler.set_bm_data(tmp_image)
        self.image_data = image

        self.util_panel.rx_vert.setValue(x_pos)
        self.util_panel.ry_hor.setValue(y_pos)
        self.util_panel.rx_vert_value.setText('({:.3f})'.format(
            self.data_handler.x_ax[x_pos]))
        self.util_panel.ry_hor_value.setText('({:.3f})'.format(
            self.data_handler.y_ax[y_pos]))

    def set_raster_image(self, *args: dict, **kwargs: dict) -> None:
        """
        Wraps underlying :meth:`image_panels.ImagePlot.set_image()` method.

        :param args: additional arguments for
                     :meth:`~image_panels.ImagePlot.set_image`
        :param kwargs: additional keyword arguments for
                       :meth:`~image_panels.ImagePlot.set_image`
        """

        image = self.data_handler.raster_data
        self.raster_plot.set_image(image, *args, lut=self.lut, **kwargs)

    def set_gamma(self) -> None:
        """
        Set the exponent for the power-law norm that maps the colors to
        values.
        """

        gamma = self.util_panel.image_gamma.value()
        self.gamma = gamma
        sliders_pos = self.get_sliders_positions()
        self.cmap.set_gamma(gamma)
        self.cmap_changed()
        self.set_sliders_postions(sliders_pos)
        self.update_binning_lines()

    def normalize_data(self) -> None:
        """
        Normalize data along selected direction.
        """

        self.data_handler.update_scan_data()
        self.set_bm_image()

    def change_raster_plot(self):
        """
        Change representation of the raster map.
        """

        self.data_handler.update_raster_data()
        self.set_raster_image()

    # axes methods
    def apply_energy_correction(self) -> None:
        """
        Apply saved energy corrections (*e.g.* for offset of the Fermi level)
        and update energy axis between kinetic and binding scales.
        """

        Ef = self.util_panel.axes_energy_Ef.value()

        scale = self.util_panel.axes_energy_scale.currentText()
        if self.data_handler.axes[2].min() > 0:
            org_is_kin = True
        else:
            org_is_kin = False

        if (not org_is_kin) and (scale == 'kinetic'):
            hv = -self.util_panel.axes_energy_hv.value()
            wf = -self.util_panel.axes_energy_wf.value()
        elif org_is_kin and (scale == 'binding'):
            hv = self.util_panel.axes_energy_hv.value()
            wf = self.util_panel.axes_energy_wf.value()
        else:
            hv = 0
            wf = 0

        new_energy_axis = self.data_handler.axes[2] + Ef - hv + wf
        self.new_energy_axis = new_energy_axis
        new_range = [new_energy_axis[0], new_energy_axis[-1]]
        self.bm_plot.plotItem.getAxis(self.bm_plot.main_yaxis).setRange(
            *new_range)
        self.plot_y.plotItem.getAxis(self.plot_y.main_xaxis).setRange(
            *new_range)

        # update energy labels
        erg_idx = self.bm_plot.sliders.hpos.get_value()
        self.util_panel.energy_vert_value.setText('({:.4f})'.format(
            self.new_energy_axis[erg_idx]))

    def convert_to_kspace(self) -> None:
        """
        Convert angles to *k*-space. See
        :func:`~working_procedures.angle2kspace` for more details.
        """

        scan_ax = np.array([0])
        ana_axis = self.data_handler.axes[1]
        d_scan_ax = self.util_panel.axes_angle_off.value()
        d_ana_ax = self.data_handler.axes[1][
            self.util_panel.axes_gamma_x.value()]
        orientation = self.util_panel.axes_slit_orient.currentText()
        a = self.util_panel.axes_conv_lc.value()
        energy = self.new_energy_axis
        hv = self.util_panel.axes_energy_hv.value()
        wf = self.util_panel.axes_energy_wf.value()

        if hv == 0 or wf == 0:
            warning_box = QMessageBox()
            warning_box.setIcon(QMessageBox.Information)
            warning_box.setWindowTitle('Wrong conversion values.')
            if hv == 0 and wf == 0:
                msg = 'Photon energy and work fonction values not given.'
            elif hv == 0:
                msg = 'Photon energy value not given.'
            elif wf == 0:
                msg = 'Work fonction value not given.'
            warning_box.setText(msg)
            warning_box.setStandardButtons(QMessageBox.Ok)
            if warning_box.exec() == QMessageBox.Ok:
                return

        nma, erg = wp.angle2kspace(scan_ax, ana_axis, d_scan_ax=d_scan_ax,
                                   d_ana_ax=d_ana_ax,
                                   orientation=orientation, a=a, energy=energy,
                                   hv=hv, work_func=wf)
        self.k_axis = nma[-1]
        new_range = [nma[-1][0], nma[-1][-1]]
        self.bm_plot.plotItem.getAxis(self.bm_plot.main_xaxis).setRange(
            *new_range)
        self.plot_x.plotItem.getAxis(self.plot_x.main_xaxis).setRange(
            *new_range)
        self.util_panel.momentum_hor_value.setText('({:.4f})'.format(
            self.k_axis[self.bm_plot.pos[0].get_value()]))
        # self.util_panel.dp_add_k_space_conversion_entry(self.data_set)

    def reset_kspace_conversion(self) -> None:
        """
        Reset *k*-space conversion and bring analyzer axis back to [deg].
        """

        self.k_axis = None
        org_range = [self.data_handler.axes[1][0],
                     self.data_handler.axes[1][-1]]
        self.bm_plot.plotItem.getAxis(
            self.bm_plot.main_xaxis).setRange(*org_range)
        self.plot_x.plotItem.getAxis(
            self.plot_x.main_xaxis).setRange(*org_range)
        self.util_panel.momentum_hor_value.setText('({:.4f})'.format(
            self.data_handler.axes[1][self.bm_plot.pos[0].get_value()]))
        # self.data_set.data_provenance['k_space_conv'] = []

    # general functionalities
    def open_2dviewer(self) -> None:
        """
        Open currently displayed cut along selected direction in
        :class:`~_2Dviewer.DataViewer2D`.
        """

        x_pos = self.raster_plot.pos[0].get_value()
        y_pos = self.raster_plot.pos[1].get_value()
        data_set = deepcopy(self.scan[x_pos, y_pos])

        idx = '[{}, {}]'.format(x_pos, y_pos)
        n_bin = '-'
        viewer_idx = '{} [x:{}, y:{}]'.format(self.index, x_pos, y_pos)
        data_set.scan_type = 'cut'
        data_set.scan_dim = []

        if viewer_idx in self.db.data_viewers.keys():
            viewer_running_box = QMessageBox()
            viewer_running_box.setIcon(QMessageBox.Information)
            viewer_running_box.setWindowTitle('Doh.')
            viewer_running_box.setText('Same cut is already opened')
            viewer_running_box.setStandardButtons(QMessageBox.Ok)
            if viewer_running_box.exec() == QMessageBox.Ok:
                return

        try:
            self.util_panel.dp_add_file_cut_entry(data_set, '-', idx, n_bin)
            self.db.data_viewers[viewer_idx] = \
                p2d.DataViewer2D(self.db, data_set=data_set,
                                 index=viewer_idx, slice=True)
        except Exception as e:
            raise e
            error_box = QMessageBox()
            error_box.setIcon(QMessageBox.Information)
            error_box.setText('Couldn\'t load data,  something went wrong.')
            error_box.setStandardButtons(QMessageBox.Ok)
            if error_box.exec() == QMessageBox.Ok:
                return

    def open_mdc_fitter(self) -> None:
        """
        Open current dataset in :class:`~fitters.MDCFitter` to inspect momentum
        distribution curves (MDCs).
        """

        title = self.title + ' - mdc fitter'
        self.mdc_thread_lbl = title

        if title in self.data_viewers:
            already_opened_box = QMessageBox()
            already_opened_box.setIcon(QMessageBox.Information)
            already_opened_box.setText('MDC viewer already opened.')
            already_opened_box.setStandardButtons(QMessageBox.Ok)
            if already_opened_box.exec() == QMessageBox.Ok:
                return

        x_pos = self.raster_plot.pos[0].get_value()
        y_pos = self.raster_plot.pos[1].get_value()
        data_set = deepcopy(self.scan[x_pos, y_pos])

        if self.new_energy_axis is not None:
            erg_ax = self.new_energy_axis
        else:
            erg_ax = data_set.zscale
        if self.k_axis is not None:
            k_ax = self.k_axis
        else:
            k_ax = data_set.yscale
        axes = [k_ax, erg_ax]
        try:
            self.data_viewers[title] = \
                MDCFitter(self, data_set, axes, title)
        except Exception as e:
            error_box = QMessageBox()
            error_box.setIcon(QMessageBox.Information)
            error_box.setText('Couldn\'t load data,  something went wrong.')
            error_box.setStandardButtons(QMessageBox.Ok)
            if error_box.exec() == QMessageBox.Ok:
                raise e
                return

    def open_edc_fitter(self) -> None:
        """
        Open current dataset in :class:`~fitters.EDCFitter` to inspect energy
        distribution curves (EDCs).
        """

        title = self.title + ' - edc fitter'
        self.edc_thread_lbl = title

        if title in self.data_viewers:
            already_opened_box = QMessageBox()
            already_opened_box.setIcon(QMessageBox.Information)
            already_opened_box.setText('EDC viewer already opened.')
            already_opened_box.setStandardButtons(QMessageBox.Ok)
            if already_opened_box.exec() == QMessageBox.Ok:
                return

        x_pos = self.raster_plot.pos[0].get_value()
        y_pos = self.raster_plot.pos[1].get_value()
        data_set = deepcopy(self.scan[x_pos, y_pos])

        if self.new_energy_axis is not None:
            erg_ax = self.new_energy_axis
        else:
            erg_ax = data_set.zscale
        if self.k_axis is not None:
            k_ax = self.k_axis
        else:
            k_ax = data_set.yscale
        axes = [erg_ax, k_ax]
        try:
            self.data_viewers[title] = \
                EDCFitter(self, data_set, axes, title)
        except Exception as e:
            raise e

    def save_to_pickle(self) -> None:
        """
        Save current dataset and all applied corrections to :mod:`pickle` file.
        """

        # energy axis
        dataset = self.scan
        savedir = self.fname[:-len(self.title)]
        file_selection = True
        # Prepare a filename with the .p suffix
        init_fname = '.'.join(self.title.split('.')[:-1] + ['p'])

        while file_selection:
            fname, fname_return_value = \
                QtWidgets.QInputDialog.getText(self, '', 'File name:',
                                               QtWidgets.QLineEdit.Normal,
                                               init_fname)
            if not fname_return_value:
                return

            # check if there is no fname colosions
            if fname in os.listdir(savedir):
                fname_colision_box = QMessageBox()
                fname_colision_box.setIcon(QMessageBox.Question)
                fname_colision_box.setWindowTitle('File name already used.')
                fname_colision_box.setText('File {} already exists.\nDo you '
                                           'want to overwrite '
                                           'it?'.format(fname))
                fname_colision_box.setStandardButtons(QMessageBox.Ok |
                                                      QMessageBox.Cancel)
                if fname_colision_box.exec() == QMessageBox.Ok:
                    file_selection = False
                else:
                    init_fname = fname
            else:
                file_selection = False

        dl.dump(dataset, (savedir + fname), force=True)

    def closeEvent(self, event: Any) -> None:
        """
        Ensure that this instance is closed and un-registered from the
        :class:`~data_browser.DataBrowser`.
        """

        # self.db.delete_viewer_from_linked_lists(self.title)
        del (self.db.data_viewers[self.index])
