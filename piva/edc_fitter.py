
import os
import warnings

import numpy as np
from PyQt5.QtWidgets import QTabWidget, QWidget, QLabel, QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, \
    QLineEdit, QMainWindow, QMessageBox
from PyQt5.QtGui import QFont
from PyQt5 import QtCore
from pyqtgraph.Qt import QtGui, QtWidgets
from pyqtgraph import InfiniteLine, PlotWidget, AxisItem, mkPen, PColorMeshItem, mkBrush, FillBetweenItem, \
    PlotDataItem, ScatterPlotItem
from pyqtgraph.graphicsItems.ImageItem import ImageItem
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning

from piva.imageplot import TracedVariable
import piva.arpys_wp as wp
from piva.cmaps import cmaps, my_cmaps

warnings.filterwarnings("error")

BASE_LINECOLOR = (255, 255, 0, 255)
BINLINES_LINECOLOR = (168, 168, 104, 255)
ORIENTLINES_LINECOLOR = (164, 37, 22, 255)
HOVER_COLOR = (195, 155, 0, 255)
BGR_COLOR = (64, 64, 64)
MDC_PANEL_BGR = (236, 236, 236)
util_panel_style = """
QFrame{margin:5px; border:1px solid rgb(150,150,150);}
QLabel{color: rgb(246, 246, 246); border:1px solid rgb(64, 64, 64);}
QCheckBox{color: rgb(246, 246, 246);}
"""
SIGNALS = 5
MY_CMAPS = True
DEFAULT_CMAP = 'coolwarm'

bold_font = QFont()
bold_font.setBold(True)


class EDCFitter(QMainWindow):
    # TODO Smoothing

    def __init__(self, data_viewer, data_set, axes, title, index=None):
        super(EDCFitter, self).__init__()

        self.central_widget = QWidget()
        self.edc_fitter_layout = QtWidgets.QGridLayout()
        self.central_widget.setLayout(self.edc_fitter_layout)
        self.tabs = QTabWidget()

        self.settings_panel = QWidget()
        self.cut_panel = PlotWidget(background=MDC_PANEL_BGR)
        self.edc_panel = PlotWidget(background=MDC_PANEL_BGR)

        self.data_viewer = data_viewer
        self.data_set = data_set
        self.title = title
        self.thread_index = index
        self.k_ax = axes[0]
        self.erg_ax = axes[1]
        self.fit_results = None

        self.align()
        self.set_settings_panel()
        self.set_cut_panel(data_set.data[0, :, :])
        self.set_edc_panel()

        self.initUI()
        self.set_cmap()
        self.setCentralWidget(self.central_widget)
        self.setWindowTitle(self.title)
        self.show()

    def initUI(self):

        self.image_cmaps.currentIndexChanged.connect(self.set_cmap)
        self.image_invert_colors.stateChanged.connect(self.set_cmap)
        self.image_gamma.valueChanged.connect(self.set_gamma)
        self.image_e_pos.valueChanged.connect(self.update_mdc_slider)
        self.image_k_pos.valueChanged.connect(self.update_edc_slider)
        self.image_bin.stateChanged.connect(self.set_binning_lines)
        self.image_bin_n.valueChanged.connect(self.set_binning_lines)
        self.image_edc_range_start.valueChanged.connect(self.set_edc_panel)
        self.image_edc_range_stop.valueChanged.connect(self.set_edc_panel)
        self.image_close_button.clicked.connect(self.close)

        self.symmetrize_box.stateChanged.connect(self.set_edc_panel)

    def align(self):

        edl = self.edc_fitter_layout

        edl.addWidget(self.tabs,             0, 0, 3, 8)
        edl.addWidget(self.cut_panel,        4, 0, 4, 4)
        edl.addWidget(self.edc_panel,        4, 4, 4, 4)

    def set_settings_panel(self):

        self.set_image_tab()
        self.set_fitting_tab()

    def set_image_tab(self):
        # create elements
        self.image_tab = QWidget()
        itl = QtWidgets.QGridLayout()

        self.image_cmaps_label = QLabel('cmaps:')
        self.image_cmaps = QComboBox()
        self.image_invert_colors = QCheckBox('invert colors')
        self.image_gamma_label = QLabel('gamma:')
        self.image_gamma = QDoubleSpinBox()
        self.image_gamma.setRange(0.05, 10)
        self.image_gamma.setSingleStep(0.05)
        self.image_gamma.setValue(1)

        self.image_e_pos_lbl = QLabel('Energy:')
        self.image_e_pos = QSpinBox()
        self.image_e_pos.setRange(0, self.erg_ax.size)
        self.image_e_pos_value_lbl = QLabel('')

        self.image_k_pos_lbl = QLabel('Momentum:')
        self.image_k_pos = QSpinBox()
        self.image_k_pos.setRange(0, self.k_ax.size)
        self.image_k_pos_value_lbl = QLabel('')

        self.image_bin = QCheckBox('bin E')
        self.image_bin_n = QSpinBox()
        self.image_bin_n.setValue(3)

        self.image_edc_range_lbl = QLabel('EDC range:')
        self.image_edc_range_lbl.setFont(bold_font)
        self.image_edc_range_start = QDoubleSpinBox()
        self.image_edc_range_start.setRange(self.erg_ax.min(), self.erg_ax.max())
        self.image_edc_range_start.setSingleStep(wp.get_step(self.erg_ax))
        self.image_edc_range_start.setDecimals(6)
        # self.image_edc_range_start.setValue(self.erg_ax.min())
        self.image_edc_range_start.setValue(-0.77)

        self.image_edc_range_stop = QDoubleSpinBox()
        self.image_edc_range_stop.setRange(self.erg_ax.min(), self.erg_ax.max())
        self.image_edc_range_stop.setSingleStep(wp.get_step(self.erg_ax))
        self.image_edc_range_stop.setDecimals(6)
        # self.image_edc_range_stop.setValue(self.erg_ax.max())
        self.image_edc_range_stop.setValue(0.1)

        self.image_close_button = QPushButton('close')

        row = 0
        itl.addWidget(self.image_e_pos_lbl,         row, 0)
        itl.addWidget(self.image_e_pos,             row, 1)
        itl.addWidget(self.image_e_pos_value_lbl,   row, 2)
        itl.addWidget(self.image_edc_range_lbl,     row, 5)
        itl.addWidget(self.image_edc_range_start,   row, 6)
        itl.addWidget(self.image_edc_range_stop,    row, 7)
        itl.addWidget(self.image_close_button,      row, 8)

        row = 1
        itl.addWidget(self.image_k_pos_lbl,         row, 0)
        itl.addWidget(self.image_k_pos,             row, 1)
        itl.addWidget(self.image_k_pos_value_lbl,   row, 2)
        itl.addWidget(self.image_bin,               row, 3)
        itl.addWidget(self.image_bin_n,             row, 4)

        row = 2
        itl.addWidget(self.image_cmaps_label,       row, 0)
        itl.addWidget(self.image_cmaps,             row, 1)
        itl.addWidget(self.image_invert_colors,     row, 2)
        itl.addWidget(self.image_gamma_label,       row, 3)
        itl.addWidget(self.image_gamma,             row, 4)

        dummy_lbl = QLabel('')
        itl.addWidget(dummy_lbl, 3, 0, 1, 9)

        self.setup_cmaps()
        self.image_tab.layout = itl
        self.image_tab.setLayout(itl)
        self.tabs.addTab(self.image_tab, 'Image')

    def set_fitting_tab(self):
        self.symmetrize_tab = QWidget()
        stl = QtWidgets.QGridLayout()

        self.symmetrize_box = QCheckBox('symmetrize')

        row = 0
        stl.addWidget(self.symmetrize_box,  row, 0)

        self.symmetrize_tab.layout = stl
        self.symmetrize_tab.setLayout(stl)
        self.tabs.addTab(self.symmetrize_tab, 'Symmetrize')

    def setup_cmaps(self):

        cm = self.image_cmaps
        if MY_CMAPS:
            cm.addItems(my_cmaps)
        else:
            for cmap in cmaps.keys():
                cm.addItem(cmap)
        cm.setCurrentText(DEFAULT_CMAP)

    def set_cmap(self):
        """ Set the colormap to *cmap* where *cmap* is one of the names
        registered in :mod:`<data_slicer.cmaps>` which includes all matplotlib and
        kustom cmaps.
        WP: small changes made to use only my list of cmaps (see cmaps.py) and to shorten the list
        by using 'invert_colors' checkBox
        """
        try:
            cmap = self.image_cmaps.currentText()
            if self.image_invert_colors.isChecked() and MY_CMAPS:
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
        self.cmap.set_alpha(1)
        self.cmap.set_gamma()
        self.cmap_changed()

    def cmap_changed(self, image=None, *args, **kwargs):
        """ Recalculate the lookup table and redraw the plots such that the
        changes are immediately reflected.
        """
        self.lut = self.cmap.getLookupTable()
        self._transform_factors = []
        if image is None:
            image = self.data_set.data[0, :, :].T
        image_item = ImageItem(image, *args, lut=self.lut, **kwargs)

        self.cut_panel.removeItem(self.image_item)
        self.cut_panel.removeItem(self.edc_line)
        self.image_item = image_item
        self.image_data = image
        self.cut_panel.addItem(image_item)
        self.cut_panel.addItem(self.edc_line)
        self.set_mdc_line()
        self.set_binning_lines()

    def set_gamma(self):
        """ Set the exponent for the power-law norm that maps the colors to
        values. I.e. the values where the colours are defined are mapped like
        ``y=x**gamma``.
        WP: changed to work with applied QDoubleSpinBox
        """
        gamma = self.image_gamma.value()
        self.gamma = gamma
        self.cmap.set_gamma(gamma)
        self.cmap_changed()

    def set_cut_panel(self, image):

        # Show top and tight axes by default, but without ticklabels
        self.cut_panel.showAxis('top')
        self.cut_panel.showAxis('right')
        self.cut_panel.getAxis('top').setStyle(showValues=False)
        self.cut_panel.getAxis('right').setStyle(showValues=False)
        self.cut_panel.main_xaxis = 'bottom'
        self.cut_panel.main_xaxis_grid = (255, 1)
        self.cut_panel.main_yaxis = 'left'
        self.cut_panel.main_yaxis_grid = (2, 0)

        # moved here to get rid of warnings:
        self.cut_panel.right_axis = 'top'
        self.cut_panel.angle = 90
        self.cut_panel.slider_axis_index = 1

        # Convert array to ImageItem
        if isinstance(image, np.ndarray):
            if 0 not in image.shape:
                self.image_item = ImageItem(image.T)
            else:
                return
        else:
            self.image_item = image

        self.cut_panel.addItem(self.image_item)
        self.set_ticks(self.k_ax.min(), self.k_ax.max(), 'left')
        self.set_ticks(self.erg_ax.min(), self.erg_ax.max(), 'bottom')
        k_min, k_max, e_min, e_max = 0, self.k_ax.size, 0, self.erg_ax.size
        self.cut_panel.setLimits(xMin=e_min, xMax=e_max, yMin=k_min, yMax=k_max, maxXRange=e_max - e_min,
                                 maxYRange=k_max - k_min)
        self.set_edc_line()

    def set_edc_line(self):
        if self.erg_ax.min() < 0:
            k_idx = wp.indexof(0, self.k_ax)
        else:
            k_idx = self.k_ax.size // 2
        self.edc_line = InfiniteLine(k_idx, movable=True, angle=0)
        self.edc_line.setBounds([1, self.k_ax.size - 1])
        self.edc_pos = TracedVariable(k_idx, name='pos')
        self.update_allowed_values(1, self.k_ax.size - 1)
        self.edc_pos.sig_value_changed.connect(self.update_position)
        self.edc_line.sigDragged.connect(self.on_dragged)
        self.edc_line.setPen(BASE_LINECOLOR)
        self.edc_line.setHoverPen(HOVER_COLOR)

        self.image_k_pos_value_lbl.setText('({:.4f})'.format(self.k_ax[int(self.edc_pos.get_value())]))
        self.image_k_pos.setValue(int(self.edc_pos.get_value()))
        self.cut_panel.addItem(self.edc_line)

    def set_mdc_line(self):
        try:
            self.cut_panel.removeItem(self.mdc_line_cut)
            self.edc_panel.removeItem(self.mdc_line_edc)
        except AttributeError:
            pass

        e_idx = self.erg_ax.size // 2
        self.mdc_line_cut = InfiniteLine(e_idx, movable=True, angle=90)
        self.mdc_line_edc = InfiniteLine(self.erg_ax[e_idx], movable=True, angle=90)
        self.mdc_line_cut.setBounds([1, self.erg_ax.size - 1])
        self.mdc_line_edc.setBounds([self.erg_ax.min(), self.erg_ax.max()])
        self.mdc_pos = TracedVariable(e_idx, name='pos')
        self.mdc_pos.set_allowed_values(np.arange(1, self.erg_ax.size - 1, 1))
        self.mdc_line_cut.sigDragged.connect(self.on_dragged_cut_mdc)
        self.mdc_line_edc.sigDragged.connect(self.on_dragged_edc_mdc)
        self.mdc_line_cut.setPen(BASE_LINECOLOR)
        self.mdc_line_cut.setHoverPen(HOVER_COLOR)
        self.mdc_line_edc.setPen((202, 49, 66))
        self.mdc_line_edc.setHoverPen((240, 149, 115))

        self.image_e_pos_value_lbl.setText('({:.4f})'.format(self.erg_ax[int(self.mdc_pos.get_value())]))
        self.image_e_pos.setValue(int(self.mdc_pos.get_value()))
        self.cut_panel.addItem(self.mdc_line_cut)
        self.edc_panel.addItem(self.mdc_line_edc)

    def set_ticks(self, min_val, max_val, axis):
        """
        Set customized to reflect the dimensions of the physical data
        :param min_val:     float; first axis' value
        :param max_val:     float; last axis' value
        :param axis:        str; axis of which ticks should be put
        """

        plotItem = self.cut_panel.plotItem

        # Remove the old top-axis
        plotItem.layout.removeItem(plotItem.getAxis(axis))
        # Create the new axis and set its range
        new_axis = AxisItem(orientation=axis)
        new_axis.setRange(min_val, max_val)
        plotItem.axes[axis]['item'] = new_axis
        if axis == 'bottom':
            plotItem.layout.addItem(new_axis, *self.cut_panel.main_xaxis_grid)
        else:
            plotItem.layout.addItem(new_axis, *self.cut_panel.main_yaxis_grid)

    def update_position(self):
        self.edc_line.setValue(self.edc_pos.get_value())
        self.set_edc_panel()

    def on_dragged(self):
        self.edc_pos.set_value(self.edc_line.value())
        self.image_k_pos_value_lbl.setText('({:.4f})'.format(self.k_ax[int(self.edc_pos.get_value())]))
        self.image_k_pos.setValue(int(self.edc_pos.get_value()))
        # if it's an energy plot, and binning option is active, update also binning boundaries
        if self.image_bin.isChecked():
            pos = self.edc_line.value()
            n = self.image_bin_n.value()
            self.edc_up_line.setValue(pos + n)
            self.edc_down_line.setValue(pos - n)

    def on_dragged_cut_mdc(self):
        self.mdc_pos.set_value(self.mdc_line_cut.value())
        self.mdc_line_edc.setValue(self.erg_ax[int(self.mdc_line_cut.value())])
        self.image_e_pos_value_lbl.setText('({:.4f})'.format(self.erg_ax[int(self.mdc_pos.get_value())]))
        self.image_e_pos.setValue(int(self.mdc_pos.get_value()))

    def on_dragged_edc_mdc(self):
        self.mdc_pos.set_value(wp.indexof(self.mdc_line_edc.value(), self.erg_ax))
        self.mdc_line_cut.setValue(wp.indexof(self.mdc_line_edc.value(), self.erg_ax))
        self.image_e_pos_value_lbl.setText('({:.4f})'.format(self.erg_ax[int(self.mdc_pos.get_value())]))
        self.image_e_pos.setValue(int(self.mdc_pos.get_value()))

    def update_mdc_slider(self):
        e = self.image_e_pos.value()
        self.mdc_pos.set_value(e)
        self.mdc_line_cut.setValue(e)
        self.mdc_line_edc.setValue(self.erg_ax[e])
        self.set_binning_lines()
        self.image_e_pos_value_lbl.setText('({:.4f})'.format(self.erg_ax[int(self.mdc_pos.get_value())]))

    def update_edc_slider(self):
        k = self.image_k_pos.value()
        self.edc_pos.set_value(k)
        # self.mdc_line_cut.setValue(k)
        # self.mdc_line_edc.setValue(self.erg_ax[e])
        # self.set_binning_lines()
        self.image_k_pos_value_lbl.setText('({:.4f})'.format(self.k_ax[k]))

    def update_allowed_values(self, min, max):
        """ Update the allowed values silently.
        This assumes that the displayed image is in pixel coordinates and
        sets the allowed values to the available pixels.
        """
        self.edc_pos.set_allowed_values(np.arange(min, max + 1, 1))
        # self.image_e_pos.setRange(min, max)

    def set_binning_lines(self):
        if self.image_bin.isChecked():
            try:
                self.cut_panel.removeItem(self.edc_up_line)
                self.cut_panel.removeItem(self.edc_down_line)
            except AttributeError:
                pass

            pos = self.edc_line.value()
            n = self.image_bin_n.value()
            self.edc_line.setBounds([n, self.k_ax.size - n])
            self.update_allowed_values(n, self.k_ax.size - n)
            self.edc_up_line = InfiniteLine(pos + n, movable=False, angle=0)
            self.edc_down_line = InfiniteLine(pos - n, movable=False, angle=0)
            self.edc_up_line.setPen(color=BINLINES_LINECOLOR, width=1)
            self.edc_down_line.setPen(color=BINLINES_LINECOLOR, width=1)
            self.cut_panel.addItem(self.edc_up_line)
            self.cut_panel.addItem(self.edc_down_line)
        else:
            try:
                self.cut_panel.removeItem(self.edc_up_line)
                self.cut_panel.removeItem(self.edc_down_line)
                self.edc_line.setBounds([1, self.k_ax.size - 1])
                self.update_allowed_values(1, self.k_ax.size - 1)
            except AttributeError:
                pass

        self.set_edc_panel()

    def set_edc_panel(self):
        edc_plot = self.edc_panel

        for di in edc_plot.listDataItems():
            try:
                edc_plot.removeItem(di)
            except AttributeError:
                pass

        k_idx = self.edc_pos.get_value()
        e_start, e_stop = self.image_edc_range_start.value(), self.image_edc_range_stop.value()
        e_start, e_stop = wp.indexof(e_start, self.erg_ax), wp.indexof(e_stop, self.erg_ax)
        try:
            self.mdc_line_cut.setBounds([e_start, e_stop])
            self.mdc_line_edc.setBounds([self.erg_ax[e_start], self.erg_ax[e_stop]])
            self.image_e_pos.setRange(e_start, e_stop)
        except AttributeError:
            pass

        if self.image_bin.isChecked():
            n = self.image_bin_n.value()
            self.edc = np.sum(self.data_set.data[0, (k_idx - n):(k_idx + n), e_start:e_stop], axis=0)
        else:
            self.edc = self.data_set.data[0, k_idx, e_start:e_stop]
        self.edc_erg_ax = self.erg_ax[e_start:e_stop]

        if self.symmetrize_box.isChecked():
            self.edc, self.edc_erg_ax = wp.symmetrize_edc(self.edc, self.edc_erg_ax)

        edc_plot.plot(self.edc_erg_ax, self.edc, pen=mkPen('k', width=2))

        for di in self.cut_panel.listDataItems():
            try:
                self.cut_panel.removeItem(di)
            except AttributeError:
                pass

        edc_plot.setYRange(0, self.edc.max())
        edc_plot.setXRange(self.edc_erg_ax.min(), self.edc_erg_ax.max())

        self.edc_line_start = PlotDataItem([e_start, e_start], [0, self.k_ax.size],
                                           pen=mkPen('m', style=QtCore.Qt.DashLine))
        self.edc_line_stop = PlotDataItem([e_stop, e_stop], [0, self.k_ax.size],
                                          pen=mkPen('m', width=2, style=QtCore.Qt.DashLine))
        self.cut_panel.addItem(self.edc_line_start)
        self.cut_panel.addItem(self.edc_line_stop)

    def close(self):
        self.destroy()
        self.data_viewer.thread[self.thread_index].quit()
        self.data_viewer.thread[self.thread_index].wait()
        del(self.data_viewer.thread[self.thread_index])
        del(self.data_viewer.data_viewers[self.thread_index])

