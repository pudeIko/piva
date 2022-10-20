import os

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

import piva.arpys_wp as wp
from piva.cmaps import cmaps, my_cmaps
from piva.imageplot import TracedVariable

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


class MDCFitter(QMainWindow):
    # TODO Smoothing

    def __init__(self, data_viewer, data_set, axes, title, index=None):
        super(MDCFitter, self).__init__()

        self.central_widget = QWidget()
        self.mdc_fitter_layout = QtWidgets.QGridLayout()
        self.central_widget.setLayout(self.mdc_fitter_layout)
        self.tabs = QTabWidget()

        self.settings_panel = QWidget()
        self.cut_panel = PlotWidget(background=MDC_PANEL_BGR)
        self.mdc_panel = PlotWidget(background=MDC_PANEL_BGR)

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
        self.set_mdc_panel()

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
        self.image_bin.stateChanged.connect(self.set_binning_lines)
        self.image_bin_n.valueChanged.connect(self.set_binning_lines)
        self.image_close_button.clicked.connect(self.close)

        self.fitting_button.clicked.connect(self.fit_mdc)
        self.fitting_range_start.valueChanged.connect(self.set_mdc_panel)
        self.fitting_range_stop.valueChanged.connect(self.set_mdc_panel)
        self.fitting_bgr_range_first.valueChanged.connect(self.set_mdc_panel)
        self.fitting_bgr_range_second.valueChanged.connect(self.set_mdc_panel)
        self.fitting_bgr_poly_button.clicked.connect(self.fit_bgr)

        self.fitting_result_append.clicked.connect(self.append_fit_results)
        self.fitting_result_edit.clicked.connect(self.edit_fit_results)
        self.fitting_result_update.clicked.connect(self.load_update_fit_result)
        self.fitting_result_save.clicked.connect(self.save_fit_results)

    def align(self):

        mdl = self.mdc_fitter_layout

        mdl.addWidget(self.tabs,             0, 0, 3, 8)
        mdl.addWidget(self.cut_panel,        4, 0, 4, 4)
        mdl.addWidget(self.mdc_panel,        4, 4, 4, 4)

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

        self.image_bin = QCheckBox('bin k')
        self.image_bin_n = QSpinBox()
        self.image_bin_n.setValue(3)

        self.image_close_button = QPushButton('close')

        row = 0
        itl.addWidget(self.image_e_pos_lbl,         row, 0)
        itl.addWidget(self.image_e_pos,             row, 1)
        itl.addWidget(self.image_e_pos_value_lbl,   row, 2, 1, 2)
        itl.addWidget(self.image_bin,               row, 4)
        itl.addWidget(self.image_bin_n,             row, 5)
        itl.addWidget(self.image_close_button,      row, 7)

        row = 1
        itl.addWidget(self.image_k_pos_lbl,         row, 0)
        itl.addWidget(self.image_k_pos,             row, 1)
        itl.addWidget(self.image_k_pos_value_lbl,   row, 2, 1, 2)

        row = 2
        itl.addWidget(self.image_cmaps_label,       row, 0)
        itl.addWidget(self.image_cmaps,             row, 1, 1, 2)
        itl.addWidget(self.image_invert_colors,     row, 3)
        itl.addWidget(self.image_gamma_label,       row, 4)
        itl.addWidget(self.image_gamma,             row, 5)

        self.setup_cmaps()
        self.image_tab.layout = itl
        self.image_tab.setLayout(itl)
        self.tabs.addTab(self.image_tab, 'Cut')

    def set_fitting_tab(self):
        self.fitting_tab = QWidget()
        ftl = QtWidgets.QGridLayout()

        self.fitting_mu_lbl = QLabel('\u03BC:')
        self.fitting_mu = QDoubleSpinBox()
        self.fitting_mu.setRange(self.k_ax.min(), self.k_ax.max())
        self.fitting_mu.setDecimals(4)
        self.fitting_gamma_lbl = QLabel('\u0393:')
        self.fitting_gamma = QDoubleSpinBox()
        self.fitting_gamma.setValue(0.1)
        self.fitting_mu.setDecimals(4)
        self.fitting_alpha_lbl = QLabel('\u03B1:')
        self.fitting_alpha = QDoubleSpinBox()
        self.fitting_alpha.setValue(1.)
        self.fitting_beta_lbl = QLabel('\u03B2:')
        self.fitting_beta = QDoubleSpinBox()
        self.fitting_beta.setValue(1.)
        self.fitting_a_lbl = QLabel('a:')
        self.fitting_a = QDoubleSpinBox()
        self.fitting_a.setValue(100)
        self.fitting_a.setRange(0, 1e5)
        self.fitting_button = QPushButton('fit MDC')

        self.fitting_message_cell = QLineEdit('Some stuff will show up here.')

        self.fitting_range_lbl = QLabel('fit range:')
        self.fitting_range_lbl.setFont(bold_font)
        # self.fitting_range_start = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.fitting_range_start = QDoubleSpinBox()
        self.fitting_range_start.setRange(self.k_ax.min(), self.k_ax.max())
        self.fitting_range_start.setSingleStep(wp.get_step(self.k_ax))
        self.fitting_range_start.setDecimals(6)
        # self.fitting_range_start.setValue(self.k_ax.min())
        self.fitting_range_start.setValue(-0.84)

        self.fitting_range_stop = QDoubleSpinBox()
        self.fitting_range_stop.setRange(self.k_ax.min(), self.k_ax.max())
        self.fitting_range_stop.setSingleStep(wp.get_step(self.k_ax))
        self.fitting_range_stop.setDecimals(6)
        # self.fitting_range_stop.setValue(self.k_ax.max())
        self.fitting_range_stop.setValue(-0.3)

        self.fitting_bgr_range_lbl = QLabel('bgr range:')
        self.fitting_bgr_range_lbl.setFont(bold_font)
        self.fitting_bgr_range_first = QDoubleSpinBox()
        self.fitting_bgr_range_first.setRange(self.k_ax.min(), self.k_ax.max())
        self.fitting_bgr_range_first.setSingleStep(wp.get_step(self.k_ax))
        self.fitting_bgr_range_first.setDecimals(6)
        # self.fitting_bgr_range_first.setValue(self.k_ax.min())
        self.fitting_bgr_range_first.setValue(-0.66)

        self.fitting_bgr_range_second = QDoubleSpinBox()
        self.fitting_bgr_range_second.setRange(self.k_ax.min(), self.k_ax.max())
        self.fitting_bgr_range_second.setSingleStep(wp.get_step(self.k_ax))
        self.fitting_bgr_range_second.setDecimals(6)
        # self.fitting_bgr_range_second.setValue(self.k_ax.max())
        self.fitting_bgr_range_second.setValue(-0.4)

        self.fitting_bgr_poly_lbl = QLabel('bgr poly:')
        self.fitting_bgr_poly_lbl.setFont(bold_font)
        self.fitting_bgr_poly_order_lbl = QLabel('order')
        self.fitting_bgr_poly_order = QSpinBox()
        self.fitting_bgr_poly_order.setRange(1, 10)
        self.fitting_bgr_poly_button = QPushButton('fit bgr')

        self.fitting_results_lbl = QLabel('results:')
        self.fitting_results_lbl.setFont(bold_font)
        self.fitting_result_append = QPushButton('append')
        self.fitting_result_edit = QPushButton('edit')
        self.fitting_result_update = QPushButton('load/update')
        self.fitting_result_save = QPushButton('save')

        row = 0
        ftl.addWidget(self.fitting_mu_lbl,              row, 0)
        ftl.addWidget(self.fitting_mu,                  row, 1)
        ftl.addWidget(self.fitting_gamma_lbl,           row, 2)
        ftl.addWidget(self.fitting_gamma,               row, 3)
        ftl.addWidget(self.fitting_range_lbl,           row, 5)
        ftl.addWidget(self.fitting_range_start,         row, 6, 1, 2)
        ftl.addWidget(self.fitting_range_stop,          row, 8, 1, 2)

        row = 1
        ftl.addWidget(self.fitting_alpha_lbl,           row, 0)
        ftl.addWidget(self.fitting_alpha,               row, 1)
        ftl.addWidget(self.fitting_beta_lbl,            row, 2)
        ftl.addWidget(self.fitting_beta,                row, 3)
        ftl.addWidget(self.fitting_bgr_range_lbl,       row, 5)
        ftl.addWidget(self.fitting_bgr_range_first,     row, 6, 1, 2)
        ftl.addWidget(self.fitting_bgr_range_second,    row, 8, 1, 2)

        row = 2
        ftl.addWidget(self.fitting_a_lbl,               row, 0)
        ftl.addWidget(self.fitting_a,                   row, 1)
        ftl.addWidget(self.fitting_button,              row, 2, 1, 2)
        ftl.addWidget(self.fitting_bgr_poly_lbl,        row, 5)
        ftl.addWidget(self.fitting_bgr_poly_order_lbl,  row, 6)
        ftl.addWidget(self.fitting_bgr_poly_order,      row, 7)
        ftl.addWidget(self.fitting_bgr_poly_button,     row, 8, 1, 2)

        row = 3
        ftl.addWidget(self.fitting_message_cell,        row, 0, 1, 5)
        ftl.addWidget(self.fitting_results_lbl,         row, 5)
        ftl.addWidget(self.fitting_result_append,       row, 6)
        ftl.addWidget(self.fitting_result_edit,         row, 7)
        ftl.addWidget(self.fitting_result_update,       row, 8)
        ftl.addWidget(self.fitting_result_save,         row, 9)
        #
        # dummy_lbl = QLabel('')
        # ftl.addWidget(dummy_lbl, 2, 0, 1, 8)

        self.fitting_tab.layout = ftl
        self.fitting_tab.setLayout(ftl)
        self.tabs.addTab(self.fitting_tab, 'Fitting')

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
            image = self.data_set.data[0, :, :]
        image_item = ImageItem(image, *args, lut=self.lut, **kwargs)

        self.cut_panel.removeItem(self.image_item)
        self.cut_panel.removeItem(self.mdc_line)
        self.image_item = image_item
        self.image_data = image
        self.cut_panel.addItem(image_item)
        self.cut_panel.addItem(self.mdc_line)
        self.set_edc_line()
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
                self.image_item = ImageItem(image)
            else:
                return
        else:
            self.image_item = image

        self.cut_panel.addItem(self.image_item)
        self.set_ticks(self.k_ax.min(), self.k_ax.max(), 'bottom')
        self.set_ticks(self.erg_ax.min(), self.erg_ax.max(), 'left')
        k_min, k_max, e_min, e_max = 0, self.k_ax.size, 0, self.erg_ax.size
        self.cut_panel.setLimits(xMin=k_min, xMax=k_max, yMin=e_min, yMax=e_max, maxXRange=k_max - k_min,
                                 maxYRange=e_max - e_min)

        self.set_mdc_line()

    def set_mdc_line(self):
        if self.erg_ax.min() < 0:
            e_idx = wp.indexof(0, self.erg_ax)
        else:
            e_idx = self.erg_ax.size // 2
        self.mdc_line = InfiniteLine(e_idx, movable=True, angle=0)
        self.mdc_line.setBounds([1, self.erg_ax.size - 1])
        self.mdc_pos = TracedVariable(e_idx, name='pos')
        self.update_allowed_values(1, self.erg_ax.size - 1)
        self.mdc_pos.sig_value_changed.connect(self.update_position)
        self.mdc_line.sigDragged.connect(self.on_dragged)
        self.mdc_line.setPen(BASE_LINECOLOR)
        self.mdc_line.setHoverPen(HOVER_COLOR)

        self.image_e_pos_value_lbl.setText('({:.4f})'.format(self.erg_ax[int(self.mdc_pos.get_value())]))
        self.image_e_pos.setValue(int(self.mdc_pos.get_value()))
        self.cut_panel.addItem(self.mdc_line)

    def set_edc_line(self):
        try:
            self.cut_panel.removeItem(self.edc_line_cut)
            self.mdc_panel.removeItem(self.edc_line_mdc)
        except AttributeError:
            pass

        k_idx = self.k_ax.size // 2
        self.edc_line_cut = InfiniteLine(k_idx, movable=True, angle=90)
        self.edc_line_mdc = InfiniteLine(self.k_ax[k_idx], movable=True, angle=90)
        self.edc_line_cut.setBounds([1, self.k_ax.size - 1])
        self.edc_line_mdc.setBounds([self.k_ax.min(), self.k_ax.max()])
        self.edc_pos = TracedVariable(k_idx, name='pos')
        self.edc_pos.set_allowed_values(np.arange(1, self.k_ax.size - 1, 1))
        # self.edc_pos.sig_value_changed.connect(self.update_position)
        self.edc_line_cut.sigDragged.connect(self.on_dragged_cut_edc)
        self.edc_line_mdc.sigDragged.connect(self.on_dragged_mdc_edc)
        self.edc_line_cut.setPen(BASE_LINECOLOR)
        self.edc_line_cut.setHoverPen(HOVER_COLOR)
        self.edc_line_mdc.setPen((202, 49, 66))
        self.edc_line_mdc.setHoverPen((240, 149, 115))

        self.image_k_pos_value_lbl.setText('({:.4f})'.format(self.k_ax[int(self.edc_pos.get_value())]))
        self.image_k_pos.setValue(int(self.edc_pos.get_value()))
        self.fitting_mu.setValue(self.edc_line_mdc.value())
        self.cut_panel.addItem(self.edc_line_cut)
        self.mdc_panel.addItem(self.edc_line_mdc)

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
        self.mdc_line.setValue(self.mdc_pos.get_value())
        self.set_mdc_panel()

    def on_dragged(self):
        self.mdc_pos.set_value(self.mdc_line.value())
        self.image_e_pos_value_lbl.setText('({:.4f})'.format(self.erg_ax[int(self.mdc_pos.get_value())]))
        self.image_e_pos.setValue(int(self.mdc_pos.get_value()))
        # if it's an energy plot, and binning option is active, update also binning boundaries
        if self.image_bin.isChecked():
            pos = self.mdc_line.value()
            n = self.image_bin_n.value()
            self.mdc_up_line.setValue(pos + n)
            self.mdc_down_line.setValue(pos - n)

    def on_dragged_cut_edc(self):
        self.edc_pos.set_value(self.edc_line_cut.value())
        self.edc_line_mdc.setValue(self.k_ax[int(self.edc_line_cut.value())])
        self.image_k_pos_value_lbl.setText('({:.4f})'.format(self.k_ax[int(self.edc_pos.get_value())]))
        self.image_k_pos.setValue(int(self.edc_pos.get_value()))
        self.fitting_mu.setValue(self.edc_line_mdc.value())

    def on_dragged_mdc_edc(self):
        self.edc_pos.set_value(wp.indexof(self.edc_line_mdc.value(), self.k_ax))
        self.edc_line_cut.setValue(wp.indexof(self.edc_line_mdc.value(), self.k_ax))
        self.image_k_pos_value_lbl.setText('({:.4f})'.format(self.k_ax[int(self.edc_pos.get_value())]))
        self.image_k_pos.setValue(int(self.edc_pos.get_value()))
        self.fitting_mu.setValue(self.edc_line_mdc.value())

    def update_mdc_slider(self):
        e = self.image_e_pos.value()
        self.mdc_pos.set_value(e)
        self.set_binning_lines()
        self.image_e_pos_value_lbl.setText('({:.4f})'.format(self.erg_ax[int(self.mdc_pos.get_value())]))

    def update_allowed_values(self, min, max):
        """ Update the allowed values silently.
        This assumes that the displayed image is in pixel coordinates and
        sets the allowed values to the available pixels.
        """
        self.mdc_pos.set_allowed_values(np.arange(min, max + 1, 1))

    def set_binning_lines(self):
        if self.image_bin.isChecked():
            try:
                self.cut_panel.removeItem(self.mdc_up_line)
                self.cut_panel.removeItem(self.mdc_down_line)
            except AttributeError:
                pass

            pos = self.mdc_line.value()
            n = self.image_bin_n.value()
            self.mdc_line.setBounds([n, self.erg_ax.size - n])
            self.update_allowed_values(n, self.erg_ax.size - n)
            self.mdc_up_line = InfiniteLine(pos + n, movable=False, angle=0)
            self.mdc_down_line = InfiniteLine(pos - n, movable=False, angle=0)
            self.mdc_up_line.setPen(color=BINLINES_LINECOLOR, width=1)
            self.mdc_down_line.setPen(color=BINLINES_LINECOLOR, width=1)
            self.cut_panel.addItem(self.mdc_up_line)
            self.cut_panel.addItem(self.mdc_down_line)
        else:
            try:
                self.cut_panel.removeItem(self.mdc_up_line)
                self.cut_panel.removeItem(self.mdc_down_line)
                self.mdc_line.setBounds([1, self.erg_ax.size - 1])
                self.update_allowed_values(1, self.erg_ax.size - 1)
            except AttributeError:
                pass

        self.set_mdc_panel()

    def set_mdc_panel(self):
        mdc_plot = self.mdc_panel
        self.fit_result = None
        self.bgr_fit = None

        for di in mdc_plot.listDataItems():
            try:
                mdc_plot.removeItem(di)
            except AttributeError:
                pass

        e_idx = self.mdc_pos.get_value()
        if self.image_bin.isChecked():
            n = self.image_bin_n.value()
            self.mdc = np.sum(self.data_set.data[0, :, (e_idx - n):(e_idx + n)], axis=1)
        else:
            self.mdc = self.data_set.data[0, :, e_idx]

        range_start, range_stop = self.fitting_range_start.value(), self.fitting_range_stop.value()
        bgr_range_f, bgr_range_s = self.fitting_bgr_range_first.value(), self.fitting_bgr_range_second.value()
        self.fitting_range_start.setRange(self.k_ax.min(), range_stop)
        self.fitting_range_stop.setRange(range_start, self.k_ax.max())
        self.fitting_bgr_range_first.setRange(range_start, range_stop)
        self.fitting_bgr_range_second.setRange(range_start, range_stop)

        mdc_plot.plot(self.k_ax, self.mdc, pen=mkPen('k', width=2))
        shadow_bottom = PlotDataItem([range_start, range_stop], [-self.mdc.max(), -self.mdc.max()])
        shadow_top = PlotDataItem([range_start, range_stop], [2 * self.mdc.max(), 2 * self.mdc.max()])

        try:
            mdc_plot.removeItem(self.shadow)
        except AttributeError:
            pass
        self.shadow = FillBetweenItem(shadow_top, shadow_bottom, mkBrush((229, 229, 229)))

        self.bgr_first = PlotDataItem([bgr_range_f, bgr_range_f],
                                      [-self.mdc.max(), self.mdc[wp.indexof(bgr_range_f, self.k_ax)]],
                                      pen=mkPen('k', style=QtCore.Qt.DashLine))
        self.bgr_second = PlotDataItem([bgr_range_s, bgr_range_s],
                                       [-self.mdc.max(), self.mdc[wp.indexof(bgr_range_s, self.k_ax)]],
                                       pen=mkPen('k', style=QtCore.Qt.DashLine))
        mdc_plot.addItem(self.shadow)
        mdc_plot.addItem(self.bgr_first)
        mdc_plot.addItem(self.bgr_second)
        mdc_plot.setYRange(0, self.mdc.max())

    def fit_bgr(self):
        bgr0, bgr1 = self.fitting_range_start.value(), self.fitting_bgr_range_first.value()
        bgr2, bgr3 = self.fitting_bgr_range_second.value(), self.fitting_range_stop.value()
        bgr0, bgr1 = wp.indexof(bgr0, self.k_ax), wp.indexof(bgr1, self.k_ax)
        bgr2, bgr3 = wp.indexof(bgr2, self.k_ax), wp.indexof(bgr3, self.k_ax)

        bgr_k = np.hstack((self.k_ax[bgr0:bgr1], self.k_ax[bgr2:bgr3]))
        bgr_mdc = np.hstack((self.mdc[bgr0:bgr1], self.mdc[bgr2:bgr3]))
        order = self.fitting_bgr_poly_order.value()
        coefs_bgr = np.polyfit(bgr_k, bgr_mdc, order)

        self.fit_k = self.k_ax[bgr0:bgr3]
        self.bgr_fit = np.poly1d(coefs_bgr)(self.fit_k)

        try:
            self.mdc_panel.removeItem(self.bgr)
        except AttributeError:
            pass

        self.bgr = PlotDataItem(self.fit_k, self.bgr_fit, pen=mkPen('c', width=2))
        self.mdc_panel.addItem(self.bgr)

    def fit_mdc(self):

        try:
            self.mdc_panel.removeItem(self.fit)
        except AttributeError:
            pass

        range0 = wp.indexof(self.fitting_range_start.value(), self.k_ax)
        range1 = wp.indexof(self.fitting_range_stop.value(), self.k_ax)
        k_fit = self.k_ax[range0:range1]
        try:
            mdc_fit = self.mdc[range0:range1] - self.bgr_fit
        except AttributeError:
            mdc_fit = self.mdc[range0:range1]
            self.bgr_fit = np.zeros_like(mdc_fit)
        except TypeError:
            mdc_fit = self.mdc[range0:range1]
            self.bgr_fit = np.zeros_like(mdc_fit)
        except ValueError:
            message = 'Fitting range doesn\'t match background size.'
            self.fitting_message_cell.setText(message)
            return

        a, mu, gamma, alpha, beta = self.fitting_a.value(), self.fitting_mu.value(), self.fitting_gamma.value(), \
                                    self.fitting_alpha.value(), self.fitting_beta.value()
        if alpha == 1:
            fit_alpha = False
        else:
            fit_alpha = True
        if beta == 1:
            fit_beta = False
        else:
            fit_beta = True
        self.set_fit_fun(fit_alpha, fit_beta)

        try:
            self.p, cov = curve_fit(self.fit_fun, k_fit, mdc_fit, p0=self.p0)
            message = self.set_fitting_message(fit_alpha, fit_beta)
            self.fitting_message_cell.setText(message)
        except RuntimeWarning:
            message = 'Runtime: Couldn\'t fit with current parameters.'
            self.fitting_message_cell.setText(message)
            return
        except OptimizeWarning:
            message = 'Optimization: Couldn\'t fit with current parameters.'
            self.fitting_message_cell.setText(message)
            return

        self.prepare_fitting_results(fit_alpha, fit_beta)
        res_func = lambda x: self.fit_fun(x, *self.p)
        fit = res_func(k_fit) + self.bgr_fit

        self.fit = PlotDataItem(k_fit, fit, pen=mkPen('m', width=2))
        self.mdc_panel.addItem(self.fit)

    def set_fit_fun(self, fit_alpha, fit_beta):
        self.p0 = [self.fitting_a.value(), self.fitting_mu.value(), self.fitting_gamma.value()]
        if fit_alpha and fit_beta:
            self.fit_fun = lambda x, a0, mu, gamma, alpha, beta: wp.asym_lorentzian(x, a0, mu, gamma, alpha, beta)
            self.p0.append(self.fitting_alpha.value())
            self.p0.append(self.fitting_beta.value())
        elif fit_alpha and not fit_beta:
            self.fit_fun = lambda x, a0, mu, gamma, alpha: wp.asym_lorentzian(x, a0, mu, gamma, alpha=alpha)
            self.p0.append(self.fitting_alpha.value())
        elif not fit_alpha and fit_beta:
            self.fit_fun = lambda x, a0, mu, gamma, beta: wp.asym_lorentzian(x, a0, mu, gamma, beta=beta)
            self.p0.append(self.fitting_beta.value())
        else:
            self.fit_fun = lambda x, a0, mu, gamma: wp.asym_lorentzian(x, a0, mu, gamma)

    def set_fitting_message(self, fit_alpha, fit_beta):
        e = self.erg_ax[int(self.mdc_pos.get_value())]
        message = 'E = {:.4f};  a = {:.1f};  \u03BC = {:.4f};  \u0393 = {:.4f}'.format(e, self.p[0], self.p[1], self.p[2])
        if fit_alpha:
            message += ';  \u03B1 = {:.4f}'.format(self.p[3])
        else:
            message += ';  \u03B1 = -'

        if fit_beta:
            message += ';  \u03B2 = {:.4f}'.format(self.p[-1])
        else:
            message += ';  \u03B2 = -'

        return message

    def prepare_fitting_results(self, fit_alpha, fit_beta):
        e = self.erg_ax[int(self.mdc_pos.get_value())]
        res = [e, self.p[0], self.p[1], self.p[2]]
        if fit_alpha:
            res.append(self.p[3])
        else:
            res.append(1)

        if fit_beta:
            res.append(self.p[-1])
        else:
            res.append(1)
        self.fit_result = np.array(res)

    def append_fit_results(self):

        if self.fit_result is None:
            no_fit_box = QMessageBox()
            no_fit_box.setIcon(QMessageBox.Information)
            no_fit_box.setText('No result to save.')
            no_fit_box.setStandardButtons(QMessageBox.Ok)
            if no_fit_box.exec() == QMessageBox.Ok:
                return

        if self.fit_results is None:
            self.fit_results = self.fit_result
        else:
            self.fit_results = np.vstack((self.fit_results, self.fit_result))

        if len(self.fit_results.shape) == 1:
            pass
        else:
            self.fit_results = np.flip(np.sort(self.fit_results, axis=0), axis=0)

        try:
            self.cut_panel.removeItem(self.fit_points)
        except AttributeError:
            pass

        try:
            fit_points_k = np.array([wp.indexof(ki, self.k_ax) for ki in self.fit_results[:, 2]])
            fit_points_e = np.array([wp.indexof(ei, self.erg_ax) for ei in self.fit_results[:, 0]])
            self.fit_points = ScatterPlotItem(fit_points_k, fit_points_e, symbol='h')
            self.cut_panel.addItem(self.fit_points)
        except IndexError:
            pass

    def save_fit_results(self):

        fname = self.title[:-13] + '_mdc_fit_results.txt'
        if os.path.exists(fname):
            no_fit_box = QMessageBox()
            no_fit_box.setIcon(QMessageBox.Information)
            no_fit_box.setText('File already exists, want to overwrite?')
            no_fit_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            if no_fit_box.exec() == QMessageBox.Cancel:
                return
            else:
                pass

        f = open(fname, 'w+')
        line = 'E [eV]\t\t\ta [a.u.] \t\t mu [1/A]\t\tgamma [1/A]\t\talpha\tbeta [a.u]\n'
        f.write(line)
        try:
            for result in self.fit_results:
                line = ''
                for entry in result:
                    line += str(entry) + '\t'
                f.write(line + '\n')
        except TypeError:
            for entry in self.fit_results:
                line += str(entry) + '\t'
            f.write(line + '\n')
        f.close()

    def edit_fit_results(self):

        fname = self.title[:-13] + '_mdc_fit_results.txt'
        if not os.path.exists(fname):
            no_file_box = QMessageBox()
            no_file_box.setIcon(QMessageBox.Information)
            no_file_box.setText('File not found.')
            no_file_box.setStandardButtons(QMessageBox.Ok)
            if no_file_box.exec() == QMessageBox.Ok:
                return

        os.system('open ' + fname)

    def load_update_fit_result(self):

        fname = self.title[:-13] + '_mdc_fit_results.txt'
        if not os.path.exists(fname):
            no_file_box = QMessageBox()
            no_file_box.setIcon(QMessageBox.Information)
            no_file_box.setText('File not found.')
            no_file_box.setStandardButtons(QMessageBox.Ok)
            if no_file_box.exec() == QMessageBox.Ok:
                return

        overwriting_box = QMessageBox()
        overwriting_box.setIcon(QMessageBox.Question)
        overwriting_box.setText('Fit results will be overwritten, sure to proceed?')
        overwriting_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        if overwriting_box.exec() == QMessageBox.Cancel:
            return
        else:
            pass

        f = open(fname)
        lines = f.readlines()
        self.fit_results = None
        for line in lines[1:]:
            row = np.array([float(entry) for entry in line.split('\t')[:-1]])
            if self.fit_results is None:
                self.fit_results = row
            else:
                self.fit_results = np.vstack((self.fit_results, row))

        f.close()

        try:
            self.cut_panel.removeItem(self.fit_points)
        except AttributeError:
            pass

        try:
            fit_points_k = np.array([wp.indexof(ki, self.k_ax) for ki in self.fit_results[:, 2]])
            fit_points_e = np.array([wp.indexof(ei, self.erg_ax) for ei in self.fit_results[:, 0]])
            self.fit_points = ScatterPlotItem(fit_points_k, fit_points_e, symbol='h')
            self.cut_panel.addItem(self.fit_points)
        except IndexError:
            pass

    def close(self):
        self.destroy()
        self.data_viewer.thread[self.thread_index].quit()
        self.data_viewer.thread[self.thread_index].wait()
        del(self.data_viewer.thread[self.thread_index])
        del(self.data_viewer.data_viewers[self.thread_index])


