from __future__ import annotations
import os
import numpy as np
from typing import TYPE_CHECKING, Any
from PyQt5.QtWidgets import QTabWidget, QWidget, QLabel, QCheckBox, QComboBox,\
    QDoubleSpinBox, QSpinBox, QPushButton, QLineEdit, QMainWindow, QMessageBox
from PyQt5.QtGui import QFont
from PyQt5 import QtCore
from pyqtgraph.Qt import QtWidgets
from pyqtgraph import InfiniteLine, PlotWidget, AxisItem, mkPen, mkBrush, \
    FillBetweenItem, PlotDataItem, ScatterPlotItem
from pyqtgraph.graphicsItems.ImageItem import ImageItem
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning

import piva.working_procedures as wp
from piva.data_loaders import Dataset
from piva.cmaps import cmaps, my_cmaps
from piva.image_panels import TracedVariable
if TYPE_CHECKING:
    from piva.data_viewer_2d import DataViewer2D


BASE_LINECOLOR = (255, 255, 0, 255)
BINLINES_LINECOLOR = (168, 168, 104, 255)
ORIENTLINES_LINECOLOR = (164, 37, 22, 255)
HOVER_COLOR = (195, 155, 0, 255)
BGR_COLOR = (64, 64, 64)
PANEL_BGR = (236, 236, 236)
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


class Fitter(QMainWindow):
    """
    Generic class for **Fitters**, aligning panels and shared widgets.
    Specific **Fitters** for MDC and EDC curves inherit from it.
    """

    def __init__(self, data_viewer: DataViewer2D, data_set: Dataset,
                 axes: list, title: str) -> None:
        """
        Create **setting, Image and DC (distribution curve) panels** and
        initialize basic functionalities.

        :param data_viewer: reference to the parent :class:`DataViewer2D` to
                            access its data
        :param data_set: object containing all available data and metadata
        :param axes: list of [*momentum*, *energy*] axes
        :param title: title of the window and index to store withing parent's
                      record of opened **DataViewers**/**Fitters**
        """

        super(Fitter, self).__init__()

        self.central_widget = QWidget()
        self.fitter_layout = QtWidgets.QGridLayout()
        self.central_widget.setLayout(self.fitter_layout)
        self.tabs = QTabWidget()

        self.settings_panel = QWidget()
        self.cut_panel = PlotWidget(background=PANEL_BGR)
        self.dc_panel = PlotWidget(background=PANEL_BGR)

        self.data_viewer = data_viewer
        self.data_set = data_set
        self.data = data_set.data[0, :, :]
        self.title = title
        self.x_axis = axes[0]
        self.y_axis = axes[1]

        self.align()

        self.setCentralWidget(self.central_widget)
        self.setWindowTitle(self.title)

    def align(self) -> None:
        """
        Align all GUI widgets in the window.
        """

        mdl = self.fitter_layout

        mdl.addWidget(self.tabs,             0, 0, 3, 8)
        mdl.addWidget(self.cut_panel,        4, 0, 4, 4)
        mdl.addWidget(self.dc_panel,        4, 4, 4, 4)

    def set_image_tab(self) -> None:
        """
        Create and align widgets in the **Image tab** of the settings panel.
        """

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

        self.image_x_pos_lbl = QLabel('')
        self.image_x_pos = QSpinBox()
        self.image_x_pos.setRange(0, self.x_axis.size)
        self.image_x_pos_value_lbl = QLabel('')

        self.image_y_pos_lbl = QLabel('')
        self.image_y_pos = QSpinBox()
        self.image_y_pos.setRange(0, self.y_axis.size)
        self.image_y_pos_value_lbl = QLabel('')

        self.image_close_button = QPushButton('close')

        row = 0
        itl.addWidget(self.image_y_pos_lbl,         row, 0)
        itl.addWidget(self.image_y_pos,             row, 1)
        itl.addWidget(self.image_y_pos_value_lbl,   row, 2)
        itl.addWidget(self.image_close_button,      row, 8)

        row = 1
        itl.addWidget(self.image_x_pos_lbl,         row, 0)
        itl.addWidget(self.image_x_pos,             row, 1)
        itl.addWidget(self.image_x_pos_value_lbl,   row, 2)

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

    def setup_cmaps(self) -> None:
        """
        Set up :class:`QComboBox` with color maps.
        """

        cm = self.image_cmaps
        if MY_CMAPS:
            cm.addItems(my_cmaps)
        else:
            for cmap in cmaps.keys():
                cm.addItem(cmap)
        cm.setCurrentText(DEFAULT_CMAP)

    def set_cmap(self) -> None:
        """
        Set colormap to one of the standard :mod:`matplotlib` cmaps.
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

    def cmap_changed(self, image: np.ndarray = None, *args: dict,
                     **kwargs: dict) -> None:
        """
        Recalculate the lookup table and redraw the plots such that the
        changes are immediately reflected.

        :param image: array with a data
        :param args: additional arguments
        :param kwargs: additional keyword arguments
        """

        self.lut = self.cmap.getLookupTable()
        self._transform_factors = []
        if image is None:
            image = self.data
        image_item = ImageItem(image, *args, lut=self.lut, **kwargs)

        self.cut_panel.removeItem(self.image_item)
        self.cut_panel.removeItem(self.hor_line)
        self.image_item = image_item
        self.image_data = image
        self.cut_panel.addItem(image_item)
        self.cut_panel.addItem(self.hor_line)
        self.set_ver_line()
        self.set_binning_lines()
        self.update_labels()

    def set_gamma(self) -> None:
        """
        Set the exponent for the power-law norm that maps the colors to values.
        """

        gamma = self.image_gamma.value()
        self.gamma = gamma
        self.cmap.set_gamma(gamma)
        self.cmap_changed()

    def set_cut_panel(self, image: np.ndarray) -> None:
        """
        Set up a **graphic panel** with spectra.

        :param image: array with data values
        """

        # Show top and right axes by default, but without ticklabels
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
        self.set_ticks(self.x_axis.min(), self.x_axis.max(), 'bottom')
        self.set_ticks(self.y_axis.min(), self.y_axis.max(), 'left')
        x_min, x_max, y_min, y_max = 0, self.x_axis.size, 0, self.y_axis.size
        self.cut_panel.setLimits(xMin=x_min, xMax=x_max, yMin=y_min,
                                 yMax=y_max, maxXRange=x_max - x_min,
                                 maxYRange=y_max - y_min)

        self.set_hor_line()

    def set_hor_line(self) -> None:
        """
        Set up the horizontal sliders.
        """

        if self.y_axis.min() < 0:
            y_idx = wp.indexof(0, self.y_axis)
        else:
            y_idx = self.y_axis.size // 2
        self.hor_line = InfiniteLine(y_idx, movable=True, angle=0)
        self.hor_line.setBounds([1, self.y_axis.size - 1])
        self.hor_pos = TracedVariable(y_idx, name='pos')
        self.update_allowed_values(1, self.y_axis.size - 1)
        self.hor_pos.sig_value_changed.connect(self.update_position)
        self.hor_line.sigDragged.connect(self.on_dragged)
        self.hor_line.setPen(BASE_LINECOLOR)
        self.hor_line.setHoverPen(HOVER_COLOR)

        self.image_x_pos.setValue(int(self.hor_pos.get_value()))
        self.cut_panel.addItem(self.hor_line)
        self.update_labels()

    def set_ver_line(self) -> None:
        """
        Set up the vertical sliders.
        """

        try:
            self.cut_panel.removeItem(self.ver_line_cut)
            self.dc_panel.removeItem(self.ver_line_dc)
        except AttributeError:
            pass

        k_idx = self.x_axis.size // 2
        self.ver_line_cut = InfiniteLine(k_idx, movable=True, angle=90)
        self.ver_line_dc = InfiniteLine(self.x_axis[k_idx], movable=True,
                                        angle=90)
        self.ver_line_cut.setBounds([1, self.x_axis.size - 1])
        self.ver_line_dc.setBounds([self.x_axis.min(), self.x_axis.max()])
        self.ver_pos = TracedVariable(k_idx, name='pos')
        self.ver_pos.set_allowed_values(np.arange(1, self.x_axis.size - 1, 1))
        self.ver_pos.sig_value_changed.connect(self.update_position)
        self.ver_line_cut.sigDragged.connect(self.on_dragged_cut_ver)
        self.ver_line_dc.sigDragged.connect(self.on_dragged_dc_ver)
        self.ver_line_cut.setPen(BASE_LINECOLOR)
        self.ver_line_cut.setHoverPen(HOVER_COLOR)
        self.ver_line_dc.setPen((202, 49, 66))
        self.ver_line_dc.setHoverPen((240, 149, 115))

        self.cut_panel.addItem(self.ver_line_cut)
        self.dc_panel.addItem(self.ver_line_dc)
        self.update_labels()

    def set_sliders_initial_pos(self) -> None:
        """
        Set sliders initial positions to either middle of axes or *zeros*.
        """

        if self.x_axis.min() < 0:
            x_idx = wp.indexof(0, self.x_axis)
        else:
            x_idx = self.x_axis.size // 2
        self.ver_pos.set_value(x_idx)

        if self.y_axis.min() < 0:
            y_idx = wp.indexof(0, self.y_axis)
        else:
            y_idx = self.y_axis.size // 2
        self.hor_pos.set_value(y_idx)

    def set_ticks(self, min_val: float, max_val: float, axis: str) -> None:
        """
        Set ticks to reflect the dimensions of the physical data.

        :param min_val:     first value
        :param max_val:     last value
        :param axis:        concerned axis ('top', 'left', *etc.*)
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

    def update_position(self) -> None:
        """
        Change positions of the sliders and update related widgets.
        """

        self.hor_line.setValue(self.hor_pos.get_value())
        self.ver_line_cut.setValue(self.ver_pos.get_value())
        self.ver_line_dc.setValue(self.x_axis[self.ver_pos.get_value()])
        self.image_x_pos.setValue(self.ver_pos.get_value())
        self.image_y_pos.setValue(self.hor_pos.get_value())
        self.set_dc_panel()

    def on_dragged(self) -> None:
        """
        Update **cut panle** and displayed values if horizontal sliders was
        dragged.
        """

        self.hor_pos.set_value(self.hor_line.value())
        self.image_y_pos.setValue(int(self.hor_pos.get_value()))
        # update also binning boundaries
        if self.image_bin.isChecked():
            pos = self.hor_line.value()
            n = self.image_bin_n.value()
            self.hor_up_line.setValue(pos + n)
            self.hor_down_line.setValue(pos - n)
        self.update_labels()

    def on_dragged_cut_ver(self) -> None:
        """
        Update displayed values and vertical sliders on **DC panel** if
        vertical sliders was dragged in **Image panel**.
        """

        self.ver_pos.set_value(self.ver_line_cut.value())
        self.ver_line_dc.setValue(self.x_axis[int(self.ver_line_cut.value())])
        self.image_x_pos.setValue(int(self.ver_pos.get_value()))
        self.update_labels()

    def on_dragged_dc_ver(self) -> None:
        """
        Update displayed values and vertical sliders on **Image panel** if
        vertical sliders was dragged in **DC panel**.
        """

        self.ver_pos.set_value(wp.indexof(self.ver_line_dc.value(),
                                          self.x_axis))
        self.ver_line_cut.setValue(wp.indexof(self.ver_line_dc.value(),
                                              self.x_axis))
        self.image_x_pos.setValue(int(self.ver_pos.get_value()))
        self.update_labels()

    def update_ver_slider(self) -> None:
        """
        Update position and displayed values of the vertical sliders.
        """

        x = self.image_x_pos.value()
        self.ver_pos.set_value(x)
        self.ver_line_cut.setValue(x)
        self.ver_line_dc.setValue(self.x_axis[x])
        self.set_binning_lines()
        self.update_labels()

    def update_hor_slider(self) -> None:
        """
        Update position and displayed values of the horizontal sliders.
        """

        y = self.image_y_pos.value()
        self.hor_pos.set_value(y)
        self.set_binning_lines()
        self.update_labels()

    def update_allowed_values(self, min: int, max: int) -> None:
        """
        Update allowed values of the horizontal sliders if binning is enabled.
        This assumes they *live* in pixel coordinates and sets the allowed
        values to the available pixels.

        :param min: new minimal value fo the sliders's range
        :param max: new maximal value fo the sliders's range
        """

        self.hor_pos.set_allowed_values(np.arange(min, max + 1, 1))

    def set_binning_lines(self) -> None:
        """
        Display/update binning lines showed in the **Image panel**.
        """

        if self.image_bin.isChecked():
            try:
                self.cut_panel.removeItem(self.hor_up_line)
                self.cut_panel.removeItem(self.hor_down_line)
            except AttributeError:
                pass

            pos = self.hor_line.value()
            n = self.image_bin_n.value()
            self.hor_line.setBounds([n, self.y_axis.size - n])
            self.update_allowed_values(n, self.y_axis.size - n)
            self.hor_up_line = InfiniteLine(pos + n, movable=False, angle=0)
            self.hor_down_line = InfiniteLine(pos - n, movable=False, angle=0)
            self.hor_up_line.setPen(color=BINLINES_LINECOLOR, width=1)
            self.hor_down_line.setPen(color=BINLINES_LINECOLOR, width=1)
            self.cut_panel.addItem(self.hor_up_line)
            self.cut_panel.addItem(self.hor_down_line)
        else:
            try:
                self.cut_panel.removeItem(self.hor_up_line)
                self.cut_panel.removeItem(self.hor_down_line)
                self.hor_line.setBounds([1, self.y_axis.size - 1])
                self.update_allowed_values(1, self.y_axis.size - 1)
            except AttributeError:
                pass

        self.set_dc_panel()

    def set_dc_panel(self) -> None:
        """
        Defined here for initialization, needs to be overriden in inheriting
        **Fitter**.
        """

        pass

    def update_labels(self) -> None:
        """
        Defined here for initialization, needs to be overriden in inheriting
        **Fitter**.
        """

        pass

    def closeEvent(self, event: Any) -> None:
        """
        Ensure that this instance is closed and un-registered from the
        :class:`~DataViewer2D`.
        """

        del(self.data_viewer.data_viewers[self.title])


class MDCFitter(Fitter):
    """
    Utility for fitting momentum distribution curves (MDCs) from the acquired
    ARPES spectra.
    Allows for specifying region of interest, subtract background (as a
    polynomial) and fitting a Lorentzian curve (see
    :func:`~working_procedures.lorentzian` for more details).
    In addition, one can account for asymmetry parameters originating from
    `e.g.` replica bands.
    """

    def __init__(self, data_viewer: DataViewer2D, data_set: Dataset,
                 axes: list, title: str) -> None:
        """
        Initialize fitter window.

        :param data_viewer: reference to the parent :class:`DataViewer2D` to
                            access its data
        :param data_set: object containing all available data and metadata
        :param axes: list of [*momentum*, *energy*] axes
        :param title: title of the window and index to store withing parent's
                      record of opened **DataViewers**/**Fitters**
        """

        super(MDCFitter, self).__init__(data_viewer, data_set, axes, title)

        self.fit_results = None

        self.set_settings_panel()
        self.set_cut_panel(self.data)
        self.set_dc_panel()
        self.set_cmap()

        self.initUI()
        self.set_sliders_initial_pos()
        self.show()

    def initUI(self) -> None:
        """
        Initialize widgets by connecting triggered signals to actions.
        """

        self.image_cmaps.currentIndexChanged.connect(self.set_cmap)
        self.image_invert_colors.stateChanged.connect(self.set_cmap)
        self.image_gamma.valueChanged.connect(self.set_gamma)
        self.image_y_pos.valueChanged.connect(self.update_hor_slider)
        self.image_bin.stateChanged.connect(self.set_binning_lines)
        self.image_bin_n.valueChanged.connect(self.set_binning_lines)
        self.image_x_pos.valueChanged.connect(self.update_ver_slider)
        self.image_close_button.clicked.connect(self.close)

        self.fitting_button.clicked.connect(self.fit_mdc)
        self.fitting_range_start.valueChanged.connect(self.set_dc_panel)
        self.fitting_range_stop.valueChanged.connect(self.set_dc_panel)
        self.fitting_bgr_range_first.valueChanged.connect(self.set_dc_panel)
        self.fitting_bgr_range_second.valueChanged.connect(self.set_dc_panel)
        self.fitting_bgr_poly_button.clicked.connect(self.fit_bgr)

        self.fitting_result_append.clicked.connect(self.append_fit_results)
        self.fitting_result_edit.clicked.connect(self.edit_fit_results)
        self.fitting_result_update.clicked.connect(self.load_update_fit_result)
        self.fitting_result_save.clicked.connect(self.save_fit_results)

    def set_settings_panel(self) -> None:
        """
        Set tabs in the **settings panel**.
        """

        self.set_image_tab()
        self.customize_image_tab()
        self.set_fitting_tab()

    def customize_image_tab(self) -> None:
        """
        Create and align widgets specific for this **Fitter** in the **Image
        tab** of the **settings panel**.
        """

        # create elements
        itl = self.image_tab.layout

        self.image_x_pos_lbl.setText('Momentum:')
        self.image_y_pos_lbl.setText('Energy:')

        self.image_bin = QCheckBox('bin k')
        self.image_bin_n = QSpinBox()
        self.image_bin_n.setValue(3)

        row = 1
        itl.addWidget(self.image_bin,               row, 3)
        itl.addWidget(self.image_bin_n,             row, 4)

    def set_fitting_tab(self) -> None:
        """
        Create and align widgets in the **fitting tab** of the **settings
        panel**.
        """

        self.fitting_tab = QWidget()
        ftl = QtWidgets.QGridLayout()

        self.fitting_mu_lbl = QLabel('\u03BC:')
        self.fitting_mu = QDoubleSpinBox()
        self.fitting_mu.setRange(self.x_axis.min(), self.x_axis.max())
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
        self.fitting_range_start = QDoubleSpinBox()
        self.fitting_range_start.setRange(self.x_axis.min(), self.x_axis.max())
        self.fitting_range_start.setSingleStep(wp.get_step(self.x_axis))
        self.fitting_range_start.setDecimals(6)
        self.fitting_range_start.setValue(self.x_axis.min())

        self.fitting_range_stop = QDoubleSpinBox()
        self.fitting_range_stop.setRange(self.x_axis.min(), self.x_axis.max())
        self.fitting_range_stop.setSingleStep(wp.get_step(self.x_axis))
        self.fitting_range_stop.setDecimals(6)
        self.fitting_range_stop.setValue(self.x_axis.max())

        self.fitting_bgr_range_lbl = QLabel('bgr range:')
        self.fitting_bgr_range_lbl.setFont(bold_font)
        self.fitting_bgr_range_first = QDoubleSpinBox()
        self.fitting_bgr_range_first.setRange(self.x_axis.min(), self.x_axis.max())
        self.fitting_bgr_range_first.setSingleStep(wp.get_step(self.x_axis))
        self.fitting_bgr_range_first.setDecimals(6)
        self.fitting_bgr_range_first.setValue(self.x_axis.min())

        self.fitting_bgr_range_second = QDoubleSpinBox()
        self.fitting_bgr_range_second.setRange(self.x_axis.min(),
                                               self.x_axis.max())
        self.fitting_bgr_range_second.setSingleStep(wp.get_step(self.x_axis))
        self.fitting_bgr_range_second.setDecimals(6)
        self.fitting_bgr_range_second.setValue(self.x_axis.max())

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

        self.fitting_tab.layout = ftl
        self.fitting_tab.setLayout(ftl)
        self.tabs.addTab(self.fitting_tab, 'Fitting')

    def set_dc_panel(self) -> None:
        """
        Set up an **MDC panel** with extracted curve.
        """

        mdc_plot = self.dc_panel
        self.fit_result = None
        self.bgr_fit = None

        for di in mdc_plot.listDataItems():
            try:
                mdc_plot.removeItem(di)
            except AttributeError:
                pass

        e_idx = self.hor_pos.get_value()
        if self.image_bin.isChecked():
            n = self.image_bin_n.value()
            self.mdc = np.sum(self.data[:, (e_idx - n):(e_idx + n)], axis=1)
        else:
            self.mdc = self.data[:, e_idx]

        range_start, range_stop = self.fitting_range_start.value(), \
                                  self.fitting_range_stop.value()
        bgr_range_f, bgr_range_s = self.fitting_bgr_range_first.value(), \
                                   self.fitting_bgr_range_second.value()
        self.fitting_range_start.setRange(self.x_axis.min(), range_stop)
        self.fitting_range_stop.setRange(range_start, self.x_axis.max())
        self.fitting_bgr_range_first.setRange(range_start, range_stop)
        self.fitting_bgr_range_second.setRange(range_start, range_stop)

        # print(self.x_axis.shape, self.mdc.shape)
        mdc_plot.plot(self.x_axis, self.mdc, pen=mkPen('k', width=2))
        shadow_bottom = PlotDataItem([range_start, range_stop],
                                     [-self.mdc.max(), -self.mdc.max()])
        shadow_top = PlotDataItem([range_start, range_stop],
                                  [2 * self.mdc.max(), 2 * self.mdc.max()])

        try:
            mdc_plot.removeItem(self.shadow)
        except AttributeError:
            pass
        self.shadow = FillBetweenItem(shadow_top, shadow_bottom,
                                      mkBrush((229, 229, 229)))

        self.bgr_first = PlotDataItem([bgr_range_f, bgr_range_f],
                                      [-self.mdc.max(),
                                       self.mdc[wp.indexof(bgr_range_f,
                                                           self.x_axis)]],
                                      pen=mkPen('k', style=QtCore.Qt.DashLine))
        self.bgr_second = PlotDataItem([bgr_range_s, bgr_range_s],
                                       [-self.mdc.max(),
                                        self.mdc[wp.indexof(bgr_range_s,
                                                            self.x_axis)]],
                                       pen=mkPen('k',
                                                 style=QtCore.Qt.DashLine))
        mdc_plot.addItem(self.shadow)
        mdc_plot.addItem(self.bgr_first)
        mdc_plot.addItem(self.bgr_second)
        mdc_plot.setYRange(0, self.mdc.max())

    def update_labels(self) -> None:
        """
        Update labels indicating values in physical coordinates.
        """

        try:
            self.image_x_pos_value_lbl.setText('({:.4f})'.format(
                self.x_axis[int(self.ver_pos.get_value())]))
            self.image_y_pos_value_lbl.setText('({:.4f})'.format(
                self.y_axis[int(self.hor_pos.get_value())]))
            self.fitting_mu.setValue(self.ver_line_dc.value())
        except AttributeError:
            pass

    def fit_bgr(self) -> None:
        """
        Fit polynomial background within the specified region.
        """

        bgr0, bgr1 = self.fitting_range_start.value(), \
                     self.fitting_bgr_range_first.value()
        bgr2, bgr3 = self.fitting_bgr_range_second.value(), \
                     self.fitting_range_stop.value()
        bgr0, bgr1 = wp.indexof(bgr0, self.x_axis), wp.indexof(bgr1, self.x_axis)
        bgr2, bgr3 = wp.indexof(bgr2, self.x_axis), wp.indexof(bgr3, self.x_axis)

        bgr_k = np.hstack((self.x_axis[bgr0:bgr1], self.x_axis[bgr2:bgr3]))
        bgr_mdc = np.hstack((self.mdc[bgr0:bgr1], self.mdc[bgr2:bgr3]))
        order = self.fitting_bgr_poly_order.value()
        coefs_bgr = np.polyfit(bgr_k, bgr_mdc, order)

        self.fit_k = self.x_axis[bgr0:bgr3]
        self.bgr_fit = np.poly1d(coefs_bgr)(self.fit_k)

        try:
            self.dc_panel.removeItem(self.bgr)
        except AttributeError:
            pass

        self.bgr = PlotDataItem(self.fit_k, self.bgr_fit,
                                pen=mkPen('c', width=2))
        self.dc_panel.addItem(self.bgr)

    def fit_mdc(self) -> None:
        """
        Subtract fitted background and fits a(n asymmetric) Lorentzian within
        specified region.
        """

        try:
            self.dc_panel.removeItem(self.fit)
        except AttributeError:
            pass

        range0 = wp.indexof(self.fitting_range_start.value(), self.x_axis)
        range1 = wp.indexof(self.fitting_range_stop.value(), self.x_axis)
        k_fit = self.x_axis[range0:range1]
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

        a, mu, gamma, alpha, beta = self.fitting_a.value(), \
                                    self.fitting_mu.value(), \
                                    self.fitting_gamma.value(), \
                                    self.fitting_alpha.value(), \
                                    self.fitting_beta.value()
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
        self.dc_panel.addItem(self.fit)

    def set_fit_fun(self, fit_alpha: bool, fit_beta: bool) -> None:
        """
        Define exact profile of the function for fitting.

        :param fit_alpha: if :py:obj:`True`, include asymmetry on the side
                          below the expectation values.
        :param fit_beta: if :py:obj:`True`, include asymmetry on the side
                         above the expectation values.
        """

        self.p0 = [self.fitting_a.value(), self.fitting_mu.value(),
                   self.fitting_gamma.value()]
        if fit_alpha and fit_beta:
            self.fit_fun = lambda x, a0, mu, gamma, alpha, beta: \
                wp.asym_lorentzian(x, a0, mu, gamma, alpha, beta)
            self.p0.append(self.fitting_alpha.value())
            self.p0.append(self.fitting_beta.value())
        elif fit_alpha and not fit_beta:
            self.fit_fun = lambda x, a0, mu, gamma, alpha: \
                wp.asym_lorentzian(x, a0, mu, gamma, alpha=alpha)
            self.p0.append(self.fitting_alpha.value())
        elif not fit_alpha and fit_beta:
            self.fit_fun = lambda x, a0, mu, gamma, beta: \
                wp.asym_lorentzian(x, a0, mu, gamma, beta=beta)
            self.p0.append(self.fitting_beta.value())
        else:
            self.fit_fun = lambda x, a0, mu, gamma: \
                wp.asym_lorentzian(x, a0, mu, gamma)

    def set_fitting_message(self, fit_alpha: bool, fit_beta: bool) -> str:
        """
        Set message text to display fitted parameters.

        :param fit_alpha: if :py:obj:`True`, include also `alpha` value.
        :param fit_beta: if :py:obj:`True`, include also `beta` value.
        :return: Message text.
        """

        e = self.y_axis[int(self.hor_pos.get_value())]
        message = 'E = {:.4f};  a = {:.1f};  \u03BC = {:.4f};  ' \
                  '\u0393 = {:.4f}'.format(e, self.p[0], self.p[1], self.p[2])
        if fit_alpha:
            message += ';  \u03B1 = {:.4f}'.format(self.p[3])
        else:
            message += ';  \u03B1 = -'

        if fit_beta:
            message += ';  \u03B2 = {:.4f}'.format(self.p[-1])
        else:
            message += ';  \u03B2 = -'

        return message

    def prepare_fitting_results(self, fit_alpha: bool, fit_beta: bool) -> None:
        """
        Extract fit results into more convenient format.

        :param fit_alpha: if :py:obj:`True`, include also `alpha` value.
        :param fit_beta: if :py:obj:`True`, include also `beta` value.
        """

        e = self.y_axis[int(self.hor_pos.get_value())]
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

    def append_fit_results(self) -> None:
        """
        Append current fit results to previously saved.
        """

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
            self.fit_results = np.flip(np.sort(self.fit_results, axis=0),
                                       axis=0)

        try:
            self.cut_panel.removeItem(self.fit_points)
        except AttributeError:
            pass

        try:
            fit_points_k = np.array([wp.indexof(ki, self.x_axis)
                                     for ki in self.fit_results[:, 2]])
            fit_points_e = np.array([wp.indexof(ei, self.y_axis)
                                     for ei in self.fit_results[:, 0]])
            self.fit_points = ScatterPlotItem(fit_points_k, fit_points_e,
                                              symbol='h')
            self.cut_panel.addItem(self.fit_points)
        except IndexError:
            pass

    def save_fit_results(self) -> None:
        """
        Save all fit results to ``org_file_name`` + ``_mdc_fit_results.txt``,
        in the directory of the original file.
        """

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
        line = 'E [eV]\t\t\ta [a.u.] \t\t mu [1/A]\t\tgamma [1/A]\t\t' \
               'alpha\tbeta [a.u]\n'
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

    def edit_fit_results(self) -> None:
        """
        Opens **\ *.txt** file with saved fit results after clicking **Edit**
        button. For convenience.
        """

        fname = self.title[:-13] + '_mdc_fit_results.txt'
        if not os.path.exists(fname):
            no_file_box = QMessageBox()
            no_file_box.setIcon(QMessageBox.Information)
            no_file_box.setText('File not found.')
            no_file_box.setStandardButtons(QMessageBox.Ok)
            if no_file_box.exec() == QMessageBox.Ok:
                return

        os.system('open ' + fname)

    def load_update_fit_result(self) -> None:
        """
        Load previously saved fit results form **\ *.txt** file.
        """

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
        overwriting_box.setText('Fit results will be overwritten, '
                                'sure to proceed?')
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
            fit_points_k = np.array([wp.indexof(ki, self.x_axis)
                                     for ki in self.fit_results[:, 2]])
            fit_points_e = np.array([wp.indexof(ei, self.y_axis)
                                     for ei in self.fit_results[:, 0]])
            self.fit_points = ScatterPlotItem(fit_points_k, fit_points_e,
                                              symbol='h')
            self.cut_panel.addItem(self.fit_points)
        except IndexError:
            pass


class EDCFitter(Fitter):
    """
    Utility for convenient inspection of energy distribution curves (EDCs).
    """

    def __init__(self, data_viewer: DataViewer2D, data_set: Dataset,
                 axes: list, title: str) -> None:
        """
        Initialize fitter window.

        :param data_viewer: reference to the parent :class:`DataViewer2D` to
                            access its data
        :param data_set: object containing all available data and metadata
        :param title: title of the window and index to store withing parent's
                      record of opened **DataViewers**/**Fitters**
        """

        super(EDCFitter, self).__init__(data_viewer, data_set, axes, title)

        self.data = self.data.T

        self.set_settings_panel()

        self.set_cut_panel(self.data)
        self.set_dc_panel()
        self.set_cmap()

        self.initUI()
        self.set_sliders_initial_pos()
        self.show()

    def initUI(self) -> None:
        """
        Initialize widgets by connecting triggered signals to actions.
        """

        self.image_cmaps.currentIndexChanged.connect(self.set_cmap)
        self.image_invert_colors.stateChanged.connect(self.set_cmap)
        self.image_gamma.valueChanged.connect(self.set_gamma)
        self.image_x_pos.valueChanged.connect(self.update_ver_slider)
        self.image_y_pos.valueChanged.connect(self.update_hor_slider)
        self.image_bin.stateChanged.connect(self.set_binning_lines)
        self.image_bin_n.valueChanged.connect(self.set_binning_lines)
        self.image_edc_range_start.valueChanged.connect(self.set_dc_panel)
        self.image_edc_range_stop.valueChanged.connect(self.set_dc_panel)
        self.image_close_button.clicked.connect(self.close)

        self.symmetrize_box.stateChanged.connect(self.set_dc_panel)

    def set_settings_panel(self) -> None:
        """
        Set tabs in the **settings panel**.
        """

        self.set_image_tab()
        self.customize_image_tab()

    def customize_image_tab(self) -> None:
        """
        Create and align widgets specific for this **Fitter** in the **Image
        tab** of the **settings panel**.
        """

        # create elements
        itl = self.image_tab.layout

        self.image_x_pos_lbl.setText('Energy:')
        self.image_y_pos_lbl.setText('Momentum:')

        self.image_bin = QCheckBox('bin E')
        self.image_bin_n = QSpinBox()
        self.image_bin_n.setValue(3)

        self.image_edc_range_lbl = QLabel('EDC range:')
        self.image_edc_range_lbl.setFont(bold_font)
        self.image_edc_range_start = QDoubleSpinBox()
        self.image_edc_range_start.setRange(self.x_axis.min(),
                                            self.x_axis.max())
        self.image_edc_range_start.setSingleStep(wp.get_step(self.x_axis))
        self.image_edc_range_start.setDecimals(6)
        self.image_edc_range_start.setValue(self.x_axis.min())

        self.symmetrize_box = QCheckBox('symmetrize')

        self.image_edc_range_stop = QDoubleSpinBox()
        self.image_edc_range_stop.setRange(self.x_axis.min(),
                                           self.x_axis.max())
        self.image_edc_range_stop.setSingleStep(wp.get_step(self.x_axis))
        self.image_edc_range_stop.setDecimals(6)
        self.image_edc_range_stop.setValue(self.x_axis.max())

        row = 0
        itl.addWidget(self.image_edc_range_lbl,     row, 5)
        itl.addWidget(self.image_edc_range_start,   row, 6)
        itl.addWidget(self.image_edc_range_stop,    row, 7)

        row = 1
        itl.addWidget(self.image_bin,               row, 3)
        itl.addWidget(self.image_bin_n,             row, 4)
        itl.addWidget(self.symmetrize_box,          row, 5)

    def set_dc_panel(self) -> None:
        """
        Set up an **EDC panel** with extracted curve.
        """

        edc_plot = self.dc_panel

        for di in edc_plot.listDataItems():
            try:
                edc_plot.removeItem(di)
            except AttributeError:
                pass

        k_idx = self.hor_pos.get_value()
        e_start, e_stop = self.image_edc_range_start.value(), \
                          self.image_edc_range_stop.value()
        e_start, e_stop = wp.indexof(e_start, self.x_axis), \
                          wp.indexof(e_stop, self.x_axis)

        try:
            self.ver_line_cut.setBounds([e_start, e_stop])
            self.ver_line_edc.setBounds([self.x_axis[e_start],
                                         self.x_axis[e_stop]])
            self.image_x_pos.setRange(e_start, e_stop)
        except AttributeError:
            pass

        # remember analyzer and energy axes are swapped here
        if self.image_bin.isChecked():
            n = self.image_bin_n.value()
            self.edc = np.sum(self.data[e_start:e_stop,
                              (k_idx - n):(k_idx + n)], axis=1)
        else:
            self.edc = self.data[e_start:e_stop, k_idx]
        self.edc_erg_ax = self.x_axis[e_start:e_stop]

        if self.symmetrize_box.isChecked():
            if self.image_edc_range_start.value() > 0:
                kin_erg_box = QMessageBox()
                kin_erg_box.setIcon(QMessageBox.Information)
                kin_erg_box.setText('Energy must be in binding')
                kin_erg_box.setStandardButtons(QMessageBox.Ok)
                if kin_erg_box.exec() == QMessageBox.Ok:
                    self.symmetrize_box.setChecked(False)
                    self.edc, self.edc_erg_ax = \
                        wp.symmetrize_edc(self.edc, self.edc_erg_ax)
            else:
                self.edc, self.edc_erg_ax = \
                    wp.symmetrize_edc(self.edc, self.edc_erg_ax)

        edc_plot.plot(self.edc_erg_ax, self.edc, pen=mkPen('k', width=2))

        for di in self.cut_panel.listDataItems():
            try:
                self.cut_panel.removeItem(di)
            except AttributeError:
                pass

        edc_plot.setYRange(0, self.edc.max())
        edc_plot.setXRange(self.edc_erg_ax.min(), self.edc_erg_ax.max())

        self.edc_line_start = PlotDataItem(
            [e_start, e_start], [0, self.y_axis.size],
            pen=mkPen('m', style=QtCore.Qt.DashLine))
        self.edc_line_stop = PlotDataItem(
            [e_stop, e_stop], [0, self.y_axis.size],
            pen=mkPen('m', width=2, style=QtCore.Qt.DashLine))
        self.cut_panel.addItem(self.edc_line_start)
        self.cut_panel.addItem(self.edc_line_stop)

    def update_labels(self) -> None:
        """
        Update labels indicating values in physical coordinates.
        """

        try:
            self.image_y_pos_value_lbl.setText('({:.4f})'.format(
                self.y_axis[int(self.hor_pos.get_value())]))
            self.image_x_pos_value_lbl.setText('({:.4f})'.format(
                self.x_axis[int(self.ver_pos.get_value())]))
        except AttributeError:
            pass
