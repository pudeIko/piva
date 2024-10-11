from __future__ import annotations
import os
import subprocess
from sys import platform
from datetime import datetime
from typing import Union, Any, TYPE_CHECKING
import numpy as np
from PyQt5.QtWidgets import QWidget, QLabel, QCheckBox, QComboBox, \
    QDoubleSpinBox, QSpinBox, QPushButton, QLineEdit, QMainWindow, \
    QDialogButtonBox, QMessageBox, QScrollArea, QTableWidget, QVBoxLayout
from PyQt5.QtWidgets import QTableWidgetItem as QTabItem, QSizePolicy
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
if TYPE_CHECKING:
    from piva.data_viewer_2d import DataViewer2D
    from piva.data_viewer_3d import DataViewer3D

import piva.working_procedures as wp
import piva.data_loaders as dl
from piva.data_loaders import Dataset
from piva.cmaps import cmaps, my_cmaps
from piva.qcheckcombobox import CheckComboBox

BASE_LINECOLOR = (255, 255, 0, 255)
BINLINES_LINECOLOR = (168, 168, 104, 255)
ORIENTLINES_LINECOLOR = (164, 37, 22, 255)
HOVER_COLOR = (195, 155, 0, 255)
BGR_COLOR = (64, 64, 64)
util_panel_style = """
QFrame{margin:5px; border:1px solid rgb(150,150,150);}
QLabel{color: rgb(246, 246, 246); border:1px solid rgb(64, 64, 64);}
QCheckBox{color: rgb(246, 246, 246);}
QTabWidget{background-color: rgb(64, 64, 64);}
"""
SIGNALS = 5
MY_CMAPS = True
DEFAULT_CMAP = 'coolwarm'

bold_font = QtGui.QFont()
bold_font.setBold(True)


class UtilitiesPanel(QWidget):
    """
    :class:`QWidget` object containing **settings tabs** with image
    visualization and basic analysis utilities.
    """

    def __init__(self, main_window: Union[DataViewer2D, DataViewer3D],
                 name: str = None, dim: int = 3) -> None:
        """
        Initialize Utilities panel window

        :param main_window: *hosting* main window widget
        :param name: name of the **DataViewer** window, corresponds to the
                     name of the opened data file
        :param dim: dimensionality of the data
        """

        super().__init__()

        self.mw = main_window
        self.layout = QtWidgets.QGridLayout()
        self.tabs = QtWidgets.QTabWidget()
        self.tabs_visible = True
        self.dim = dim
        self.jl_session_running = False

        self.close_button = QPushButton('close')
        self.save_button = QPushButton('save')
        self.hide_button = QPushButton('hide tabs')
        self.pit_button = QPushButton('open in PIT')

        self.buttons = QWidget()
        self.buttons_layout = QtWidgets.QGridLayout()
        self.buttons_layout.addWidget(self.close_button,    1, 0)
        self.buttons_layout.addWidget(self.save_button,     2, 0)
        if self.dim in (2, 4):
            self.file_mdc_fitter_button = QPushButton('MDC fitter')
            self.file_edc_fitter_button = QPushButton('EDC fitter')
            self.buttons_layout.addWidget(self.file_mdc_fitter_button, 3, 0)
            self.buttons_layout.addWidget(self.file_edc_fitter_button, 4, 0)
        if self.dim == 2:
            self.buttons_layout.addWidget(self.pit_button,      5, 0)
        elif self.dim == 3:
            self.buttons_layout.addWidget(self.pit_button,      3, 0)
        self.buttons.setLayout(self.buttons_layout)

        if name is not None:
            self.name = name
        else:
            self.name = 'Unnamed'

        self.initUI()

    def initUI(self) -> None:
        """
        Initialize (shared) widgets by connecting triggered signals to actions.
        """

        self.setStyleSheet(util_panel_style)
        momentum_labels_width = 80
        energy_labels_width = 80
        self.tabs_rows_span = 4
        self.tabs_cols_span = 8

        self.align()

        if self.dim in (2, 4):
            self.energy_vert_value.setFixedWidth(energy_labels_width)
            self.momentum_hor_value.setFixedWidth(momentum_labels_width)
        elif self.dim == 3:
            self.energy_main_value.setFixedWidth(energy_labels_width)
            self.energy_hor_value.setFixedWidth(energy_labels_width)
            self.energy_vert_value.setFixedWidth(energy_labels_width)
            self.momentum_hor_value.setFixedWidth(momentum_labels_width)
            self.momentum_vert_value.setFixedWidth(momentum_labels_width)

        self.layout.addWidget(self.tabs,
                              0, 0, self.tabs_rows_span, self.tabs_cols_span)
        self.layout.addWidget(self.buttons, 0, self.tabs_cols_span + 1)
        self.setLayout(self.layout)

        # linking options
        if self.dim in (2, 3):
            self.link_windows.clicked.connect(self.link_selected_windows)

        # file options
        self.file_show_dp_button.clicked.connect(
            self.show_data_provenance_window)
        self.file_show_md_button.clicked.connect(self.show_metadata_window)
        self.file_add_md_button.clicked.connect(self.add_metadata)
        self.file_remove_md_button.clicked.connect(self.remove_metadata)
        self.file_sum_datasets_sum_button.clicked.connect(self.sum_datasets)
        self.file_sum_datasets_reset_button.clicked.connect(
            self.reset_summation)
        self.file_jl_fname_button.clicked.connect(self.create_jl_file)
        self.file_jl_session_button.clicked.connect(self.open_jl_session)
        self.file_jl_explog_button.clicked.connect(
            self.create_experimental_logbook_file)

        self.setup_cmaps()
        # self.setup_gamma()
        # self.setup_colorscale()
        # self.setup_bin_z()
        if platform == 'win32':
            self.set_tabs_color()

    def align(self) -> None:
        """
        Align all GUI widgets in the window.
        """

        self.set_volume_tab()
        self.set_image_tab()
        self.set_axes_tab()
        if self.dim == 3:
            self.set_orientate_tab()
        self.set_file_tab()

    def set_image_tab(self) -> None:
        """
        Create and align widgets in the **Image tab** of the utilities panel.
        """

        max_w = 80
        # create elements
        self.image_tab = QWidget()
        itl = QtWidgets.QGridLayout()
        self.image_colors_label = QLabel('Colors')
        self.image_colors_label.setFont(bold_font)
        self.image_cmaps_label = QLabel('cmaps:')
        self.image_cmaps = QComboBox()
        self.image_invert_colors = QCheckBox('invert colors')
        self.image_gamma_label = QLabel('gamma:')
        self.image_gamma = QDoubleSpinBox()
        self.image_gamma.setRange(0.05, 10)
        self.image_gamma.setValue(1)
        self.image_gamma.setSingleStep(0.05)
        # self.image_colorscale_label = QLabel('color scale:')
        # self.image_colorscale = QDoubleSpinBox()
        # self.image_colorscale.setRange(0., 10.)

        self.image_normalize_lbl = QLabel('Normalize')
        self.image_normalize_lbl.setFont(bold_font)
        self.image_normalize_to_lbl = QLabel('to:')
        self.image_normalize_to = QComboBox()
        self.image_normalize_along_lbl = QLabel('along axis:')
        self.image_normalize_along = QComboBox()
        self.image_normalize = QCheckBox('normalize')

        # self.image_BZ_contour_lbl = QLabel('BZ contour')
        # self.image_BZ_contour_lbl.setFont(bold_font)
        # self.image_show_BZ = QCheckBox('show')
        # self.image_symmetry_label = QLabel('symmetry:')
        # self.image_symmetry = QSpinBox()
        # self.image_symmetry.setRange(4, 6)
        # self.image_rotate_BZ_label = QLabel('rotate:')
        # self.image_rotate_BZ = QDoubleSpinBox()
        # self.image_rotate_BZ.setRange(-90, 90)
        # self.image_rotate_BZ.setSingleStep(0.5)

        if self.dim in (3, 4):
            self.image_2dv_lbl = QLabel('Open in 2D viewer')
            self.image_2dv_lbl.setFont(bold_font)
            self.image_2dv_cut_selector_lbl = QLabel('select cut')
            self.image_2dv_cut_selector = QComboBox()
            self.image_2dv_cut_selector.addItems(['vertical', 'horizontal'])
            self.image_2dv_button = QPushButton('Open')

        # self.image_smooth_lbl = QLabel('Smooth')
        # self.image_smooth_lbl.setFont(bold_font)
        # self.image_smooth_n_lbl = QLabel('box size:')
        # self.image_smooth_n = QSpinBox()
        # self.image_smooth_n.setValue(3)
        # self.image_smooth_n.setRange(3, 50)
        # self.image_smooth_n.setMaximumWidth(max_w)
        # self.image_smooth_rl_lbl = QLabel('recursion:')
        # self.image_smooth_rl = QSpinBox()
        # self.image_smooth_rl.setValue(3)
        # self.image_smooth_rl.setRange(1, 20)
        # self.image_smooth_button = QPushButton('Smooth')
        #
        # self.image_curvature_lbl = QLabel('Curvature')
        # self.image_curvature_lbl.setFont(bold_font)
        # self.image_curvature_method_lbl = QLabel('method:')
        # self.image_curvature_method = QComboBox()
        # curvature_methods = ['2D', '1D (EDC)', '1D (MDC)']
        # self.image_curvature_method.addItems(curvature_methods)
        # self.image_curvature_a_lbl = QLabel('a:')
        # self.image_curvature_a = QDoubleSpinBox()
        # self.image_curvature_a.setRange(-10e5, 10e10)
        # self.image_curvature_a.setSingleStep(0.001)
        # self.image_curvature_a.setValue(10.)
        # self.image_curvature_a.setMaximumWidth(max_w)
        # self.image_curvature_button = QPushButton('Do it')

        # addWidget(widget, row, column, rowSpan, columnSpan)

        if self.dim in (2, 3):
            row = 0
            itl.addWidget(self.image_colors_label,          row, 0)
            itl.addWidget(self.image_cmaps_label,           row, 1)
            itl.addWidget(self.image_cmaps,                 row, 2)
            itl.addWidget(self.image_invert_colors,         row, 3, 1, 2)

            row = 1
            itl.addWidget(self.image_gamma_label,           row, 1)
            itl.addWidget(self.image_gamma,                 row, 2)

            row = 2
            itl.addWidget(self.image_normalize_lbl,         row, 0)
            itl.addWidget(self.image_normalize_to_lbl,      row, 1)
            itl.addWidget(self.image_normalize_to,          row, 2)
            itl.addWidget(self.image_normalize_along_lbl,   row, 3)
            itl.addWidget(self.image_normalize_along,       row, 4)
            itl.addWidget(self.image_normalize,             row, 5)

        if self.dim == 2:
            self.image_smooth_lbl = QLabel('Smooth')
            self.image_smooth_lbl.setFont(bold_font)
            self.image_smooth_n_lbl = QLabel('box size:')
            self.image_smooth_n = QSpinBox()
            self.image_smooth_n.setValue(3)
            self.image_smooth_n.setRange(3, 50)
            self.image_smooth_n.setMaximumWidth(max_w)
            self.image_smooth_rl_lbl = QLabel('recursion:')
            self.image_smooth_rl = QSpinBox()
            self.image_smooth_rl.setValue(3)
            self.image_smooth_rl.setRange(1, 20)
            self.image_smooth_button = QPushButton('Smooth')

            self.image_curvature_lbl = QLabel('Curvature')
            self.image_curvature_lbl.setFont(bold_font)
            self.image_curvature_method_lbl = QLabel('method:')
            self.image_curvature_method = QComboBox()
            curvature_methods = ['2D', '1D (EDC)', '1D (MDC)']
            self.image_curvature_method.addItems(curvature_methods)
            self.image_curvature_a_lbl = QLabel('a:')
            self.image_curvature_a = QDoubleSpinBox()
            self.image_curvature_a.setRange(10e-15, 10e10)
            self.image_curvature_a.setSingleStep(0.000001)
            self.image_curvature_a.setValue(0.1)
            self.image_curvature_a.setDecimals(6)
            self.image_curvature_a.setMaximumWidth(max_w)
            self.image_curvature_button = QPushButton('Do it')

            self.image_normalize_to.addItems(['maximum', 'intensity sum'])
            self.image_normalize_along.addItems(['slit', 'energy'])

            row = 3
            itl.addWidget(self.image_smooth_lbl,        row, 0)
            itl.addWidget(self.image_smooth_n_lbl,      row, 1)
            itl.addWidget(self.image_smooth_n,          row, 2)
            itl.addWidget(self.image_smooth_rl_lbl,     row, 3)
            itl.addWidget(self.image_smooth_rl,         row, 4)
            itl.addWidget(self.image_smooth_button,     row, 5, 1, 2)

            row = 4
            itl.addWidget(self.image_curvature_lbl,         row, 0)
            itl.addWidget(self.image_curvature_method_lbl,  row, 1)
            itl.addWidget(self.image_curvature_method,      row, 2)
            itl.addWidget(self.image_curvature_a_lbl,       row, 3)
            itl.addWidget(self.image_curvature_a,           row, 4)
            itl.addWidget(self.image_curvature_button,      row, 5, 1, 2)

        elif self.dim == 3:

            self.image_BZ_contour_lbl = QLabel('BZ contour')
            self.image_BZ_contour_lbl.setFont(bold_font)
            self.image_show_BZ = QCheckBox('show')
            self.image_symmetry_label = QLabel('symmetry:')
            self.image_symmetry = QSpinBox()
            self.image_symmetry.setRange(4, 6)
            self.image_rotate_BZ_label = QLabel('rotate:')
            self.image_rotate_BZ = QDoubleSpinBox()
            self.image_rotate_BZ.setRange(-90, 90)
            self.image_rotate_BZ.setSingleStep(0.5)

            self.image_normalize_to.addItems(['maximum', 'intensity sum'])
            self.image_normalize_along.addItems(['scanned', 'slit', 'energy'])

            row = 3
            itl.addWidget(self.image_BZ_contour_lbl,    row, 0)
            itl.addWidget(self.image_symmetry_label,    row, 1)
            itl.addWidget(self.image_symmetry,          row, 2)
            itl.addWidget(self.image_rotate_BZ_label,   row, 3)
            itl.addWidget(self.image_rotate_BZ,         row, 4)
            itl.addWidget(self.image_show_BZ,           row, 5)

            row = 4
            itl.addWidget(self.image_2dv_lbl,               row, 0, 1, 2)
            itl.addWidget(self.image_2dv_cut_selector_lbl,  row, 2)
            itl.addWidget(self.image_2dv_cut_selector,      row, 3)
            itl.addWidget(self.image_2dv_button,            row, 4)

            # dummy item
            dummy_lbl = QLabel('')
            itl.addWidget(dummy_lbl, 5, 0, 1, 7)

        elif self.dim == 4:

            self.image_normalize_lbl.setText('Normalize spectrum')
            self.image_only_spectrum = QCheckBox('only spectrum')
            self.image_raster_label = QLabel('Spectra in raster image')
            self.image_raster_label.setFont(bold_font)
            self.image_raster_options = QComboBox()
            self.image_raster_options.addItems(
                ['sum', 'signal/noise', 'sharpest edge'])
            # self.image_raster_button = QPushButton('Do it')

            self.image_normalize_to.addItems(['maximum', 'intensity sum'])
            self.image_normalize_along.addItems(['slit', 'energy'])

            row = 0
            itl.addWidget(self.image_colors_label,          row, 0)
            itl.addWidget(self.image_cmaps_label,           row, 1)
            itl.addWidget(self.image_cmaps,                 row, 2)
            itl.addWidget(self.image_invert_colors,         row, 3)

            row = 1
            itl.addWidget(self.image_gamma_label,           row, 1)
            itl.addWidget(self.image_gamma,                 row, 2)
            itl.addWidget(self.image_only_spectrum,         row, 3)

            row = 2
            itl.addWidget(self.image_normalize_lbl,         row, 0)
            itl.addWidget(self.image_normalize_to_lbl,      row, 1)
            itl.addWidget(self.image_normalize_to,          row, 2)
            itl.addWidget(self.image_normalize_along_lbl,   row, 3)
            itl.addWidget(self.image_normalize_along,       row, 4)
            itl.addWidget(self.image_normalize,             row, 5)

            row = 3
            itl.addWidget(self.image_raster_label,          row, 0)
            itl.addWidget(self.image_raster_options,        row, 1, 1, 2)
            # itl.addWidget(self.image_raster_button,         row, 3)

            row = 4
            itl.addWidget(self.image_2dv_lbl,               row, 0)
            itl.addWidget(self.image_2dv_cut_selector_lbl,  row, 2)
            itl.addWidget(self.image_2dv_cut_selector,      row, 3)
            itl.addWidget(self.image_2dv_button,            row, 4)

            # row = 3
            # itl.addWidget(self.image_smooth_lbl,        row, 0)
            # itl.addWidget(self.image_smooth_n_lbl,      row, 1)
            # itl.addWidget(self.image_smooth_n,          row, 2)
            # itl.addWidget(self.image_smooth_rl_lbl,     row, 3)
            # itl.addWidget(self.image_smooth_rl,         row, 4)
            # itl.addWidget(self.image_smooth_button,     row, 5, 1, 2)
            #
            # row = 4
            # itl.addWidget(self.image_curvature_lbl,         row, 0)
            # itl.addWidget(self.image_curvature_method_lbl,  row, 1)
            # itl.addWidget(self.image_curvature_method,      row, 2)
            # itl.addWidget(self.image_curvature_a_lbl,       row, 3)
            # itl.addWidget(self.image_curvature_a,           row, 4)
            # itl.addWidget(self.image_curvature_button,      row, 5, 1, 2)

        self.image_tab.layout = itl
        self.image_tab.setLayout(itl)
        self.tabs.addTab(self.image_tab, 'Image')

    def set_volume_tab(self) -> None:
        """
        Create and align widgets in the **Volume tab** of the utilities panel.
        """

        self.volume_tab = QWidget()
        vtl = QtWidgets.QGridLayout()
        max_lbl_w = 40
        bin_box_w = 50
        coords_box_w = 70

        # if self.dim in (2, 3):
        #     print('elo')
        self.link_windows_lbl = QLabel('Link windows')
        self.link_windows_lbl.setFont(bold_font)
        self.link_windows_list = CheckComboBox(
            placeholderText='--select file--')
        self.set_opened_viewers_list()
        self.link_windows_status = QComboBox()
        self.link_windows_status.addItems(['free', 'master', 'slave'])
        self.link_windows_status.setDisabled(True)
        self.link_windows_status.blockSignals(True)
        self.link_windows = QPushButton('Link')

        if self.dim in (2, 4):
            # binning option
            self.bins_label = QLabel('Integrate')
            self.bins_label.setFont(bold_font)
            self.bin_y = QCheckBox('bin EDCs')
            self.bin_y_nbins = QSpinBox()
            self.bin_z = QCheckBox('bin MDCs')
            self.bin_z_nbins = QSpinBox()
            self.bin_z_nbins.setRange(0, 1000)
            self.bin_z_nbins.setValue(0)

            # cross' hairs positions
            self.positions_momentum_label = QLabel('Momentum sliders')
            self.positions_momentum_label.setFont(bold_font)
            self.energy_vert_label = QLabel('E:')
            self.energy_vert = QSpinBox()
            self.energy_vert_value = QLabel('eV')
            self.momentum_hor_label = QLabel('kx:')
            self.momentum_hor = QSpinBox()
            self.momentum_hor_value = QLabel('deg')

        if self.dim == 2:
            col = 0
            vtl.addWidget(self.positions_momentum_label,    0, col, 1, 3)
            vtl.addWidget(self.energy_vert_label,           1, col)
            vtl.addWidget(self.energy_vert,                 1, col + 1)
            vtl.addWidget(self.energy_vert_value,           1, col + 2)
            vtl.addWidget(self.momentum_hor_label,          2, col)
            vtl.addWidget(self.momentum_hor,                2, col + 1)
            vtl.addWidget(self.momentum_hor_value,          2, col + 2)

            col = 4
            vtl.addWidget(self.bins_label,                  0, col, 1, 2)
            vtl.addWidget(self.bin_y,                       1, col)
            vtl.addWidget(self.bin_y_nbins,                 1, col + 1)
            vtl.addWidget(self.bin_z,                       2, col)
            vtl.addWidget(self.bin_z_nbins,                 2, col + 1)

            vtl.addWidget(self.link_windows_lbl,            4, 0)
            vtl.addWidget(self.link_windows_list,           4, 1, 1, 3)
            vtl.addWidget(self.link_windows_status,         4, 4)
            vtl.addWidget(self.link_windows,                4, 5)

            # dummy lbl
            dummy_lbl = QLabel('')
            vtl.addWidget(dummy_lbl, 3, 0, 1, 6)

        elif self.dim == 3:
            # binning option
            self.bin_z = QCheckBox('bin E')
            self.bin_z_nbins = QSpinBox()
            self.bin_z_nbins.setMaximumWidth(bin_box_w)
            self.bin_x = QCheckBox('bin kx')
            self.bin_x_nbins = QSpinBox()
            self.bin_x_nbins.setMaximumWidth(bin_box_w)
            self.bin_y = QCheckBox('bin ky')
            self.bin_y_nbins = QSpinBox()
            self.bin_y_nbins.setMaximumWidth(bin_box_w)
            self.bin_zx = QCheckBox('bin E (kx)')
            self.bin_zx_nbins = QSpinBox()
            self.bin_zx_nbins.setMaximumWidth(bin_box_w)
            self.bin_zy = QCheckBox('bin E (ky)')
            self.bin_zy_nbins = QSpinBox()
            self.bin_zy_nbins.setMaximumWidth(bin_box_w)

            # cross' hairs positions
            self.positions_energies_label = QLabel('Energy sliders')
            self.positions_energies_label.setFont(bold_font)
            self.energy_main_label = QLabel('main:')
            self.energy_main_label.setMaximumWidth(50)
            self.energy_main = QSpinBox()
            self.energy_main.setMaximumWidth(coords_box_w)
            self.energy_main_value = QLabel('eV')
            self.energy_hor_label = QLabel('kx:')
            self.energy_hor_label.setMaximumWidth(max_lbl_w)
            self.energy_hor = QSpinBox()
            self.energy_hor.setMaximumWidth(coords_box_w)
            self.energy_hor_value = QLabel('eV')
            self.energy_vert_label = QLabel('ky:')
            self.energy_vert_label.setMaximumWidth(max_lbl_w)
            self.energy_vert = QSpinBox()
            self.energy_vert.setMaximumWidth(coords_box_w)
            self.energy_vert_value = QLabel('eV')

            self.positions_momentum_label = QLabel('Momentum sliders')
            self.positions_momentum_label.setFont(bold_font)
            self.momentum_hor_label = QLabel('ky:')
            self.momentum_hor_label.setMaximumWidth(max_lbl_w)
            self.momentum_hor = QSpinBox()
            self.momentum_hor.setMaximumWidth(coords_box_w)
            self.momentum_hor_value = QLabel('deg')
            self.momentum_vert_label = QLabel('kx:')
            self.momentum_vert_label.setMaximumWidth(max_lbl_w)
            self.momentum_vert = QSpinBox()
            self.momentum_vert.setMaximumWidth(coords_box_w)
            self.momentum_vert_value = QLabel('deg')

            col = 0
            vtl.addWidget(self.positions_energies_label,        0, col, 1, 3)
            vtl.addWidget(self.energy_main_label,               1, col)
            vtl.addWidget(self.energy_main,                     1, col + 1)
            vtl.addWidget(self.energy_main_value,               1, col + 2)
            vtl.addWidget(self.energy_hor_label,                2, col)
            vtl.addWidget(self.energy_hor,                      2, col + 1)
            vtl.addWidget(self.energy_hor_value,                2, col + 2)
            vtl.addWidget(self.energy_vert_label,               3, col)
            vtl.addWidget(self.energy_vert,                     3, col + 1)
            vtl.addWidget(self.energy_vert_value,               3, col + 2)

            col = 3
            vtl.addWidget(self.positions_momentum_label,        0, col, 1, 3)
            vtl.addWidget(self.momentum_vert_label,             1, col)
            vtl.addWidget(self.momentum_vert,                   1, col + 1)
            vtl.addWidget(self.momentum_vert_value,             1, col + 2)
            vtl.addWidget(self.momentum_hor_label,              2, col)
            vtl.addWidget(self.momentum_hor,                    2, col + 1)
            vtl.addWidget(self.momentum_hor_value,              2, col + 2)

            col = 6
            vtl.addWidget(self.bin_z,                           0, col)
            vtl.addWidget(self.bin_z_nbins,                     0, col + 1)
            vtl.addWidget(self.bin_x,                           1, col)
            vtl.addWidget(self.bin_x_nbins,                     1, col + 1)
            vtl.addWidget(self.bin_y,                           2, col)
            vtl.addWidget(self.bin_y_nbins,                     2, col + 1)
            vtl.addWidget(self.bin_zx,                          3, col)
            vtl.addWidget(self.bin_zx_nbins,                    3, col + 1)
            vtl.addWidget(self.bin_zy,                          4, col)
            vtl.addWidget(self.bin_zy_nbins,                    4, col + 1)

            vtl.addWidget(self.link_windows_lbl,                5, 0, 1, 2)
            vtl.addWidget(self.link_windows_list,               5, 2, 1, 4)
            vtl.addWidget(self.link_windows_status,             5, 6)
            vtl.addWidget(self.link_windows,                    5, 7)

        elif self.dim == 4:
            # cross' hairs positions
            self.raster_label = QLabel('Position sliders')
            self.raster_label.setFont(bold_font)
            self.rx_vert_label = QLabel('x:')
            self.rx_vert = QSpinBox()
            self.rx_vert_value = QLabel('eV')
            self.ry_hor_label = QLabel('y:')
            self.ry_hor = QSpinBox()
            self.ry_hor_value = QLabel('deg')
            self.bin_y.setText('bin E')
            self.bin_z.setText('bin k')

            # # binning option
            # self.bins_r_label = QLabel('Integrate')
            # self.bins_r_label.setFont(bold_font)
            # self.bin_ry = QCheckBox('bin EDCs')
            # self.bin_ry_nbins = QSpinBox()
            # self.bin_rz = QCheckBox('bin MDCs')
            # self.bin_rz_nbins = QSpinBox()
            # self.bin_rz_nbins.setRange(0, 1000)
            # self.bin_rz_nbins.setValue(0)

            col = 0
            vtl.addWidget(self.raster_label,    0, col, 1, 3)
            vtl.addWidget(self.rx_vert_label,   1, col)
            vtl.addWidget(self.rx_vert,         1, col + 1)
            vtl.addWidget(self.rx_vert_value,   1, col + 2)
            vtl.addWidget(self.ry_hor_label,    2, col)
            vtl.addWidget(self.ry_hor,          2, col + 1)
            vtl.addWidget(self.ry_hor_value,    2, col + 2)

            col = 3
            vtl.addWidget(self.positions_momentum_label,    0, col, 1, 3)
            vtl.addWidget(self.energy_vert_label,           1, col)
            vtl.addWidget(self.energy_vert,                 1, col + 1)
            vtl.addWidget(self.energy_vert_value,           1, col + 2)
            vtl.addWidget(self.momentum_hor_label,          2, col)
            vtl.addWidget(self.momentum_hor,                2, col + 1)
            vtl.addWidget(self.momentum_hor_value,          2, col + 2)

            col = 6
            vtl.addWidget(self.bins_label,                  0, col, 1, 2)
            vtl.addWidget(self.bin_y,                       1, col)
            vtl.addWidget(self.bin_y_nbins,                 1, col + 1)
            vtl.addWidget(self.bin_z,                       2, col)
            vtl.addWidget(self.bin_z_nbins,                 2, col + 1)

            # dummy lbl
            dummy_lbl = QLabel('')
            vtl.addWidget(dummy_lbl, 3, 0, 1, 6)
        self.volume_tab.layout = vtl
        self.volume_tab.setLayout(vtl)
        self.tabs.addTab(self.volume_tab, 'Volume')

    def set_axes_tab(self) -> None:
        """
        Create and align widgets in the **Axes tab** of the utilities panel.
        """

        self.axes_tab = QWidget()
        atl = QtWidgets.QGridLayout()
        box_max_w = 100
        lbl_max_h = 30

        self.axes_energy_main_lbl = QLabel('Energy correction')
        self.axes_energy_main_lbl.setFont(bold_font)
        self.axes_energy_main_lbl.setMaximumHeight(lbl_max_h)
        self.axes_energy_Ef_lbl = QLabel('Ef (eV):')
        self.axes_energy_Ef = QDoubleSpinBox()
        self.axes_energy_Ef.setMaximumWidth(box_max_w)
        self.axes_energy_Ef.setRange(-5000., 5000)
        self.axes_energy_Ef.setDecimals(6)
        self.axes_energy_Ef.setSingleStep(0.001)

        self.axes_energy_hv_lbl = QLabel('h\u03BD (eV):')
        self.axes_energy_hv = QDoubleSpinBox()
        self.axes_energy_hv.setMaximumWidth(box_max_w)
        self.axes_energy_hv.setRange(-2000., 2000)
        self.axes_energy_hv.setDecimals(4)
        self.axes_energy_hv.setSingleStep(0.001)

        self.axes_energy_wf_lbl = QLabel('wf (eV):')
        self.axes_energy_wf = QDoubleSpinBox()
        self.axes_energy_wf.setMaximumWidth(box_max_w)
        self.axes_energy_wf.setRange(0, 5)
        self.axes_energy_wf.setDecimals(4)
        self.axes_energy_wf.setSingleStep(0.001)

        self.axes_energy_scale_lbl = QLabel('scale:')
        self.axes_energy_scale = QComboBox()
        self.axes_energy_scale.addItems(['binding', 'kinetic'])
        self.axes_energy_scale.setCurrentIndex(1)

        self.axes_momentum_main_lbl = QLabel('k-space conversion')
        self.axes_momentum_main_lbl.setFont(bold_font)
        self.axes_momentum_main_lbl.setMaximumHeight(lbl_max_h)
        self.axes_gamma_x_lbl = QLabel('\u0393 x0:')
        self.axes_gamma_x = QSpinBox()
        self.axes_gamma_x.setRange(0, 5000)

        self.axes_transform_kz = QCheckBox('kz')

        self.axes_conv_lc_lbl = QLabel('a (\u212B):')
        self.axes_conv_lc = QDoubleSpinBox()
        self.axes_conv_lc.setMaximumWidth(box_max_w)
        self.axes_conv_lc.setRange(0, 10)
        self.axes_conv_lc.setDecimals(4)
        self.axes_conv_lc.setSingleStep(0.001)
        self.axes_conv_lc.setValue(3.1416)

        self.axes_conv_lc_op_lbl = QLabel('c (\u212B):')
        self.axes_conv_lc_op = QDoubleSpinBox()
        self.axes_conv_lc_op.setMaximumWidth(box_max_w)
        self.axes_conv_lc_op.setRange(0, 100)
        self.axes_conv_lc_op.setDecimals(4)
        self.axes_conv_lc_op.setSingleStep(0.001)
        self.axes_conv_lc_op.setValue(3.1416)

        self.axes_slit_orient_lbl = QLabel('Slit:')
        self.axes_slit_orient = QComboBox()
        self.axes_slit_orient.addItems(['horizontal', 'vertical'])
        self.axes_copy_values = QPushButton('Copy from \'Orientate\'')
        self.axes_do_kspace_conv = QPushButton('Convert')
        self.axes_reset_conv = QPushButton('Reset')

        row = 0
        atl.addWidget(self.axes_energy_main_lbl,        row + 0, 0, 1, 2)
        atl.addWidget(self.axes_energy_scale_lbl,       row + 0, 4)
        atl.addWidget(self.axes_energy_scale,           row + 0, 5)
        atl.addWidget(self.axes_energy_Ef_lbl,          row + 1, 0)
        atl.addWidget(self.axes_energy_Ef,              row + 1, 1)
        atl.addWidget(self.axes_energy_hv_lbl,          row + 1, 2)
        atl.addWidget(self.axes_energy_hv,              row + 1, 3)
        atl.addWidget(self.axes_energy_wf_lbl,          row + 1, 4)
        atl.addWidget(self.axes_energy_wf,              row + 1, 5)

        if self.dim in (2, 4):
            self.axes_angle_off_lbl = QLabel('ang offset:')
            self.axes_angle_off = QDoubleSpinBox()
            self.axes_angle_off.setMaximumWidth(box_max_w)
            self.axes_angle_off.setDecimals(4)
            self.axes_angle_off.setSingleStep(0.001)

            row = 2
            atl.addWidget(self.axes_momentum_main_lbl,  row + 0, 0, 1, 2)
            atl.addWidget(self.axes_gamma_x_lbl,        row + 1, 0)
            atl.addWidget(self.axes_gamma_x,            row + 1, 1)
            atl.addWidget(self.axes_angle_off_lbl,      row + 1, 2)
            atl.addWidget(self.axes_angle_off,          row + 1, 3)
            atl.addWidget(self.axes_conv_lc_lbl,        row + 1, 4)
            atl.addWidget(self.axes_conv_lc,            row + 1, 5)

            row = 4
            atl.addWidget(self.axes_slit_orient_lbl,    row, 0)
            atl.addWidget(self.axes_slit_orient,        row, 1)
            atl.addWidget(self.axes_do_kspace_conv,     row, 2, 1, 2)
            atl.addWidget(self.axes_reset_conv,         row, 4, 1, 2)

        elif self.dim == 3:

            self.axes_gamma_y_lbl = QLabel('\u0393 y0')
            self.axes_gamma_y = QSpinBox()
            self.axes_gamma_y.setRange(0, 5000)

            row = 2
            atl.addWidget(self.axes_momentum_main_lbl,  row + 0, 0, 1, 2)
            atl.addWidget(self.axes_slit_orient_lbl,    row + 0, 4)
            atl.addWidget(self.axes_slit_orient,        row + 0, 5)
            atl.addWidget(self.axes_gamma_x_lbl,        row + 1, 0)
            atl.addWidget(self.axes_gamma_x,            row + 1, 1)
            atl.addWidget(self.axes_gamma_y_lbl,        row + 1, 2)
            atl.addWidget(self.axes_gamma_y,            row + 1, 3)
            atl.addWidget(self.axes_copy_values,        row + 1, 4, 1, 2)

            row = 4
            atl.addWidget(self.axes_conv_lc_lbl,        row, 0)
            atl.addWidget(self.axes_conv_lc,            row, 1)
            atl.addWidget(self.axes_conv_lc_op_lbl,     row, 2)
            atl.addWidget(self.axes_conv_lc_op,         row, 3)
            atl.addWidget(self.axes_transform_kz,       row, 4)
            atl.addWidget(self.axes_do_kspace_conv,     row, 5)

        self.axes_tab.layout = atl
        self.axes_tab.setLayout(atl)
        self.tabs.addTab(self.axes_tab, 'Axes')

    def set_orientate_tab(self) -> None:
        """
        Create and align widgets in the **Orientate tab** of the utilities
        panel.
        """

        self.orientate_tab = QWidget()
        otl = QtWidgets.QGridLayout()

        self.orientate_init_cooradinates_lbl = QLabel(
            'Give initial coordinates')
        self.orientate_init_cooradinates_lbl.setFont(bold_font)
        self.orientate_init_x_lbl = QLabel('scanned axis:')
        self.orientate_init_x = QSpinBox()
        self.orientate_init_x.setRange(0, 1000)
        self.orientate_init_y_lbl = QLabel('slit axis:')
        self.orientate_init_y = QSpinBox()
        self.orientate_init_y.setRange(0, 1000)

        self.orientate_find_gamma = QPushButton('Find \t \u0393')
        self.orientate_copy_coords = QPushButton('Copy from \'Volume\'')

        self.orientate_find_gamma_message = QLineEdit(
            'NOTE: algorithm will process the main plot image.')
        self.orientate_find_gamma_message.setReadOnly(True)

        self.orientate_lines_lbl = QLabel('Show rotatable lines')
        self.orientate_lines_lbl.setFont(bold_font)
        self.orientate_hor_line = QCheckBox('horizontal line')
        self.orientate_hor_line
        self.orientate_ver_line = QCheckBox('vertical line')
        self.orientate_angle_lbl = QLabel('rotation angle:')
        self.orientate_angle = QDoubleSpinBox()
        self.orientate_angle.setRange(-180, 180)
        self.orientate_angle.setSingleStep(0.5)

        self.orientate_info_lbl = QLabel('Orientations table:')
        self.orientate_info_button = QPushButton('open')

        # addWidget(widget, row, column, rowSpan, columnSpan)
        row = 0
        otl.addWidget(self.orientate_init_cooradinates_lbl, row,     0, 1, 2)
        otl.addWidget(self.orientate_init_x_lbl,            row + 1, 0)
        otl.addWidget(self.orientate_init_x,                row + 1, 1)
        otl.addWidget(self.orientate_init_y_lbl,            row + 1, 2)
        otl.addWidget(self.orientate_init_y,                row + 1, 3)

        row = 2
        otl.addWidget(self.orientate_find_gamma,            row,     0, 1, 2)
        otl.addWidget(self.orientate_copy_coords,           row,     2, 1, 2)
        otl.addWidget(self.orientate_find_gamma_message,    row + 1, 0, 1, 4)

        col = 4
        otl.addWidget(self.orientate_lines_lbl,             0, col,     1, 2)
        otl.addWidget(self.orientate_hor_line,              1, col)
        otl.addWidget(self.orientate_ver_line,              1, col + 1)
        otl.addWidget(self.orientate_angle_lbl,             2, col)
        otl.addWidget(self.orientate_angle,                 2, col + 1)
        otl.addWidget(self.orientate_info_lbl,              3, col)
        otl.addWidget(self.orientate_info_button,           3, col + 1)

        # dummy lbl
        dummy_lbl = QLabel('')
        otl.addWidget(dummy_lbl, 4, 0, 2, 8)

        self.orientate_tab.layout = otl
        self.orientate_tab.setLayout(otl)
        self.tabs.addTab(self.orientate_tab, 'Orientate')

        self.set_orientation_info_window()

    def set_file_tab(self) -> None:
        """
        Create and align widgets in the **File tab** of the utilities panel.
        """

        self.file_tab = QWidget()
        ftl = QtWidgets.QGridLayout()

        self.file_add_md_lbl = QLabel('Edit entries')
        self.file_add_md_lbl.setFont(bold_font)
        self.file_md_name_lbl = QLabel('name:')
        self.file_md_name = QLineEdit()
        self.file_md_value_lbl = QLabel('value:')
        self.file_md_value = QLineEdit()
        self.file_add_md_button = QPushButton('add')
        self.file_remove_md_button = QPushButton('remove')

        self.file_show_dp_button = QPushButton('data provenance')
        self.file_show_md_button = QPushButton('show metadata')

        self.file_sum_datasets_lbl = QLabel('Sum scans')
        self.file_sum_datasets_lbl.setFont(bold_font)
        self.file_sum_datasets_fname_lbl = QLabel('file name:')
        self.file_sum_datasets_fname = QLineEdit('File name')
        self.file_sum_datasets_sum_button = QPushButton('sum')
        self.file_sum_datasets_reset_button = QPushButton('reset')

        self.file_jl_main_lbl = QLabel('JupyterLab')
        self.file_jl_main_lbl.setFont(bold_font)
        self.file_jl_fname_lbl = QLabel('file name:')
        self.file_jl_fname = QLineEdit(self.mw.title.split('.')[0] + '.ipynb')
        self.file_jl_fname_button = QPushButton('touch')
        self.file_jl_explog_lbl = QLabel('exp.  logbook:')
        self.file_jl_explog = QComboBox()
        self.file_jl_explog.addItems(['--beamline--', 'SIS', 'ADRESS', 'Bloch',
                                      'I05', 'MERLIN', 'URANOS'])
        self.file_jl_explog_button = QPushButton('create')
        self.file_jl_session_button = QPushButton('start JL session')

        row = 0
        ftl.addWidget(self.file_show_dp_button,                 row, 4, 1, 2)
        ftl.addWidget(self.file_show_md_button,                 row, 6, 1, 2)

        row += 1
        ftl.addWidget(self.file_add_md_lbl,                     row, 0)
        ftl.addWidget(self.file_md_name_lbl,                    row, 1)
        ftl.addWidget(self.file_md_name,                        row, 2, 1, 2)
        ftl.addWidget(self.file_md_value_lbl,                   row, 4)
        ftl.addWidget(self.file_md_value,                       row, 5)
        ftl.addWidget(self.file_add_md_button,                  row, 6)
        ftl.addWidget(self.file_remove_md_button,               row, 7)

        if self.dim == 2:
            row += 1
            ftl.addWidget(self.file_sum_datasets_lbl,           row, 0)
            ftl.addWidget(self.file_sum_datasets_fname,         row, 1, 1, 5)
            ftl.addWidget(self.file_sum_datasets_sum_button,    row, 6)
            ftl.addWidget(self.file_sum_datasets_reset_button,  row, 7)

        row += 1
        ftl.addWidget(self.file_jl_main_lbl,                    row, 0)
        ftl.addWidget(self.file_jl_fname_lbl,                   row, 1, 1, 2)
        ftl.addWidget(self.file_jl_fname,                       row, 3, 1, 2)
        ftl.addWidget(self.file_jl_fname_button,                row, 5)
        ftl.addWidget(self.file_jl_session_button,              row, 6, 1, 2)

        row += 1
        ftl.addWidget(self.file_jl_explog_lbl,                  row, 1, 1, 2)
        ftl.addWidget(self.file_jl_explog,                      row, 3, 1, 2)
        ftl.addWidget(self.file_jl_explog_button,               row, 5)

        if self.dim in (3, 4):
            # dummy lbl
            row += 1
            dummy_lbl = QLabel('')
            ftl.addWidget(dummy_lbl,                            row, 0)

        self.file_tab.layout = ftl
        self.file_tab.setLayout(ftl)
        self.tabs.addTab(self.file_tab, 'File')

    def set_opened_viewers_list(self) -> None:
        """
        Set the list of all opened **DataViewers** of the same type (2D or 3D)
        and fill multi-choice :class:`CheckComboBox` with it, making them
        available for linking (see :meth:`link_selected_windows` for more
        details).
        """

        list = self.link_windows_list
        dv = self.mw.db.data_viewers
        for dvi in dv.keys():
            lbl = dv[dvi].index.split('/')[-1]
            if self.dim == dv[dvi].util_panel.dim:
                list.addItem(lbl)
                list.model().item(list.count() - 1).setCheckable(True)

    def link_selected_windows(self) -> None:
        """
        Link selected windows, to allow for simultaneous change of their
        sliders. Linking functionality operates in a *master-slave* system,
        where one (master) window controls all the others. *Master* window is
        established as the window from which **Link button** was clicked. All
        other windows receive status *slave*. Windows added to existing
        *linked* combination will also receive status *slave*.
        """

        # Status indices [0, 1, 2] => ['free', 'master', 'slave']
        # set status in linking hierarchy
        if self.link_windows_status.currentIndex() == 0:
            self.set_linking_status(self.get_linked_windows())
            if self.link_windows_status.currentIndex() == 2:
                self.link_windows.setText('Unlink')
                self.update_lists_in_other_viewers('free to slave')
                # disable all list items and uncheck them
                for idx in range(self.link_windows_list.count()):
                    self.link_windows_list.model().item(idx).setEnabled(False)
                return

        if self.link_windows_status.currentIndex() == 1:
            if 1 in self.set_linking_status(self.get_linked_windows(),
                                            get_statuses=True):
                two_master_box = QMessageBox()
                two_master_box.setIcon(QMessageBox.Information)
                two_master_box.setText('Cannot link two master viewers.')
                two_master_box.setStandardButtons(QMessageBox.Ok)
                if two_master_box.exec() == QMessageBox.Ok:
                    return
            if self.get_linked_windows():
                self.link_windows.setText('Update')
            else:
                self.link_windows.setText('Link')
                self.link_windows_status.setCurrentIndex(0)
            self.update_lists_in_other_viewers('as master')
        else:
            self.link_windows.setText('Link')
            self.update_lists_in_other_viewers('as slave')
            # enable all list items and uncheck them
            for idx in range(self.link_windows_list.count()):
                self.link_windows_list.model().item(idx).setEnabled(True)
                self.link_windows_list.setItemCheckState(idx, 0)
            self.link_windows_status.setCurrentIndex(0)

    def get_index_in_windows_list(self, wins_list: CheckComboBox) -> int:
        """
        Get index of the current **DataViewer** in the other **DataViewer's**
        record.

        :param wins_list: other **DataViewer** record of opened windows
        :return: index at which current window occurs in the other record
        """

        for idx in range(wins_list.count()):
            if self.mw.title == wins_list.itemText(idx):
                return idx

    def get_linked_windows(self) -> list:
        """
        Get list of all **DataViewers** currently for linking.

        :return: list of **DataViewer's** names
        """

        linked_windows = []
        for idx in range(self.link_windows_list.count()):
            if self.link_windows_list.itemCheckState(idx):
                linked_windows.append(self.link_windows_list.itemText(idx))
        return linked_windows

    def set_linking_status(self, windows_to_link: list,
                           get_statuses: bool = False) -> Union[list, None]:
        """
        Check statuses of all the **DataViewers** that are about to be linked
        together. If no *master* is found, make this **DataViewer** a *master*.

        :param windows_to_link: list of **DataViewer** to link
        :param get_statuses: if :py:obj:`True`, just return list of
                             **DataViewer** statuses
        :return: list of **DataViewer** statuses if ``get_statuses`` is
                 :py:obj:`True`, otherwise :py:obj:`None`
        """

        dv = self.mw.db.data_viewers
        statuses = []
        for dvi in dv.keys():
            if dv[dvi].title in windows_to_link:
                statuses.append(dv[dvi].util_panel.
                                link_windows_status.currentIndex())
        if get_statuses:
            return statuses
        else:
            if (1 in statuses) or (2 in statuses):
                self.link_windows_status.setCurrentIndex(2)
            else:
                self.link_windows_status.setCurrentIndex(1)

    def update_lists_in_other_viewers(self, action: str) -> None:
        """
        After linking and determining **DataViewer's** statuses, update
        :class:`CheckComboBox` list for all linked **DataViewers** to reflect
        established relationships.

        :param action: determines established status, takes values:
                       `as master`, `as slave` and `free to slave`
        """

        dv = self.mw.db.data_viewers
        linked_list = self.get_linked_windows()
        for dvi in dv.keys():
            dv_up = dv[dvi].util_panel
            if (action == 'free to slave') and (dv[dvi].title in linked_list):
                if dv_up.link_windows_status.currentIndex() == 1:
                    master = dv_up
                else:
                    linked2 = dv_up.get_linked_windows()
                    for dvj in dv.keys():
                        dvj_up = dv[dvj].util_panel
                        if (dvj_up.link_windows_status.currentIndex() == 1) \
                                and (dv[dvj].title in linked2):
                            master = dvj_up
                idx = self.get_index_in_windows_list(master.link_windows_list)
                master.link_windows_list.setItemCheckState(idx, 2)
                master.link_selected_windows()
                break
            elif (action == 'as master') and (dv[dvi].title in linked_list):
                dv_up.link_windows.setText('Unlink')
                viewer_linked_list = dv_up.link_windows_list
                # "enslave" checked viewers
                idx = self.get_index_in_windows_list(viewer_linked_list)
                viewer_linked_list.setItemCheckState(idx, 2)
                dv_up.link_windows_status.setCurrentIndex(2)
                for idx in range(viewer_linked_list.count()):
                    viewer_linked_list.model().item(idx).setEnabled(False)
                    if viewer_linked_list.itemText(idx) in linked_list:
                        viewer_linked_list.setItemCheckState(idx, 2)
            elif (action == 'as master') and \
                    not (dv[dvi].title in linked_list):
                if self.mw.title in dv[dvi].util_panel.get_linked_windows():
                    dv_up.link_windows_status.setCurrentIndex(0)
                    for idx in range(dv_up.link_windows_list.count()):
                        dv_up.link_windows.setText('Link')
                        dv_up.link_windows_list.model().item(
                            idx).setEnabled(True)
                        dv_up.link_windows_list.setItemCheckState(idx, 0)
            elif (action == 'as slave') and (dv[dvi].title in linked_list):
                idx = self.get_index_in_windows_list(dv_up.link_windows_list)
                dv_up.link_windows_list.setItemCheckState(idx, 0)
                # make free if master and has no more linked windows
                if (dv_up.link_windows_status.currentIndex() == 1) and \
                        not dv_up.get_linked_windows():
                    dv_up.link_windows_status.setCurrentIndex(0)
                    dv_up.link_windows.setText('Link')

    def set_tabs_color(self) -> None:
        """
        Set QWidgets' colors by hand. This is for some reason required on
        Windows machines.
        """

        to_change = ['QPushButton', 'QLineEdit', 'QSpinBox', 'QDoubleSpinBox',
                     'QComboBox']
        for i in range(self.tabs.count()):
            tab_i = self.tabs.widget(i)
            tab_i.setStyleSheet("QWidget {background-color: rgb(64, 64, 64);}")
            for wi in tab_i.children():
                wi_name = wi.metaObject().className()
                if wi_name in to_change:
                    wi.setStyleSheet(
                        "QWidget {background-color: rgb(255, 255, 255);}")

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

    def set_orientation_info_window(self) -> None:
        """
        Set up the :class:`InfoWindow` with tips on how geometry at different
        beamlines corresponds to the way data are presented in :mod:`piva`.
        Very helpful for alignment during the experiment.
        """

        self.orient_info_window = QWidget()
        oiw = QtWidgets.QGridLayout()

        self.oi_window_lbl = QLabel('piva -> beamline coordinates translator')
        self.oi_window_lbl.setFont(bold_font)
        self.oi_beamline_lbl = QLabel('Beamline')
        self.oi_beamline_lbl.setFont(bold_font)
        self.oi_azimuth_lbl = QLabel('Azimuth (clockwise)')
        self.oi_azimuth_lbl.setFont(bold_font)
        self.oi_analyzer_lbl = QLabel('Analyzer (-> +)')
        self.oi_analyzer_lbl.setFont(bold_font)
        self.oi_scanned_lbl = QLabel('Scanned (-> +)')
        self.oi_scanned_lbl.setFont(bold_font)

        entries = [
            ['SIS (SLS, SIStem)',  'phi -> -',     'theta -> +', 'tilt -> -'],
            ['SIS (SLS, SES)',     'phi -> +',     'theta -> -', 'tilt -> -'],
            ['Bloch (MaxIV)',      'azimuth -> +', 'tilt -> -',  'polar -> -'],
            ['ALS (Merlin)',       '-',            '-',          '-'],
            ['CASSIOPEE (SOLEIL)', '-',            '-',          '-'],
            ['I05 (Diamond)',      '-',            '-',          '-'],
            ['URANOS (SOLARIS)',   '-',            'R3 -> ?',    'R1 -> -'],
            ['APE (Elettra)',      '-',            '-',          '-'],
            ['ADDRES (SLS)',       '-',            '-',          '-'],
            ['-',                  '-',            '-',          '-'],
            ['-',                  '-',            '-',          '-']]
        labels = {}

        row = 0
        oiw.addWidget(self.oi_beamline_lbl,     row, 0)
        oiw.addWidget(self.oi_azimuth_lbl,      row, 1)
        oiw.addWidget(self.oi_analyzer_lbl,     row, 2)
        oiw.addWidget(self.oi_scanned_lbl,      row, 3)

        for entry in entries:
            row += 1
            labels[str(row)] = {}
            labels[str(row)]['beamline'] = QLabel(entry[0])
            labels[str(row)]['azimuth'] = QLabel(entry[1])
            labels[str(row)]['azimuth'].setAlignment(QtCore.Qt.AlignCenter)
            labels[str(row)]['analyzer'] = QLabel(entry[2])
            labels[str(row)]['analyzer'].setAlignment(QtCore.Qt.AlignCenter)
            labels[str(row)]['scanned'] = QLabel(entry[3])
            labels[str(row)]['scanned'].setAlignment(QtCore.Qt.AlignCenter)

            oiw.addWidget(labels[str(row)]['beamline'],    row, 0)
            oiw.addWidget(labels[str(row)]['azimuth'],     row, 1)
            oiw.addWidget(labels[str(row)]['analyzer'],    row, 2)
            oiw.addWidget(labels[str(row)]['scanned'],     row, 3)

        self.orient_info_window.layout = oiw
        self.orient_info_window.setLayout(oiw)

    def set_metadata_window(self, dataset: Dataset) -> None:
        """
        Set up the :class:`InfoWindow` with all metadata available from opened
        data file and held in :class:`~data_loaders.Dataset` object.

        :param dataset: loaded dataset with available metadata
        """

        self.md_window = QWidget()
        mdw = QtWidgets.QGridLayout()

        attribute_name_lbl = QLabel('Attribute')
        attribute_name_lbl.setFont(bold_font)
        attribute_value_lbl = QLabel('Value')
        attribute_value_lbl.setFont(bold_font)
        attribute_value_lbl.setAlignment(QtCore.Qt.AlignCenter)
        attribute_saved_lbl = QLabel('user saved')
        attribute_saved_lbl.setFont(bold_font)
        attribute_saved_lbl.setAlignment(QtCore.Qt.AlignCenter)

        dataset = vars(dataset)
        entries = {}

        row = 0
        mdw.addWidget(attribute_name_lbl,   row, 0)
        mdw.addWidget(attribute_value_lbl,  row, 1)

        row = 1
        for key in dataset.keys():
            if (key == 'ekin') or (key == 'saved') or \
                    (key == 'data_provenance') or \
                    (key == 'add_org_file_entry'):
                continue
            elif key == 'data':
                s = dataset[key].shape
                value = '(' + str(s[0]) + ',  ' + str(s[1]) + ',  ' + \
                        str(s[2]) + ')'
                entries[str(row)] = {}
                entries[str(row)]['name'] = QLabel(key)
                entries[str(row)]['value'] = QLabel(str(value))
                entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
            elif key == 'xscale':
                value = '({:.2f}  :  {:.2f})'.format(dataset[key][0],
                                                     dataset[key][-1])
                entries[str(row)] = {}
                entries[str(row)]['name'] = QLabel(key)
                entries[str(row)]['value'] = QLabel(str(value))
                entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
            elif key == 'yscale':
                value = '({:.4f}  :  {:.4f})'.format(dataset[key][0],
                                                     dataset[key][-1])
                entries[str(row)] = {}
                entries[str(row)]['name'] = QLabel(key)
                entries[str(row)]['value'] = QLabel(str(value))
                entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
            elif key == 'zscale':
                value = '({:.4f}  :  {:.4f})'.format(dataset[key][0],
                                                     dataset[key][-1])
                entries[str(row)] = {}
                entries[str(row)]['name'] = QLabel(key)
                entries[str(row)]['value'] = QLabel(str(value))
                entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
            elif key == 'kxscale':
                if not (dataset[key] is None):
                    value = '({:.3f}  :  {:.3f})'.format(dataset[key][0],
                                                         dataset[key][-1])
                    entries[str(row)] = {}
                    entries[str(row)]['name'] = QLabel(key)
                    entries[str(row)]['value'] = QLabel(str(value))
                    entries[str(row)]['value'].setAlignment(
                        QtCore.Qt.AlignCenter)
                else:
                    entries[str(row)] = {}
                    entries[str(row)]['name'] = QLabel(key)
                    entries[str(row)]['value'] = QLabel(str(dataset[key]))
                    entries[str(row)]['value'].setAlignment(
                        QtCore.Qt.AlignCenter)
            elif key == 'kyscale':
                if not (dataset[key] is None):
                    value = '({:.3f}  :  {:.3f})'.format(dataset[key][0],
                                                         dataset[key][-1])
                    entries[str(row)] = {}
                    entries[str(row)]['name'] = QLabel(key)
                    entries[str(row)]['value'] = QLabel(str(value))
                    entries[str(row)]['value'].setAlignment(
                        QtCore.Qt.AlignCenter)
                else:
                    entries[str(row)] = {}
                    entries[str(row)]['name'] = QLabel(key)
                    entries[str(row)]['value'] = QLabel(str(dataset[key]))
                    entries[str(row)]['value'].setAlignment(
                        QtCore.Qt.AlignCenter)
            elif key == 'pressure':
                if dataset[key] is None:
                    continue
                else:
                    entries[str(row)] = {}
                    entries[str(row)]['name'] = QLabel(key)
                    entries[str(row)]['value'] = QLabel(
                        '{:.4e}'.format((dataset[key])))
                    entries[str(row)]['value'].setAlignment(
                        QtCore.Qt.AlignCenter)
            elif key == 'scan_dim':
                if dataset[key] is None or dataset[key] == []:
                    continue
                else:
                    entries[str(row)] = {}
                    entries[str(row)]['name'] = QLabel(key)
                    entries[str(row)]['value'] = QLabel(
                        '[{:.3f}, {:.3f}, {:.3f}]'.format(dataset[key][0],
                                                          dataset[key][1],
                                                          dataset[key][2]))
                    entries[str(row)]['value'].setAlignment(
                        QtCore.Qt.AlignCenter)
            else:
                entries[str(row)] = {}
                entries[str(row)]['name'] = QLabel(key)
                if isinstance(dataset[key], float):
                    entries[str(row)]['value'] = QLabel(
                        '{:.4f}'.format((dataset[key])))
                else:
                    entries[str(row)]['value'] = QLabel(str(dataset[key]))
                entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)

            mdw.addWidget(entries[str(row)]['name'],    row, 0)
            mdw.addWidget(entries[str(row)]['value'],   row, 1)
            row += 1

        if 'saved' in dataset.keys():
            mdw.addWidget(attribute_saved_lbl,   row, 0, 1, 2)
            for key in dataset['saved'].keys():
                row += 1
                entries[str(row)] = {}
                entries[str(row)]['name'] = QLabel(key)
                if key == 'kx' or key == 'ky' or key == 'k':
                    value = '({:.2f}  :  {:.2f})'.format(
                        dataset['saved'][key][0], dataset['saved'][key][-1])
                    entries[str(row)]['value'] = QLabel(str(value))
                else:
                    entries[str(row)]['value'] = QLabel(
                        str(dataset['saved'][key]))
                entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)

                mdw.addWidget(entries[str(row)]['name'],    row, 0)
                mdw.addWidget(entries[str(row)]['value'],   row, 1)

        self.md_window.layout = mdw
        self.md_window.setLayout(mdw)

    def show_metadata_window(self) -> None:
        """
        Display metadata window. See :meth:`set_metadata_window` for more
        details.
        """

        if self.dim in (2, 3):
            self.set_metadata_window(self.mw.data_set)
        elif self.dim == 4:
            self.set_metadata_window(self.mw.scan[0, 0])
        title = self.mw.title + ' - metadata'
        self.info_box = InfoWindow(self.md_window, title)
        self.info_box.setMinimumWidth(350)
        self.info_box.show()

    def add_metadata(self) -> None:
        """
        Add/update metadata in the :class:`~.data_loaders.Dataset` object,
        which can be further saved into :mod:`pickle` file.
        """

        name = self.file_md_name.text()
        value = self.file_md_value.text()
        try:
            value = float(value)
        except ValueError:
            pass

        if name == '':
            empty_name_box = QMessageBox()
            empty_name_box.setIcon(QMessageBox.Information)
            empty_name_box.setText('Attribute\'s name not given.')
            empty_name_box.setStandardButtons(QMessageBox.Ok)
            if empty_name_box.exec() == QMessageBox.Ok:
                return

        message = 'Sure to add attribute \'{}\' with value <{}> (type: {}) ' \
                  'to the file?'.format(name, value, type(value))
        sanity_check_box = QMessageBox()
        sanity_check_box.setIcon(QMessageBox.Question)
        sanity_check_box.setText(message)
        sanity_check_box.setStandardButtons(QMessageBox.Ok |
                                            QMessageBox.Cancel)
        if sanity_check_box.exec() == QMessageBox.Ok:
            if hasattr(self.mw.data_set, name):
                old = vars(self.mw.data_set)[name]
                attr_conflict_box = QMessageBox()
                attr_conflict_box.setIcon(QMessageBox.Question)
                attr_conflict_box.setText(f'Data set already has attribute '
                                          f'\'{name}\'.  Overwrite?')
                attr_conflict_box.setStandardButtons(QMessageBox.Ok |
                                                     QMessageBox.Cancel)
                if attr_conflict_box.exec() == QMessageBox.Ok:
                    setattr(self.mw.data_set, name, value)
                    self.dp_add_edited_metadata_entry('updated', name,
                                                      old, value)
            else:
                dl.update_namespace(self.mw.data_set, [name, value])
                self.dp_add_edited_metadata_entry('added', name, '-', value)
        else:
            return

    def remove_metadata(self) -> None:
        """
        Remove some metadata from :class:`~data_loaders.Dataset` object.
        """

        name = self.file_md_name.text()

        if not hasattr(self.mw.data_set, name):
            no_attr_box = QMessageBox()
            no_attr_box.setIcon(QMessageBox.Information)
            no_attr_box.setText(f'Attribute \'{name}\' not found.')
            no_attr_box.setStandardButtons(QMessageBox.Ok)
            if no_attr_box.exec() == QMessageBox.Ok:
                return

        if name in ['data', 'xscale', 'yscale', 'zscale']:
            essential_md_box = QMessageBox()
            essential_md_box.setIcon(QMessageBox.Information)
            essential_md_box.setText(f'Sorry, no can do.  '
                                     f'Data and axes cannot be removed.')
            essential_md_box.setStandardButtons(QMessageBox.Ok)
            if essential_md_box.exec() == QMessageBox.Ok:
                return

        message = 'Sure to remove attribute \'{}\' from the data set?'.format(
            name)
        sanity_check_box = QMessageBox()
        sanity_check_box.setIcon(QMessageBox.Question)
        sanity_check_box.setText(message)
        sanity_check_box.setStandardButtons(QMessageBox.Ok |
                                            QMessageBox.Cancel)
        if sanity_check_box.exec() == QMessageBox.Ok:
            value = vars(self.mw.data_set)[name]
            delattr(self.mw.data_set, name)
            self.dp_add_edited_metadata_entry('removed', name, value, '-')
        else:
            return

    def sum_datasets(self) -> None:
        """
        Convenient one-click button to sum datasets from separate files. Sum up
        also number of sweeps ``Dataset.n_sweeps`` to keep track of the real
        acquisition time.

        Method check available metadata to make sure they were acquired under
        the same conditions.

        .. note::
            While running longer acquisition scans for higher statistics,
            it's a good practise to split up measurement into few shorter
            repetitions, to avoid possible software crashes, beam dumps,
            sample damage, *etc.*. This method allows for quick summation of
            such acquired files.
        """

        file_path = self.mw.fname[:-len(self.mw.title)] + \
                    self.file_sum_datasets_fname.text()
        org_dataset = dl.load_data(self.mw.fname)
        new_dataset, check_result = None, None

        try:
            new_dataset = dl.load_data(file_path)
        except FileNotFoundError:
            no_file_box = QMessageBox()
            no_file_box.setIcon(QMessageBox.Information)
            no_file_box.setText('File not found.')
            no_file_box.setStandardButtons(QMessageBox.Ok)
            if no_file_box.exec() == QMessageBox.Ok:
                return

        try:
            check_result = self.check_conflicts([org_dataset, new_dataset])
        except AttributeError:
            error_box = QMessageBox()
            error_box.setIcon(QMessageBox.Information)
            error_box.setText('Aborted, datasets could not be compared.')
            error_box.setStandardButtons(QMessageBox.Ok)
            if error_box.exec() == QMessageBox.Ok:
                return

        if check_result == 0:
            data_mismatch_box = QMessageBox()
            data_mismatch_box.setIcon(QMessageBox.Information)
            data_mismatch_box.setText('Aborted.\n'
                                      'Data sets\' shapes don\'t match.\n')
            data_mismatch_box.setStandardButtons(QMessageBox.Ok)
            if data_mismatch_box.exec() == QMessageBox.Ok:
                return

        check_result_box = QMessageBox()
        check_result_box.setMinimumWidth(600)
        check_result_box.setMaximumWidth(1000)
        check_result_box.setIcon(QMessageBox.Information)
        check_result_box.setText(check_result)
        check_result_box.setStandardButtons(QMessageBox.Ok |
                                            QMessageBox.Cancel)
        if check_result_box.exec() == QMessageBox.Ok:
            self.mw.org_dataset = org_dataset
            self.mw.data_set.data += new_dataset.data
            self.mw.data_set.n_sweeps += new_dataset.n_sweeps
            d = np.swapaxes(self.mw.data_set.data, 1, 2)
            self.mw.data_handler.set_data(d)
            self.mw.set_image(self.mw.data_handler.get_data())
            self.dp_add_file_sum_entry(file_path)
        else:
            return

    def reset_summation(self) -> None:
        """
        Reset summation from different files. See :meth:`sum_datasets` for
        more details.
        """

        if self.mw.org_dataset is None:
            no_summing_yet_box = QMessageBox()
            no_summing_yet_box.setIcon(QMessageBox.Information)
            no_summing_yet_box.setText('No summing done yet.')
            no_summing_yet_box.setStandardButtons(QMessageBox.Ok)
            if no_summing_yet_box.exec() == QMessageBox.Ok:
                return

        reset_summation_box = QMessageBox()
        reset_summation_box.setMinimumWidth(600)
        reset_summation_box.setMaximumWidth(1000)
        reset_summation_box.setIcon(QMessageBox.Question)
        reset_summation_box.setText('Want to reset summation?')
        reset_summation_box.setStandardButtons(QMessageBox.Ok |
                                               QMessageBox.Cancel)
        if reset_summation_box.exec() == QMessageBox.Ok:
            self.mw.data_set.data = self.mw.org_dataset.data
            self.mw.data_set.n_sweeps = self.mw.org_dataset.n_sweeps
            d = np.swapaxes(self.mw.data_set.data, 1, 2)
            self.mw.data_handler.set_data(d)
            self.mw.set_image(self.mw.data_handler.get_data())
        else:
            return

    def open_jl_session(self) -> None:
        """
        Start new ``jupyter-lab`` session.
        """

        if self.mw.db.jl_session_running:
            jl_running_box = QMessageBox()
            jl_running_box.setIcon(QMessageBox.Information)
            jl_running_box.setText('JupyterLab session is already running.\n'
                                   'Want to start another one?')
            jl_running_box.setStandardButtons(QMessageBox.Ok |
                                               QMessageBox.Cancel)
            if jl_running_box.exec() == QMessageBox.Cancel:
                return

        directory = str(QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select Directory', self.mw.fname[:-len(self.mw.title)]))

        # Open jupyter notebook as a subprocess
        openJupyter = "jupyter lab"
        subprocess.Popen(openJupyter, shell=True, cwd=directory)

        self.mw.db.jl_session_running = True

    def create_jl_file(self):
        """
        Create new ``jupyter`` notebook file  containing imported data and
        currently displayed images and curves.
        """

        directory = str(QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select Directory', self.mw.fname[:-len(self.mw.title)]))

        fname = directory + '/' + self.file_jl_fname.text()

        if os.path.isfile(fname):
            file_exists_box = QMessageBox()
            file_exists_box.setIcon(QMessageBox.Information)
            file_exists_box.setText('File already exists.\nOverwrite?')
            file_exists_box.setStandardButtons(QMessageBox.Ok |
                                               QMessageBox.Cancel)
            if file_exists_box.exec() == QMessageBox.Cancel:
                return
        os.system('touch ' + fname)

        root_dir = os.path.dirname(os.path.abspath(__file__))
        template = root_dir + '/ipynb_templates/template.ipynb'
        templ_file = open(template, 'r')
        templ_lines = templ_file.readlines()
        templ_file.close()

        # writing to file
        new_lines = []
        for line in templ_lines:
            if 'path = ' in line:
                line = '    "path = \'{}\'\\n",'.format(
                    self.mw.fname[:-len(self.mw.title)])
            if 'fname = ' in line:
                line = '    "fname = \'{}\'\\n",'.format(self.mw.title)
            if 'slit_idx, e_idx =' in line:
                if self.dim == 2:
                    line = '    "slit_idx, e_idx = {}, {}\\n",'.format(
                        self.momentum_hor.value(), self.energy_vert.value())
                    line = line + '    "bm = data[0, :, :]\\n",\n'
                    line = line + '    "plot_x = ' \
                                  'data[0, slit_idx, e_idx]\\n",\n'
                    line = line + '    "plot_y = data[0, :, e_idx]"\n'
                elif self.dim == 3:
                    line = '    "scan_idx, slit_idx, e_idx = ' \
                           '{}, {}, {}\\n",'.format(
                        self.momentum_vert.value(),
                        self.momentum_hor.value(),
                        self.energy_vert.value())
                    line = line + '    "main_cut = data[:, :, e_idx]\\n",\n'
                    line = line + '    "cut_x = data[:, slit_idx, :]\\n",\n'
                    line = line + '    "cut_y = data[scan_idx, :, :]\\n",\n'
                    line = line + '    "plot_x = ' \
                                  'data[:, slit_idx, e_idx]\\n",\n'
                    line = line + '    "plot_y = data[scan_idx, :, e_idx]"\n'
            new_lines.append(line)

        new_file = open(fname, 'w')
        new_file.writelines(new_lines)
        new_file.close()

    def create_experimental_logbook_file(self) -> None:
        """
        Create new ``jupyter`` notebook file allowing for generating
        experimental logbook for selected beamline.
        """

        beamline = self.file_jl_explog.currentText()
        if beamline == '--beamline--':
            no_bealine_box = QMessageBox()
            no_bealine_box.setIcon(QMessageBox.Information)
            no_bealine_box.setText('Select a beamline.')
            no_bealine_box.setStandardButtons(QMessageBox.Ok)
            if no_bealine_box.exec() == QMessageBox.Ok:
                return
        else:
            directory = str(QtWidgets.QFileDialog.getExistingDirectory(
                self, 'Select Directory', self.mw.fname[:-len(self.mw.title)]))
            fname = '{}/metadata-{}.ipynb'.format(directory, beamline)

        if os.path.isfile(fname):
            file_exists_box = QMessageBox()
            file_exists_box.setIcon(QMessageBox.Information)
            file_exists_box.setText('File already exists.\nOverwrite?')
            file_exists_box.setStandardButtons(QMessageBox.Ok |
                                               QMessageBox.Cancel)
            if file_exists_box.exec() == QMessageBox.Cancel:
                return
        os.system('touch ' + fname)

        root_dir = os.path.dirname(os.path.abspath(__file__))
        template = root_dir + '/ipynb_templates/template-metadata.ipynb'
        templ_file = open(template, 'r')
        templ_lines = templ_file.readlines()
        templ_file.close()

        # writing to file
        new_lines = []
        for line in templ_lines:
            if 'from piva.data_loader' in line:
                line = '    "from piva.data_loaders import Dataloader{} ' \
                       'as dl\\n",'.format(beamline)
            new_lines.append(line)

        new_file = open(fname, 'w')
        new_file.writelines(new_lines)
        new_file.close()

    @ staticmethod
    def dp_add_file_cut_entry(data_set: Dataset, direction: str,
                              cut_idx: int, binned: int) -> None:
        """
        Add an entry to ``Dataset.data_provenance`` :py:obj:`dict` with
        information on original :class:`~data_loaders.Dataset`, when new
        :class:`~data_loaders.Dataset` is generated from a slice of data.

        :param data_set: new :class:`~data_loaders.Dataset` object
        :param direction: direction along which slice was taken
        :param cut_idx: index at which slice was taken
        :param binned: number of bins (+/-) over which slice was summed up
        """

        dp = data_set.data_provenance

        entry = {'index': len(dp['file']),
                 'date_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 'path': '-',
                 'type': 'cut along {}'.format(direction),
                 'index_taken': cut_idx,
                 'binned': binned,
                 'data_loader': '-'}

        dp['file'].append(entry)

    def dp_add_file_sum_entry(self, fname: str) -> None:
        """
        Add an entry to ``Dataset.data_provenance`` :py:obj:`dict` with
        information on data added through summing the files. See
        :meth:`sum_datasets` for more details.

        :param fname: name of the file added to original
                      :class:`~data_loaders.Dataset`
        """

        dp = self.mw.data_set.data_provenance

        entry = {'index': len(dp['file']),
                 'date_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 'path': fname,
                 'type': 'added dataset',
                 'index_taken': '-',
                 'binned': '-',
                 'data_loader': '-'}

        dp['file'].append(entry)

    def dp_add_k_space_conversion_entry(self, dataset: Dataset) -> None:
        """
        Add an entry to `Dataset.data_provenance` :py:obj:`dict` with
        information on performed *k*-space conversion.

        :param dataset: loaded dataset with available metadata
        """

        dp = dataset.data_provenance
        orient = self.axes_slit_orient.currentText()
        Ef = self.axes_energy_Ef.value()
        hv = self.axes_energy_hv.value()
        wf = self.axes_energy_wf.value()

        if np.isclose(self.axes_conv_lc.value(), 3.1416):
            a = '-'
        else:
            a = self.axes_conv_lc.value()

        if self.dim in (2, 4):
            ana_off = self.axes_gamma_x.value()
            scan_off = self.axes_angle_off.value()
            kz = '-'
            c = '-'
        else:
            ana_off = self.axes_gamma_y.value()
            scan_off = self.axes_gamma_x.value()
            if self.axes_transform_kz.isChecked():
                kz = 'yes'
                if np.isclose(self.axes_conv_lc_op.value(), 3.1416):
                    c = '-'
                else:
                    c = self.axes_conv_lc_op.value()
            else:
                kz = '-'
                c = '-'

        entry = {'index': len(dp['k_space_conv']),
                 'date_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 'ana_ax_off': ana_off,
                 'scan_ax_off': scan_off,
                 'orient': orient,
                 'kz': kz,
                 'Ef': Ef,
                 'hv': hv,
                 'wf': wf,
                 'a': a,
                 'c': c}

        dp['k_space_conv'].append(entry)

    def dp_add_edited_metadata_entry(self, action: str, attr_name: str,
                                     old: Any, new: Any) -> None:
        """
        Add an entry to ``Dataset.data_provenance`` :py:obj:`dict` with
        information on adeed/updated/removed metadata.

        :param action: what action was taken, can be [`adeed`, `updated`,
                       `removed`]
        :param attr_name: name of the changed :class:`~data_loaders.Dataset`
                          attribute
        :param old: old value of the attribute
        :param new: new value of the attribute
        """

        dp = self.mw.data_set.data_provenance

        entry = {'index': len(dp['edited_entries']),
                 'date_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 'action': action,
                 'attribute': attr_name,
                 'old_value': old,
                 'new_value': new}

        dp['edited_entries'].append(entry)

    def set_dp_window(self, width: int) -> None:
        """
        Set up the :class:`InfoWindow` with all entries in
        ``Dataset.data_provenance``.

        :param width: width of the displayed window
        """

        self.dp_window = QWidget()
        dpw = QVBoxLayout()
        if self.dim in (2, 3):
            dp = self.mw.data_set.data_provenance
        elif self.dim == 4:
            dp = self.mw.scan[0, 0].data_provenance

        file_entries = ['#', 'Date & time', 'path', 'type', 'taken @',
                        'n_bins', 'Data loader']
        kspc_entries = ['#', 'Date & time', 'Analyzer axis offset',
                        'Scanned axis offset', 'Slit orientation', 'kz', 'Ef',
                        'hv', 'wf', 'a, A', 'c, A']
        edit_entries = ['#', 'Date & time', 'Action', 'Attribute', 'Old value',
                        'New Value']

        file_lbl = QLabel('File')
        kspc_lbl = QLabel('k-space conversion')
        edit_lbl = QLabel('Editted entries')
        file_lbl.setFont(bold_font)
        kspc_lbl.setFont(bold_font)
        edit_lbl.setFont(bold_font)
        file_lbl.setAlignment(QtCore.Qt.AlignCenter)
        kspc_lbl.setAlignment(QtCore.Qt.AlignCenter)
        edit_lbl.setAlignment(QtCore.Qt.AlignCenter)

        dpw.addWidget(file_lbl)
        if len(dp['file']) != 0:
            file_table = QTableWidget(len(dp['file']), len(file_entries))
            file_table.setSizePolicy(QSizePolicy.Maximum,
                                     QSizePolicy.Maximum)
            file_table.setHorizontalHeaderLabels(file_entries)
            file_table.setMinimumWidth(width)
            height = int(file_table.horizontalHeader().height() * 1.2)
            for idx in range(len(dp['file'])):
                dpfi = dp['file'][idx]
                file_table.setItem(idx, 0, QTabItem(str(dpfi['index'])))
                file_table.setItem(idx, 1, QTabItem(str(dpfi['date_time'])))
                file_table.setItem(idx, 2, QTabItem(str(dpfi['path'])))
                file_table.setItem(idx, 3, QTabItem(str(dpfi['type'])))
                file_table.setItem(idx, 4, QTabItem(str(dpfi['index_taken'])))
                file_table.setItem(idx, 5, QTabItem(str(dpfi['binned'])))
                file_table.setItem(idx, 6, QTabItem(str(dpfi['data_loader'])))
                height += file_table.rowHeight(idx)
            file_table.resizeColumnsToContents()
            file_table.setColumnWidth(0, 30)
            header = file_table.horizontalHeader()
            file_table.setMaximumHeight(height)
            file_table.setMaximumWidth(header.width())
            dpw.addWidget(file_table)

        dpw.addWidget(kspc_lbl)
        if len(dp['k_space_conv']) != 0:
            kspc_table = QTableWidget(len(dp['k_space_conv']),
                                      len(kspc_entries))
            kspc_table.setHorizontalHeaderLabels(kspc_entries)
            kspc_table.setMinimumWidth(width)
            height = int(kspc_table.horizontalHeader().height() * 1.2)
            for idx in range(len(dp['k_space_conv'])):
                dpfi = dp['k_space_conv'][idx]
                kspc_table.setItem(idx, 0, QTabItem(str(dpfi['index'])))
                kspc_table.setItem(idx, 1, QTabItem(str(dpfi['date_time'])))
                kspc_table.setItem(idx, 2, QTabItem(str(dpfi['ana_ax_off'])))
                kspc_table.setItem(idx, 3, QTabItem(str(dpfi['scan_ax_off'])))
                kspc_table.setItem(idx, 4, QTabItem(str(dpfi['orient'])))
                kspc_table.setItem(idx, 5, QTabItem(str(dpfi['kz'])))
                kspc_table.setItem(idx, 6, QTabItem(str(dpfi['Ef'])))
                kspc_table.setItem(idx, 7, QTabItem(str(dpfi['hv'])))
                kspc_table.setItem(idx, 8, QTabItem(str(dpfi['wf'])))
                kspc_table.setItem(idx, 9, QTabItem(str(dpfi['a'])))
                kspc_table.setItem(idx, 10, QTabItem(str(dpfi['c'])))
                height += kspc_table.rowHeight(idx)
            kspc_table.resizeColumnsToContents()
            kspc_table.setColumnWidth(0, 30)
            header = kspc_table.horizontalHeader()
            kspc_table.setMaximumHeight(height)
            kspc_table.setMaximumWidth(header.width())
            dpw.addWidget(kspc_table)

        dpw.addWidget(edit_lbl)
        if len(dp['edited_entries']) != 0:
            edit_table = QTableWidget(len(dp['edited_entries']),
                                      len(edit_entries))
            edit_table.setHorizontalHeaderLabels(edit_entries)
            edit_table.setMinimumWidth(width)
            height = int(edit_table.horizontalHeader().height() * 1.2)
            for idx in range(len(dp['edited_entries'])):
                dpfi = dp['edited_entries'][idx]
                edit_table.setItem(idx, 0, QTabItem(str(dpfi['index'])))
                edit_table.setItem(idx, 1, QTabItem(str(dpfi['date_time'])))
                edit_table.setItem(idx, 2, QTabItem(str(dpfi['action'])))
                edit_table.setItem(idx, 3, QTabItem(str(dpfi['attribute'])))
                edit_table.setItem(idx, 4, QTabItem(str(dpfi['old_value'])))
                edit_table.setItem(idx, 5, QTabItem(str(dpfi['new_value'])))
                height += edit_table.rowHeight(idx)
            edit_table.resizeColumnsToContents()
            edit_table.setColumnWidth(0, 30)
            header = edit_table.horizontalHeader()
            edit_table.setMaximumHeight(height)
            edit_table.setMaximumWidth(header.width())
            dpw.addWidget(edit_table)

        self.dp_window.layout = dpw
        self.dp_window.setLayout(dpw)

    def show_data_provenance_window(self) -> None:
        """
        Display **data provenance** window.

        .. note::
            Data provenance object contains the record of all changes
            performed on the data from the raw file up to this point.
            Helps to keep track of how and why data were analyzed.
        """

        width = 800
        self.set_dp_window(width)
        title = self.mw.title + ' - data provenance'
        self.dp_box = InfoWindow(self.dp_window, title)
        self.dp_box.setMinimumWidth(width + 100)
        self.dp_box.show()

    @staticmethod
    def check_conflicts(datasets: list) -> str:
        """
        Prior to summation of the datasets, compare available metadata to
        check for possible conflicts in conditions under which data were
        recorded.

        :param datasets: list of [`org_dataset`, `appended_dataset`]
        :return: message with an information whether everything matches or
                 what conflict in which parameters were detected
        """

        labels = ['fname', 'data', 'T', 'hv', 'polarization', 'PE', 'FE',
                  'exit', 'x', 'y', 'z', 'theta', 'phi', 'tilt', 'lens_mode',
                  'acq_mode', 'e_start', 'e_stop', 'e_step']
        to_check = [[] for _ in range(len(labels))]

        to_check[0] = ['original', 'new']
        for ds in datasets:
            to_check[1].append(ds.data.shape)
            to_check[2].append(ds.temp)
            to_check[3].append(ds.hv)
            to_check[4].append(ds.polarization)
            to_check[5].append(ds.PE)
            to_check[6].append(ds.FE)
            to_check[7].append(ds.exit_slit)
            to_check[8].append(ds.x)
            to_check[9].append(ds.y)
            to_check[10].append(ds.z)
            to_check[11].append(ds.theta)
            to_check[12].append(ds.phi)
            to_check[13].append(ds.tilt)
            to_check[14].append(ds.lens_mode)
            to_check[15].append(ds.acq_mode)
            to_check[16].append(ds.zscale[0])
            to_check[17].append(ds.zscale[-1])
            to_check[18].append(wp.get_step(ds.zscale))

        # check if imporatnt stuff match
        check_result = []
        for idx, lbl in enumerate(labels):
            if lbl == 'fname' or lbl == 'data':
                check_result.append(True)
            elif lbl == 'T':
                err = 1
                par = np.array(to_check[idx])
                try:
                    to_compare = np.ones(par.size) * par[0]
                    check_result.append(np.allclose(par, to_compare, atol=err))
                except TypeError:
                    pass
            elif lbl == 'hv':
                err = 0.1
                par = np.array(to_check[idx])
                try:
                    to_compare = np.ones(par.size) * par[0]
                    check_result.append(np.allclose(par, to_compare, atol=err))
                except TypeError:
                    pass
            elif lbl == 'e_start':
                err = to_check[-1][0]
                par = np.array(to_check[idx])
                try:
                    to_compare = np.ones(par.size) * par[0]
                    check_result.append(np.allclose(par, to_compare, atol=err))
                except TypeError:
                    pass
            elif lbl == 'e_stop':
                err = to_check[-1][0]
                par = np.array(to_check[idx])
                try:
                    to_compare = np.ones(par.size) * par[0]
                    check_result.append(np.allclose(par, to_compare, atol=err))
                except TypeError:
                    pass
            elif lbl == 'e_step':
                err = to_check[-1][0]
                par = np.array(to_check[idx])
                try:
                    to_compare = np.ones(par.size) * par[0]
                    check_result.append(np.allclose(par, to_compare,
                                                    atol=err * 0.1))
                except TypeError:
                    pass
            else:
                check_result.append(to_check[idx][0] == to_check[idx][1])

        if not (to_check[1][0] == to_check[1][1]):
            return 0

        if np.array(check_result).all():
            message = 'Everything match! We\'re good to go!'
        else:
            message = 'Some stuff doesn\'t match...\n\n'
            dont_match = np.where(np.array(check_result) == False)
            for idx in dont_match[0]:
                try:
                    message += '{}\t{:.3f}\t {:.3f}\n'.format(
                        str(labels[idx]), to_check[idx][0], to_check[idx][1])
                except TypeError:
                    message += '{} \t\t {}\t  {} \n'.format(
                        str(labels[idx]), str(to_check[idx][0]),
                        str(to_check[idx][1]))
                except ValueError:
                    message += '{} \t\t {}\t  {} \n'.format(
                        str(labels[idx]), str(to_check[idx][0]),
                        str(to_check[idx][1]))
            message += '\nSure to proceed?'

        return message


class InfoWindow(QMainWindow):
    """
    Generic widget for displaying collected information in separate windows.
    """

    def __init__(self, info_widget: Any, title: str) -> None:
        """
        Initialize `InfoWindow`.

        :param info_widget: widget containing information which will be
                            displayed, can be a table, collected labels, *etc.*
        :param title: title of the displayed window
        """

        super(InfoWindow, self).__init__()

        self.scroll_area = QScrollArea()
        self.central_widget = QWidget()
        self.info_window_layout = QtWidgets.QGridLayout()
        self.central_widget.setLayout(self.info_window_layout)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok)
        self.buttonBox.clicked.connect(self.close)

        self.info_window_layout.addWidget(info_widget)
        self.info_window_layout.addWidget(self.buttonBox)
        self.scroll_area.setWidget(self.central_widget)
        self.setCentralWidget(self.scroll_area)
        self.setWindowTitle(title)
