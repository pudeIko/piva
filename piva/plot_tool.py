import argparse
import datetime
from copy import deepcopy

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import QColorDialog, QFileDialog, QWidget, \
    QDoubleSpinBox, QLineEdit, QPushButton, QLabel, QComboBox, QSpinBox, \
    QCheckBox, QMessageBox

import piva.arpys_wp as wp
import piva.data_loader as dl
import piva._2Dviewer as p2d
import piva._3Dviewer as p3d
from piva.imageplot import bold_font

MDC_PANEL_BGR = (236, 236, 236)


class PlotTool(QtWidgets.QMainWindow):

    def __init__(self, data_browser, title=None):
        super(PlotTool, self).__init__()

        self.central_widget = QWidget()
        self.plotting_tool_layout = QtWidgets.QGridLayout()
        self.central_widget.setLayout(self.plotting_tool_layout)
        self.tabs = QtWidgets.QTabWidget()

        self.main_utils = QWidget()
        self.plot_panel = pg.PlotWidget(background=MDC_PANEL_BGR)

        self.data_browser = data_browser
        self.title = title
        self.data_items = {}

        self.plot_panel_design = {}
        self.init_panel_design()

        self.marker_1 = {}
        self.marker_2 = {}
        self.init_markers()

        self.annotations = {}

        self.align()
        self.set_main_utils()
        self.set_tabs()
        self.set_ds_dv_list()

        self.initUI()
        self.setCentralWidget(self.central_widget)
        self.setWindowTitle(self.title)
        self.show()

    def initUI(self):

        self.main_added.currentIndexChanged.connect(self.current_selection_changed)
        self.main_save_button.clicked.connect(self.save)
        self.main_load_button.clicked.connect(self.load)
        self.main_close_button.clicked.connect(self.close)

        self.ds_update_lists.clicked.connect(self.set_ds_dv_list)
        self.ds_dv.currentIndexChanged.connect(self.set_ds_dv_plot_list)
        self.ds_add_button.clicked.connect(self.add_dataset)
        self.ds_remove_button.clicked.connect(self.remove_dataset)

        self.ec_color.clicked.connect(self.set_color)
        self.ec_width.valueChanged.connect(lambda: self.set_pen(loading=False))
        self.ec_style.currentIndexChanged.connect(lambda: self.set_pen(loading=False))
        self.ec_offset_x.valueChanged.connect(lambda: self.update_dataset(loading=False))
        self.ec_offset_y.valueChanged.connect(lambda: self.update_dataset(loading=False))
        self.ec_normalize.stateChanged.connect(lambda: self.update_dataset(loading=False))
        self.ec_scale.valueChanged.connect(lambda: self.update_dataset(loading=False))
        self.ec_reset.clicked.connect(self.reset_scaling)

        self.ep_bgr_color.clicked.connect(lambda: self.set_bgr_color(loading=False))
        self.ep_axes_color.clicked.connect(lambda: self.set_axes_color(loading=False))
        self.ep_ticks_size.valueChanged.connect(lambda: self.set_plot_layout(loading=False))
        self.ep_xlabel.textChanged.connect(lambda: self.set_plot_layout(loading=False))
        self.ep_ylabel.textChanged.connect(lambda: self.set_plot_layout(loading=False))
        self.ep_labels_font_size.valueChanged.connect(lambda: self.set_plot_layout(loading=False))

        self.marker_1['button'].clicked.connect(lambda: self.drop_marker(marker=self.marker_1))
        self.marker_2['button'].clicked.connect(lambda: self.drop_marker(marker=self.marker_2))
        self.marker_1['x'].valueChanged.connect(lambda: self.update_marker(marker=self.marker_1))
        self.marker_2['x'].valueChanged.connect(lambda: self.update_marker(marker=self.marker_2))

        self.ann_add_update.clicked.connect(self.add_update_annotation)
        self.ann_remove.clicked.connect(self.remove_annotation)
        self.ann_color.clicked.connect(self.set_annotation_color)
        self.ann_added.currentIndexChanged.connect(self.ann_selection_changed)

    def init_panel_design(self):
        self.plot_panel_design['bgr_color'] = QColor()
        self.plot_panel_design['bgr_color'].setRgb(MDC_PANEL_BGR[0], MDC_PANEL_BGR[1], MDC_PANEL_BGR[2])
        self.plot_panel_design['axes_color'] = QColor()
        self.plot_panel_design['axes_color'].setRgb(150, 150, 150)
        self.plot_panel_design['ticks_fsize'] = QFont()
        self.plot_panel_design['ticks_fsize'].setPointSize(14)
        self.plot_panel_design['labels_fsize'] = 14

    def init_markers(self):
        max_w = 80

        self.marker_1['marker'] = pg.ScatterPlotItem()
        self.marker_1['dropped'] = False
        self.marker_1['idx'] = 1
        self.marker_1['x'] = QDoubleSpinBox()
        self.set_qspinbox(self.marker_1['x'], [-1e6, 1e6], value=0, decimals=6, step=0.01)
        self.marker_1['y'] = QLineEdit()
        self.marker_1['y'].setReadOnly(True)
        self.marker_1['y'].setText('-')
        self.marker_1['y'].setMaximumWidth(max_w)
        self.marker_1['button'] = QPushButton('drop')
        self.marker_1['dumped_at'] = None

        self.marker_2['marker'] = pg.ScatterPlotItem()
        self.marker_2['dropped'] = False
        self.marker_2['idx'] = 2
        self.marker_2['x'] = QDoubleSpinBox()
        self.set_qspinbox(self.marker_2['x'], [-1e6, 1e6], value=0, decimals=6, step=0.01)
        self.marker_2['y'] = QLineEdit()
        self.marker_2['y'].setReadOnly(True)
        self.marker_2['y'].setText('-')
        self.marker_2['y'].setMaximumWidth(max_w)
        self.marker_2['button'] = QPushButton('drop')
        self.marker_2['dumped_at'] = None

    def align(self):
        ptl = self.plotting_tool_layout

        ptl.addWidget(self.main_utils,  0, 0, 4, 1)
        ptl.addWidget(self.tabs,        0, 1, 4, 7)
        ptl.addWidget(self.plot_panel,  4, 0, 8, 8)

    def set_tabs(self):
        self.set_datasets_tab()
        self.set_edit_curves_tab()
        self.set_edit_plot_tab()
        self.set_markers_tab()
        self.set_annotate_tab()

    def set_main_utils(self):
        # self.main_utils = QWidget()
        mul = QtWidgets.QVBoxLayout()

        self.main_added_lbl = QLabel('Curves:')
        self.main_added = QComboBox()

        self.main_save_button = QPushButton('save')
        self.main_load_button = QPushButton('load')
        self.main_close_button = QPushButton('close')

        mul.addWidget(self.main_added_lbl)
        mul.addWidget(self.main_added)
        mul.addWidget(self.main_save_button)
        mul.addWidget(self.main_load_button)
        mul.addWidget(self.main_close_button)

        self.main_utils.layout = mul
        self.main_utils.setLayout(mul)

    def set_datasets_tab(self):
        # create elements
        self.datasets_tab = QWidget()
        dtl = QtWidgets.QGridLayout()

        self.ds_dv_label = QLabel('dataset:')
        self.ds_dv = QComboBox()
        self.ds_dv.addItem('custom')
        self.ds_dv_plot_label = QLabel('curve:')
        self.ds_dv_plot = QComboBox()

        self.ds_update_lists = QPushButton('update')
        self.ds_add_button = QPushButton('add')
        self.ds_remove_button = QPushButton('remove current')

        self.ds_custom_x_lbl = QLabel('x:')
        self.ds_custom_x = QLineEdit()
        self.ds_custom_y_lbl = QLabel('y:')
        self.ds_custom_y = QLineEdit()
        self.ds_custom_name_lbl = QLabel('name:')
        self.ds_custom_name = QLineEdit()

        row = 0
        dtl.addWidget(self.ds_dv_label,             row, 0)
        dtl.addWidget(self.ds_dv,                   row, 1)
        dtl.addWidget(self.ds_dv_plot_label,        row, 2)
        dtl.addWidget(self.ds_dv_plot,              row, 3)
        dtl.addWidget(self.ds_add_button,           row, 4)
        dtl.addWidget(self.ds_update_lists,         row, 5)

        row = 1
        dtl.addWidget(self.ds_custom_name_lbl,      row, 0)
        dtl.addWidget(self.ds_custom_name,          row, 1)
        dtl.addWidget(self.ds_remove_button,        row, 4, 1, 2)

        row = 2
        dtl.addWidget(self.ds_custom_x_lbl,         row, 0)
        dtl.addWidget(self.ds_custom_x,             row, 1)
        dtl.addWidget(self.ds_custom_y_lbl,         row, 2)
        dtl.addWidget(self.ds_custom_y,             row, 3)

        # dummy_lbl = QLabel('')
        # dtl.addWidget(dummy_lbl, 2, 0, 1, 6)

        self.datasets_tab.layout = dtl
        self.datasets_tab.setLayout(dtl)
        self.tabs.addTab(self.datasets_tab, 'Add/Remove data')

    def set_edit_curves_tab(self):
        # create elements
        self.edit_curves_tab = QWidget()
        etl = QtWidgets.QGridLayout()

        self.ec_color_lbl = QLabel("color:")
        self.ec_color = QPushButton('')
        self.ec_color.setStyleSheet("background-color: black")
        self.ec_color.setFixedSize(25, 15)
        self.ec_width_lbl = QLabel('width')
        self.ec_width = QSpinBox()
        self.set_qspinbox(self.ec_width, box_range=[1, 50], value=3)
        self.ec_style_lbl = QLabel('line style:')
        self.ec_style = QComboBox()
        line_types = ['-', '---', '...', '-.-', '-..']
        for lt in line_types:
            self.ec_style.addItem(lt)

        self.ec_offset_x_lbl = QLabel('x offset:')
        self.ec_offset_x = QDoubleSpinBox()
        self.set_qspinbox(self.ec_offset_x, box_range=[-1e6, 1e6], value=0, decimals=5, step=1e-3)
        self.ec_offset_y_lbl = QLabel('y offset:')
        self.ec_offset_y = QDoubleSpinBox()
        self.set_qspinbox(self.ec_offset_y, box_range=[-1e6, 1e6], value=0, decimals=4, step=0.1)

        self.ec_normalize = QCheckBox('normalize')
        self.ec_scale_lbl = QLabel('scale:')
        self.ec_scale = QDoubleSpinBox()
        self.set_qspinbox(self.ec_scale, box_range=[0, 1e6], value=1,  decimals=1, step=1)
        self.ec_reset = QPushButton('reset')

        row = 0
        etl.addWidget(self.ec_color_lbl,              row, 0)
        etl.addWidget(self.ec_color,                  row, 1)
        etl.addWidget(self.ec_width_lbl,              row, 2)
        etl.addWidget(self.ec_width,                  row, 3)
        etl.addWidget(self.ec_style_lbl,              row, 4)
        etl.addWidget(self.ec_style,                  row, 5)

        row = 1
        etl.addWidget(self.ec_offset_x_lbl,           row, 0)
        etl.addWidget(self.ec_offset_x,               row, 1, 1, 2)
        etl.addWidget(self.ec_offset_y_lbl,           row, 3)
        etl.addWidget(self.ec_offset_y,               row, 4, 1, 2)

        row = 2
        etl.addWidget(self.ec_scale_lbl,              row, 0)
        etl.addWidget(self.ec_scale,                  row, 1)
        etl.addWidget(self.ec_normalize,              row, 2)
        etl.addWidget(self.ec_reset,                  row, 3)

        # dummy_lbl = QLabel('')
        # etl.addWidget(dummy_lbl, 2, 0, 1, 6)

        self.edit_curves_tab.layout = etl
        self.edit_curves_tab.setLayout(etl)
        self.tabs.addTab(self.edit_curves_tab, 'Edit curves')

    def set_edit_plot_tab(self):
        # create elements
        self.edit_plot_tab = QWidget()
        eptl = QtWidgets.QGridLayout()

        self.ep_bgr_color_lbl = QLabel("bgr color:")
        self.ep_bgr_color = QPushButton('')
        self.ep_bgr_color.setStyleSheet(f'background-color: rgb{MDC_PANEL_BGR}')
        self.ep_bgr_color.setFixedSize(25, 15)
        self.ep_axes_color_lbl = QLabel("axes color:")
        self.ep_axes_color = QPushButton('')
        self.ep_axes_color.setStyleSheet(f'background-color: rgb(150, 150, 150)')
        self.ep_axes_color.setFixedSize(25, 15)
        self.ep_ticks_size_lbl = QLabel('ticks font size:')
        self.ep_ticks_size = QSpinBox()
        self.set_qspinbox(self.ep_ticks_size, box_range=[1, 100], value=14)

        self.ep_labels_font_size_lbl = QLabel('labels font size:')
        self.ep_labels_font_size = QSpinBox()
        self.set_qspinbox(self.ep_labels_font_size, box_range=[1, 100], value=14)

        self.ep_xlabel_lbl = QLabel('x label:')
        self.ep_xlabel = QLineEdit()
        self.ep_ylabel_lbl = QLabel('y label:')
        self.ep_ylabel = QLineEdit()

        row = 0
        eptl.addWidget(self.ep_bgr_color_lbl,               row, 0)
        eptl.addWidget(self.ep_bgr_color,                   row, 1)
        eptl.addWidget(self.ep_axes_color_lbl,              row, 2)
        eptl.addWidget(self.ep_axes_color,                  row, 3)
        eptl.addWidget(self.ep_ticks_size_lbl,              row, 4)
        eptl.addWidget(self.ep_ticks_size,                  row, 5)

        row = 1
        eptl.addWidget(self.ep_xlabel_lbl,                  row, 0)
        eptl.addWidget(self.ep_xlabel,                      row, 1, 1, 2)
        eptl.addWidget(self.ep_ylabel_lbl,                  row, 3)
        eptl.addWidget(self.ep_ylabel,                      row, 4, 1, 2)

        row = 2
        eptl.addWidget(self.ep_labels_font_size_lbl,        row, 4)
        eptl.addWidget(self.ep_labels_font_size,            row, 5)

        self.edit_plot_tab.layout = eptl
        self.edit_plot_tab.setLayout(eptl)
        self.tabs.addTab(self.edit_plot_tab, 'Edit plot')

    def set_markers_tab(self):
        # create elements
        self.markers_tab = QWidget()
        mtl = QtWidgets.QGridLayout()
        max_w = 80

        self.markers_x_lbl = QLabel("x:")
        self.markers_y_lbl = QLabel("y:")
        self.markers_dx_lbl = QLabel("dx:")
        self.markers_dy_lbl = QLabel("dy:")

        self.marker_1_lbl = QLabel('1:')
        self.marker_1_lbl.setFont(bold_font)
        self.markers_dx = QLineEdit()
        self.markers_dx.setReadOnly(True)
        self.markers_dx.setMaximumWidth(max_w)
        self.markers_dy = QLineEdit()
        self.markers_dy.setReadOnly(True)
        self.markers_dy.setMaximumWidth(max_w)

        self.marker_2_lbl = QLabel('2:')
        self.marker_2_lbl.setFont(bold_font)

        row = 0
        mtl.addWidget(self.markers_x_lbl,       row, 1)
        mtl.addWidget(self.markers_y_lbl,       row, 2)
        mtl.addWidget(self.markers_dx_lbl,      row, 3)
        mtl.addWidget(self.markers_dy_lbl,      row, 4)

        row = 1
        mtl.addWidget(self.marker_1_lbl,        row, 0)
        mtl.addWidget(self.marker_1['x'],       row, 1)
        mtl.addWidget(self.marker_1['y'],       row, 2)
        mtl.addWidget(self.markers_dx,          row, 3)
        mtl.addWidget(self.markers_dy,          row, 4)
        mtl.addWidget(self.marker_1['button'],  row, 5)

        row = 2
        mtl.addWidget(self.marker_2_lbl,        row, 0)
        mtl.addWidget(self.marker_2['x'],       row, 1)
        mtl.addWidget(self.marker_2['y'],       row, 2)
        mtl.addWidget(self.marker_2['button'],  row, 5)

        self.markers_tab.layout = mtl
        self.markers_tab.setLayout(mtl)
        self.tabs.addTab(self.markers_tab, 'Markers')

    def set_annotate_tab(self):
        # create elements
        self.ann_tab = QWidget()
        atl = QtWidgets.QGridLayout()

        self.ann_name_lbl = QLabel('name:')
        self.ann_name = QLineEdit()
        self.ann_text_lbl = QLabel('text:')
        self.ann_text = QLineEdit()

        self.ann_added_lbl = QLabel('added:')
        self.ann_added = QComboBox()
        self.ann_fsize_lbl = QLabel('font size:')
        self.ann_fsize = QSpinBox()
        self.set_qspinbox(self.ann_fsize, box_range=[2, 50], value=14)
        self.ann_color_lbl = QLabel('color:')
        self.ann_color = QPushButton('')
        self.ann_color.setStyleSheet("background-color: black")
        self.ann_color.setFixedSize(25, 15)

        self.ann_x_lbl = QLabel('x:')
        self.ann_x = QDoubleSpinBox()
        self.set_qspinbox(self.ann_x, box_range=[-1e6, 1e6], value=-0.5, step=0.01)
        self.ann_y_lbl = QLabel('y:')
        self.ann_y = QDoubleSpinBox()
        self.set_qspinbox(self.ann_y, box_range=[-1e6, 1e6], value=-0.5, step=0.01)
        self.ann_add_update = QPushButton('add/update')
        self.ann_remove = QPushButton('delete')

        row = 0
        atl.addWidget(self.ann_name_lbl,        row, 0)
        atl.addWidget(self.ann_name,            row, 1, 1, 2)
        atl.addWidget(self.ann_text_lbl,        row, 3)
        atl.addWidget(self.ann_text,            row, 4, 1, 2)

        row = 1
        atl.addWidget(self.ann_added_lbl,       row, 0)
        atl.addWidget(self.ann_added,           row, 1)
        atl.addWidget(self.ann_fsize_lbl,       row, 2)
        atl.addWidget(self.ann_fsize,           row, 3)
        atl.addWidget(self.ann_color_lbl,       row, 4)
        atl.addWidget(self.ann_color,           row, 5)

        row = 2
        atl.addWidget(self.ann_x_lbl,           row, 0)
        atl.addWidget(self.ann_x,               row, 1)
        atl.addWidget(self.ann_y_lbl,           row, 2)
        atl.addWidget(self.ann_y,               row, 3)
        atl.addWidget(self.ann_add_update,      row, 4)
        atl.addWidget(self.ann_remove,          row, 5)

        self.ann_tab.layout = atl
        self.ann_tab.setLayout(atl)
        self.tabs.addTab(self.ann_tab, 'Annotate')

    def set_ds_dv_list(self):
        # clear old lists
        self.dv_list = []
        try:
            self.ds_dv.clear()
            self.ds_dv_plot.clear()
        except AttributeError:
            pass

        self.ds_dv.addItem('custom')
        self.dv_list.append('custom')
        dv = self.data_browser.data_viewers
        for dvi in dv.keys():
            self.dv_list.append(dvi)
            dvi_lbl = dvi.split('/')[-1]
            self.ds_dv.addItem(dvi_lbl)
            if isinstance(self.data_browser.data_viewers[dvi], p3d.MainWindow3D):
                for key in self.data_browser.data_viewers[dvi].data_viewers.keys():
                    key_lbl = key.split('/')[-1]
                    self.dv_list.append(key)
                    self.ds_dv.addItem(key_lbl)

    def set_ds_dv_plot_list(self):

        self.ds_dv_plot.clear()
        try:
            idx = self.ds_dv.currentIndex()
            dv_lbl = self.dv_list[idx]
        except IndexError:
            return

        dv = self.data_browser.data_viewers[dv_lbl]

        if isinstance(dv, p2d.MainWindow2D):
            self.ds_dv_plot.addItem('edc')
            self.ds_dv_plot.addItem('mdc')
            for key in dv.data_viewers.keys():
                if 'edc_viewer' in key:
                    self.ds_dv_plot.addItem('edc_fitter')
                if 'mdc_viewer' in key:
                    self.ds_dv_plot.addItem('mdc_fitter')
        elif isinstance(dv, p3d.MainWindow3D):
            self.ds_dv_plot.addItem('main edc')
            self.ds_dv_plot.addItem('single point edc')
            self.ds_dv_plot.addItem('vertical (analyzer)')
            self.ds_dv_plot.addItem('horizontal (scanned)')

    def add_dataset(self):

        idx = self.ds_dv.currentIndex()

        try:
            dv_lbl = self.dv_list[idx]
        except AttributeError:
            no_entries_box = QMessageBox()
            no_entries_box.setIcon(QMessageBox.Information)
            no_entries_box.setWindowTitle('Doh.')
            no_entries_box.setText('No entries to add.  Make sure to update.')
            no_entries_box.setStandardButtons(QMessageBox.Ok)
            if no_entries_box.exec() == QMessageBox.Ok:
                return
        except IndexError:
            no_entries_box = QMessageBox()
            no_entries_box.setIcon(QMessageBox.Information)
            no_entries_box.setWindowTitle('Doh.')
            no_entries_box.setText('Something went wrong.  Open new data set and update.')
            no_entries_box.setStandardButtons(QMessageBox.Ok)
            if no_entries_box.exec() == QMessageBox.Ok:
                return

        if dv_lbl == 'custom':
            plot = self.ds_custom_name.text()
            if plot == '':
                no_name_box = QMessageBox()
                no_name_box.setIcon(QMessageBox.Information)
                no_name_box.setWindowTitle('Doh.')
                no_name_box.setText('Need to specify data set\'s name.')
                no_name_box.setStandardButtons(QMessageBox.Ok)
                if no_name_box.exec() == QMessageBox.Ok:
                    return
        else:
            try:
                dv = self.data_browser.data_viewers[dv_lbl]
                plot = self.ds_dv_plot.currentText()
            except KeyError:
                no_dv_box = QMessageBox()
                no_dv_box.setIcon(QMessageBox.Information)
                no_dv_box.setWindowTitle('Doh.')
                no_dv_box.setText('It seems the data set has been closed.  Make sure to update.')
                no_dv_box.setStandardButtons(QMessageBox.Ok)
                if no_dv_box.exec() == QMessageBox.Ok:
                    return

        if dv_lbl == 'custom':
            x = self.ds_custom_x.text().split(' ')
            y = self.ds_custom_y.text().split(' ')
            x = np.array([float(xi) for xi in x])
            y = np.array([float(yi) for yi in y])
            data_item_lbl = plot
        else:
            if isinstance(dv, p2d.MainWindow2D):
                x, y = self.get_data_from_2dviewer(dv_lbl, plot)
            elif isinstance(dv, p3d.MainWindow3D):
                x, y = self.get_data_from_3dviewer(dv_lbl, plot)
            data_item_lbl = dv_lbl.split('/')[-1] + ' - ' + plot

        if data_item_lbl in self.data_items.keys():
            data_item_colision_box = QMessageBox()
            data_item_colision_box.setIcon(QMessageBox.Question)
            data_item_colision_box.setWindowTitle('Ooh.')
            data_item_colision_box.setText('Curve with this name already has been added.  Want to add another one?')
            data_item_colision_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            if data_item_colision_box.exec() == QMessageBox.Yes:
                lbl, lbl_return_val = QtWidgets.QInputDialog.getText(self, '', 'Then add specific label:', QLineEdit.Normal, '1')
                data_item_lbl = data_item_lbl + ' (' + lbl + ')'
                if not lbl_return_val:
                    return
            else:
                return

        self.data_items[data_item_lbl] = {}
        self.data_items[data_item_lbl]['data_item'] = CustomDataItem(x, y, pen=pg.mkPen('k', width=2))
        self.data_items[data_item_lbl]['org_data'] = [deepcopy(x), deepcopy(y)]
        self.data_items[data_item_lbl]['color'] = 'k'
        self.data_items[data_item_lbl]['width'] = 2
        self.data_items[data_item_lbl]['line_style'] = '-'
        self.data_items[data_item_lbl]['x_off'] = 0
        self.data_items[data_item_lbl]['y_off'] = 0
        self.data_items[data_item_lbl]['scale'] = 1
        self.data_items[data_item_lbl]['normalize'] = False

        self.plot_panel.addItem(self.data_items[data_item_lbl]['data_item'])
        self.main_added.addItem(data_item_lbl)

    def remove_dataset(self):

        item = self.main_added.currentText()
        idx = self.main_added.currentIndex()
        self.plot_panel.removeItem(self.data_items[item]['data_item'])
        self.main_added.removeItem(idx)
        del(self.data_items[item])

    def current_selection_changed(self):
        try:
            di = self.data_items[self.main_added.currentText()]
        except KeyError:
            return

        self.change_without_emitting_signal(self.ec_width, di['width'])
        self.change_without_emitting_signal(self.ec_style, di['line_style'])
        self.change_without_emitting_signal(self.ec_offset_x, di['x_off'])
        self.change_without_emitting_signal(self.ec_offset_y, di['y_off'])
        self.change_without_emitting_signal(self.ec_scale, di['scale'])
        self.change_without_emitting_signal(self.ec_normalize, di['normalize'])

        clr = di['color']
        try:
            clr = di['color'].getRgb()[:3]
        except AttributeError:
            pass
        self.ec_color.setStyleSheet(f'background-color: rgb{clr}')

    def get_data_from_2dviewer(self, dv_lbl, plot):
        dv = self.data_browser.data_viewers[dv_lbl]

        if plot == 'edc':
            if dv.new_energy_axis is None:
                return dv.data_handler.axes[0], dv.edc
            else:
                return dv.new_energy_axis, dv.edc
        elif plot == 'mdc':
            if dv.k_axis is None:
                return dv.data_handler.axes[1], dv.mdc
            else:
                return dv.k_axis, dv.mdc
        elif plot == 'edc_fitter':
            edc_fitter = dv.data_viewers[dv_lbl + '_edc_viewer']
            return edc_fitter.edc_erg_ax, edc_fitter.edc
        elif plot == 'mdc_fitter':
            mdc_fitter = dv.data_viewers[dv_lbl + '_mdc_viewer']
            return mdc_fitter.k_ax, mdc_fitter.mdc

    def get_data_from_3dviewer(self, dv_lbl, plot):
        dv = self.data_browser.data_viewers[dv_lbl]

        if plot == 'main edc':
            edc = dv.plot_z.get_data()[1]
            if dv.new_energy_axis is None:
                return dv.data_set.zscale, edc
            else:
                return dv.new_energy_axis, edc
        elif plot == 'single point edc':
            edc = dv.sp_EDC.getData()[1]
            if dv.new_energy_axis is None:
                return dv.data_set.zscale, edc
            else:
                return dv.new_energy_axis, edc
        elif plot == 'vertical (analyzer)':
            return dv.data_set.yscale, dv.plot_y_data
        elif plot == 'horizontal (scanned)':
            return dv.data_set.xscale, dv.plot_x_data

    def set_color(self):
        cd = QColorDialog.getColor()
        cd_str = 'rgb' + str(cd.getRgb()[:3])
        self.ec_color.setStyleSheet(f'background-color: {cd_str}')
        data_item_lbl = self.main_added.currentText()
        try:
            self.data_items[data_item_lbl]['color'] = cd
        except KeyError:
            return

        self.set_pen()

    def set_pen(self, loading=False):
        data_item_lbl = self.main_added.currentText()
        if loading:
            w = self.data_items[data_item_lbl]['width']
            self.ec_style.setCurrentText(self.data_items[data_item_lbl]['line_style'])
            style, style_lbl = self.get_line_style()
            cd = self.data_items[data_item_lbl]['color']
        else:
            w = self.ec_width.value()
            style, style_lbl = self.get_line_style()
            cd = self.data_items[data_item_lbl]['color']

        self.data_items[data_item_lbl]['data_item'].setPen(color=cd, width=w, style=style)
        # self.data_items[data_item_lbl]['color'] = cd
        self.data_items[data_item_lbl]['width'] = w
        self.data_items[data_item_lbl]['line_style'] = style_lbl

    def get_line_style(self):
        s = self.ec_style.currentText()
        if s == '-':
            style = QtCore.Qt.SolidLine
        elif s == '---':
            style = QtCore.Qt.DashLine
        elif s == '...':
            style = QtCore.Qt.DotLine
        elif s == '-.-':
            style = QtCore.Qt.DashDotLine
        elif s == '-..':
            style = QtCore.Qt.DashDotDotLine

        return style, s

    def update_dataset(self, loading=False):
        data_item_lbl = self.main_added.currentText()
        if loading:
            x_off = self.data_items[data_item_lbl]['x_off']
            y_off = self.data_items[data_item_lbl]['y_off']
            scale = self.data_items[data_item_lbl]['scale']
            self.ec_normalize.setChecked(self.data_items[data_item_lbl]['normalize'])
        else:
            x_off = self.ec_offset_x.value()
            y_off = self.ec_offset_y.value()
            scale = self.ec_scale.value()

        org = deepcopy(self.data_items[data_item_lbl]['org_data'])
        x, y = org[0], org[1]
        if self.ec_normalize.isChecked():
            y = wp.normalize(y)
            self.data_items[data_item_lbl]['normalize'] = True
        y *= scale
        x += x_off
        y += y_off
        self.data_items[data_item_lbl]['data_item'].setData(x, y)
        self.data_items[data_item_lbl]['x_off'] = x_off
        self.data_items[data_item_lbl]['y_off'] = y_off
        self.data_items[data_item_lbl]['scale'] = scale

        self.update_marker(marker=self.marker_1)
        self.update_marker(marker=self.marker_2)

    def reset_scaling(self):
        data_item_lbl = self.main_added.currentText()
        org = self.data_items[data_item_lbl]['org_data']
        self.data_items[data_item_lbl]['data_item'].setData(deepcopy(org[0]), deepcopy(org[1]))
        self.change_without_emitting_signal(self.ec_offset_x, 0)
        self.change_without_emitting_signal(self.ec_offset_y, 0)
        self.change_without_emitting_signal(self.ec_scale, 1)
        self.change_without_emitting_signal(self.ec_normalize, False)
        self.data_items[data_item_lbl]['x_off'] = 0
        self.data_items[data_item_lbl]['y_off'] = 0
        self.data_items[data_item_lbl]['scale'] = 1
        self.data_items[data_item_lbl]['normalize'] = False

    def set_bgr_color(self, loading=False):
        if loading:
            cd_str = 'rgb' + str(self.plot_panel_design['bgr_color'].getRgb()[:3])
        else:
            cd = QColorDialog.getColor()
            cd_str = 'rgb' + str(cd.getRgb()[:3])
            self.plot_panel_design['bgr_color'] = cd
        self.ep_bgr_color.setStyleSheet(f'background-color: {cd_str}')

        self.set_plot_layout(loading=loading)

    def set_axes_color(self, loading=False):
        if loading:
            cd_str = 'rgb' + str(self.plot_panel_design['axes_color'].getRgb()[:3])
        else:
            cd = QColorDialog.getColor()
            cd_str = 'rgb' + str(cd.getRgb()[:3])
            self.plot_panel_design['axes_color'] = cd
        self.ep_axes_color.setStyleSheet(f'background-color: {cd_str}')

        self.set_plot_layout(loading=loading)

    def set_ticks_fsize(self, loading=False):
        self.set_plot_layout(loading=loading)

    def set_plot_layout(self, loading=False):
        bgr_clr = self.plot_panel_design['bgr_color']
        ax_clr = self.plot_panel_design['axes_color']

        if loading:
            self.ep_ticks_size.blockSignals(True)
            self.ep_labels_font_size.blockSignals(True)
            self.ep_ticks_size.setValue(self.plot_panel_design['ticks_fsize'].pointSize())
            self.ep_labels_font_size.setValue(self.plot_panel_design['labels_fsize'])
            self.ep_ticks_size.blockSignals(False)
            self.ep_labels_font_size.blockSignals(False)
        tfs = self.ep_ticks_size.value()
        self.plot_panel_design['ticks_fsize'].setPointSize(tfs)
        ticks_font = self.plot_panel_design['ticks_fsize']

        lbl_fsize = str(self.ep_labels_font_size.value()) + 'pt'
        lbl_clr = 'rgb' + str(ax_clr.getRgb()[:3])
        label_style = {'color': lbl_clr, 'font-size': lbl_fsize}
        self.plot_panel_design['labels_fsize'] = self.ep_labels_font_size.value()

        self.plot_panel.setBackground(bgr_clr)

        bottom_ax = self.plot_panel.plotItem.getAxis('bottom')
        bottom_ax.setPen(color=ax_clr)
        bottom_ax.setTextPen(color=ax_clr)
        bottom_ax.setTickFont(ticks_font)
        if not self.ep_xlabel.text() == '':
            xlbl = self.ep_xlabel.text().split(';')
            if len(xlbl) == 1:
                bottom_ax.setLabel(text=xlbl, **label_style)
            elif len(xlbl) == 2:
                bottom_ax.setLabel(text=xlbl[0], units=xlbl[1], **label_style)

        left_ax = self.plot_panel.plotItem.getAxis('left')
        left_ax.setPen(color=ax_clr)
        left_ax.setTextPen(color=ax_clr)
        left_ax.setTickFont(ticks_font)
        if not self.ep_ylabel.text() == '':
            ylbl = self.ep_ylabel.text().split(';')
            if len(ylbl) == 1:
                left_ax.setLabel(text=ylbl, **label_style)
            elif len(ylbl) == 2:
                left_ax.setLabel(text=ylbl[0], units=ylbl[1], **label_style)

    def get_data_limits(self):
        try:
            di_lbl = self.main_added.currentText()
            x, y = self.data_items[di_lbl]['data_item'].getData()
            xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
        except KeyError:
            return

        for di in self.data_items.keys():
            x, y = self.data_items[di]['data_item'].getData()
            if x.min() < xmin:
                xmin = x.min()
            if x.max() > xmax:
                xmax = x.max()
            if y.min() < ymin:
                ymin = y.min()
            if y.max() > ymax:
                ymax = y.max()
        return xmin, xmax, ymin, ymax

    def drop_marker(self, marker=None):
        if len(self.data_items.keys()) == 0:
            no_data_item_box = QMessageBox()
            no_data_item_box.setIcon(QMessageBox.Information)
            no_data_item_box.setWindowTitle('Doh.')
            no_data_item_box.setText('No dataset to dump marker on.')
            no_data_item_box.setStandardButtons(QMessageBox.Ok)
            if no_data_item_box.exec() == QMessageBox.Ok:
                return

        marker = marker
        marker['dropped'] = not marker['dropped']
        if marker['dropped']:
            di = self.main_added.currentText()
            marker['dumped_at'] = di
            x, y = self.data_items[di]['data_item'].getData()
            marker['button'].setText('remove')
            marker['x'].setRange(x.min(), x.max())
            x0 = np.abs(x.min() - x.max()) / 2 + x.min()
            marker['x'].setValue(x0)
            marker['x'].setSingleStep(wp.get_step(x))
            x0_idx = wp.indexof(x0, x)
            y0 = y[x0_idx]
            marker['y'].setText('{:.2f}'.format(y0))
            self.set_markers_differences()
            xx, yy = [x0, x0], [y0, y0]
            if marker['idx'] == 1:
                s = 'star'
                mkp = pg.mkPen(color='r')
                b = pg.mkBrush(color='r')
            else:
                s = 'x'
                mkp = pg.mkPen(color='k')
                b = pg.mkBrush(color='k')
            marker['marker'].setData(xx, yy, symbol=s, pen=mkp, brush=b)
            self.plot_panel.addItem(marker['marker'])
        else:
            marker['button'].setText('drop')
            marker['x'].setValue(0)
            marker['y'].setText('-')
            self.plot_panel.removeItem(marker['marker'])
            self.set_markers_differences()

    def set_markers_differences(self):
        if self.marker_1['dropped'] and self.marker_2['dropped']:
            dx = np.abs(self.marker_1['x'].value() - self.marker_2['x'].value())
            y_m1 = float(self.marker_1['y'].text())
            y_m2 = float(self.marker_2['y'].text())
            dy = np.abs(y_m1 - y_m2)
            self.markers_dx.setText('{:.6f}'.format(dx))
            self.markers_dy.setText('{:.2f}'.format(dy))
        else:
            self.markers_dx.setText('-')
            self.markers_dy.setText('-')

    def update_marker(self, marker=None):
        marker = marker
        if marker['dropped']:
            x, y = self.data_items[marker['dumped_at']]['data_item'].getData()
            x0 = marker['x'].value()
            x_idx = wp.indexof(x0, x)
            y0 = y[x_idx]
            marker['y'].setText('{:.2f}'.format(y0))
            marker['marker'].setData([x0, x0], [y0, y0])
            self.set_markers_differences()
        else:
            return

    def set_annotation_color(self):
        name = self.ann_name.text()
        if not (name in self.annotations.keys()):
            no_annotation_box = QMessageBox()
            no_annotation_box.setIcon(QMessageBox.Information)
            no_annotation_box.setWindowTitle('Doh.')
            no_annotation_box.setText('No annotation to edit.')
            no_annotation_box.setStandardButtons(QMessageBox.Ok)
            if no_annotation_box.exec() == QMessageBox.Ok:
                return

        cd = QColorDialog.getColor()
        cd_str = cd.getRgb()[:3]
        self.ann_color.setStyleSheet(f'background-color: rgb{str(cd_str)}')
        self.annotations[name]['color'] = cd_str
        self.add_update_annotation()

    def add_update_annotation(self):
        name = self.ann_name.text()
        if name == '':
            no_name_box = QMessageBox()
            no_name_box.setIcon(QMessageBox.Information)
            no_name_box.setWindowTitle('Doh.')
            no_name_box.setText('Must specify name.')
            no_name_box.setStandardButtons(QMessageBox.Ok)
            if no_name_box.exec() == QMessageBox.Ok:
                return

        try:
            self.plot_panel.removeItem(self.annotations[name]['text_item'])
        except AttributeError:
            pass
        except KeyError:
            pass

        text = self.ann_text.text()
        x = self.ann_x.value()
        y = self.ann_y.value()
        fsize = self.ann_fsize.value()
        text_font = QFont()
        text_font.setPointSize(fsize)

        try:
            clr = self.annotations[name]['color']
        except KeyError:
            self.annotations[name] = {}
            self.ann_added.blockSignals(True)
            self.ann_added.addItem(name)
            self.ann_added.blockSignals(False)
            clr = 'k'
            self.annotations[name]['color'] = clr

        self.annotations[name]['text_item'] = pg.TextItem(text=text, anchor=(0, 1), color=clr)
        self.annotations[name]['text_item'].setFont(text_font)
        self.annotations[name]['text_item'].setPos(x, y)
        self.plot_panel.addItem(self.annotations[name]['text_item'])
        self.annotations[name]['text'] = text
        self.annotations[name]['fsize'] = fsize
        self.annotations[name]['x'] = x
        self.annotations[name]['y'] = y

    def remove_annotation(self):
        name = self.ann_name.text()
        try:
            self.plot_panel.removeItem(self.annotations[name]['text_item'])
            self.ann_added.removeItem(self.ann_added.currentIndex())
            del(self.annotations[name])
        except AttributeError:
            pass
        except KeyError:
            pass

    def ann_selection_changed(self):
        name = self.ann_added.currentText()
        if name == '':
            self.ann_name.setText('')
            self.ann_text.setText('')
            self.ann_fsize.setValue(14)
            self.ann_x.setValue(1)
            self.ann_y.setValue(1)
            self.ann_color.setStyleSheet('background-color: black')
        else:
            self.ann_name.setText(name)
            self.ann_text.setText(self.annotations[name]['text'])
            self.ann_fsize.setValue(self.annotations[name]['fsize'])
            self.ann_x.setValue(self.annotations[name]['x'])
            self.ann_y.setValue(self.annotations[name]['y'])
            clr = self.annotations[name]['color']
            self.ann_color.setStyleSheet(f'background-color: rgb{str(clr)}')

    @staticmethod
    def change_without_emitting_signal(widget, value):
        widget.blockSignals(True)
        try:
            if isinstance(widget, QCheckBox):
                widget.setChecked(value)
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(value)
            else:
                widget.setValue(value)
        except AttributeError:
            print(f'Some error with setting value {value} in {widget}.')
        finally:
            widget.blockSignals(False)

    @staticmethod
    def set_qspinbox(box, box_range=[-1., 1.], value=1., decimals=3, step=0.1, max_w=80):
        box.setRange(box_range[0], box_range[1])
        box.setValue(value)
        box.setMaximumWidth(max_w)
        if isinstance(box, QDoubleSpinBox):
            box.setDecimals(decimals)
            box.setSingleStep(step)

    def save(self):
        save_selector_box = QMessageBox()
        save_selector_box.setIcon(QMessageBox.Question)
        save_selector_box.setWindowTitle('Save')
        save_selector_box.setText('Save an image or session?')
        save_selector_box.setStandardButtons(QMessageBox.Cancel | QMessageBox.No | QMessageBox.Yes)
        no_button = save_selector_box.button(QMessageBox.No)
        no_button.setText('Session')
        yes_button = save_selector_box.button(QMessageBox.Yes)
        yes_button.setText('Image')
        choice = save_selector_box.exec_()
        if choice == QMessageBox.Cancel:
            return
        elif choice == QMessageBox.No:
            self.save_session()
        elif choice == QMessageBox.Yes:
            self.save_image()

    def save_session(self):
        data_items_to_save = self.get_data_items_to_save()
        plot_design_to_save = self.get_plot_design_to_save()
        annotations_to_save = self.get_annotations_to_save()
        res = argparse.Namespace(
            data_items=data_items_to_save,
            plot_design=plot_design_to_save,
            annotations=annotations_to_save
        )

        full_path, types = QFileDialog.getSaveFileName(self, 'Save Session')

        dl.dump(res, full_path, force=True)

    def get_data_items_to_save(self):
        data_items_to_save = {}
        for key in self.data_items.keys():
            data_items_to_save[key] = {}
            for keyy in self.data_items[key].keys():
                if keyy == 'data_item':
                    data_items_to_save[key][keyy] = self.data_items[key][keyy].getData()
                else:
                    data_items_to_save[key][keyy] = self.data_items[key][keyy]
        return data_items_to_save

    def set_data_items_from_save(self, saved):
        self.data_items = {}
        for key in saved.keys():
            self.main_added.addItem(key)
            self.data_items[key] = {}
            for keyy in saved[key].keys():
                if keyy == 'data_item':
                    x, y = saved[key][keyy]
                    self.data_items[key][keyy] = CustomDataItem(x, y)
                else:
                    self.data_items[key][keyy] = saved[key][keyy]
            self.plot_panel.addItem(self.data_items[key]['data_item'])

    def get_plot_design_to_save(self):
        plot_design_to_save = {}
        for key in self.plot_panel_design.keys():
            if key == 'ticks_fsize':
                plot_design_to_save[key] = self.plot_panel_design[key].pointSize()
            else:
                plot_design_to_save[key] = self.plot_panel_design[key]
        return plot_design_to_save

    def set_plot_design_from_save(self, saved):
        self.plot_panel_design = {}
        for key in saved.keys():
            if key == 'ticks_fsize':
                self.plot_panel_design[key] = QFont()
                self.plot_panel_design[key].setPointSize(saved[key])
            else:
                self.plot_panel_design[key] = saved[key]

    def get_annotations_to_save(self):
        annotations_to_save = {}
        for key in self.annotations.keys():
            annotations_to_save[key] = {}
            for keyy in self.annotations[key].keys():
                if keyy == 'text_item':
                    pass
                else:
                    annotations_to_save[key][keyy] = self.annotations[key][keyy]
        return annotations_to_save

    def set_annotations_from_save(self, saved):
        self.annotations = {}
        for key in saved.keys():
            self.ann_added.addItem(key)
            self.annotations[key] = {}
            for keyy in saved[key].keys():
                if keyy == 'text':
                    self.annotations[key][keyy] = saved[key][keyy]
                else:
                    self.annotations[key][keyy] = saved[key][keyy]
                clr = saved[key]['color']
                self.annotations[key]['text_item'] = pg.TextItem(text=saved[key]['text'], anchor=(1, 0), color=clr)
        return self.annotations

    def load(self):
        warning_box = QMessageBox()
        warning_box.setIcon(QMessageBox.Information)
        warning_box.setWindowTitle('Load')
        warning_box.setText('Current progress will be lost.  Sure to continue?')
        warning_box.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        choice = warning_box.exec_()
        if choice == QMessageBox.Ok:
            pass
        else:
            return

        full_path, types = QFileDialog.getOpenFileName(self, 'Load Session')
        session = dl.load_pickle(full_path)

        self.main_added.blockSignals(True)
        self.ann_added.blockSignals(True)
        del self.data_items
        del self.plot_panel_design
        del self.annotations
        self.main_added.clear()
        self.ann_added.clear()

        self.set_data_items_from_save(session.data_items)
        self.set_plot_design_from_save(session.plot_design)
        self.set_annotations_from_save(session.annotations)

        self.main_added.blockSignals(False)
        self.ann_added.blockSignals(False)

        for idx in range(self.main_added.count()):
            self.main_added.setCurrentIndex(idx)
            self.update_dataset(loading=True)
            self.set_pen(loading=True)

        self.set_bgr_color(loading=True)
        self.set_axes_color(loading=True)

        for idx in range(self.ann_added.count()):
            self.ann_added.setCurrentIndex(idx)
            self.add_update_annotation()

    def save_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        full_path, types = QFileDialog.getSaveFileName(
            self, 'Save Image', filter='Images (*.png *.jpg *.jpeg *.bmp)', options=options)
        to_save = pg.exporters.ImageExporter(self.plot_panel.plotItem)

        to_save.export(full_path)

    def closeEvent(self, event) :
        """ closeEvent is triggered on `Alt-F4` or mouse-click closing a 
        window. 
        """
        self.data_browser.thread[self.title].quit()
        self.data_browser.thread[self.title].wait()
        del(self.data_browser.thread[self.title])
        del(self.data_browser.plotting_tools[self.title])


class CustomDataItem(pg.PlotDataItem):

    def __init__(self, *args, **kwargs):
        super(CustomDataItem, self).__init__(*args, **kwargs)

        self.created = datetime.datetime.now()

