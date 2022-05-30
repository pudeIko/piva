
from PyQt5.QtWidgets import QTabWidget, QWidget, QLabel, QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, \
    QLineEdit, QMainWindow, QMessageBox
from PyQt5.QtGui import QFont
from PyQt5 import QtCore
from pyqtgraph.Qt import QtGui
from data_browser import *
from _2Dviewer import *
from _3Dviewer import *
from pyqtgraph import InfiniteLine, PlotWidget, AxisItem, mkPen, PColorMeshItem, mkBrush, FillBetweenItem, \
    PlotDataItem, ScatterPlotItem
from numpy import arange, ndarray
from pyqtgraph.graphicsItems.ImageItem import ImageItem
from imageplot import TracedVariable

import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import arpys_wp as wp
from cmaps import cmaps, my_cmaps
import warnings
warnings.filterwarnings("error")
MDC_PANEL_BGR = (236, 236, 236)


class PlotTool(QMainWindow):

    def __init__(self, data_browser, title=None):
        super(PlotTool, self).__init__()

        self.central_widget = QWidget()
        self.plotting_tool_layout = QtGui.QGridLayout()
        self.central_widget.setLayout(self.plotting_tool_layout)
        self.tabs = QTabWidget()

        self.settings_panel = QWidget()
        self.plot_panel = PlotWidget(background=MDC_PANEL_BGR)

        self.data_browser = data_browser
        self.title = title
        self.data_items = {}

        self.align()
        self.set_settings_panel()

        self.initUI()
        self.setCentralWidget(self.central_widget)
        self.setWindowTitle(self.title)
        self.show()

    def initUI(self):

        self.ds_update_lists.clicked.connect(self.set_ds_dv_list)
        self.ds_dv.currentIndexChanged.connect(self.set_ds_dv_plot_list)
        self.ds_add_button.clicked.connect(self.add_dataset)
        self.ds_remove_button.clicked.connect(self.remove_dataset)

    def align(self):
        ptl = self.plotting_tool_layout

        ptl.addWidget(self.tabs, 0, 0, 3, 8)
        ptl.addWidget(self.plot_panel, 4, 0, 8, 8)

    def set_settings_panel(self):
        self.set_datasets_tab()

    def set_datasets_tab(self):
        # create elements
        self.datasets_tab = QWidget()
        dtl = QtGui.QGridLayout()

        self.ds_dv_label = QLabel('dataset:')
        self.ds_dv = QComboBox()
        self.ds_dv_plot_label = QLabel('curve:')
        self.ds_dv_plot = QComboBox()

        self.ds_added_lbl = QLabel('Added curved:')
        self.ds_added = QComboBox()

        self.ds_update_lists = QPushButton('update')
        self.ds_add_button = QPushButton('add')
        self.ds_remove_button = QPushButton('remove')

        row = 0
        dtl.addWidget(self.ds_dv_label,             row, 0)
        dtl.addWidget(self.ds_dv,                   row, 1)
        dtl.addWidget(self.ds_dv_plot_label,        row, 2)
        dtl.addWidget(self.ds_dv_plot,              row, 3)
        dtl.addWidget(self.ds_update_lists,         row, 4)
        dtl.addWidget(self.ds_add_button,           row, 5)

        row = 1
        dtl.addWidget(self.ds_added_lbl,            row, 0)
        dtl.addWidget(self.ds_added,                row, 1)
        dtl.addWidget(self.ds_remove_button,        row, 5)

        dummy_lbl = QLabel('')
        dtl.addWidget(dummy_lbl, 2, 0, 1, 6)

        self.datasets_tab.layout = dtl
        self.datasets_tab.setLayout(dtl)
        self.tabs.addTab(self.datasets_tab, 'Add/Remove data')

    def set_ds_dv_list(self):
        # clear old lists
        self.dv_list = []
        try:
            self.ds_dv.clear()
            self.ds_dv_plot.clear()
        except AttributeError:
            pass

        dv = self.data_browser.data_viewers
        for dvi in dv.keys():
            self.dv_list.append(dvi)
            dvi = dvi.split('/')[-1]
            self.ds_dv.addItem(dvi)

    def set_ds_dv_plot_list(self):

        self.ds_dv_plot.clear()
        try:
            idx = self.ds_dv.currentIndex()
            dv_lbl = self.dv_list[idx]
        except IndexError:
            return

        dv = self.data_browser.data_viewers[dv_lbl]
        if isinstance(dv, MainWindow2D):
            self.ds_dv_plot.addItem('edc')
            self.ds_dv_plot.addItem('mdc')
            for key in dv.data_viewers.keys():
                if 'edc_viewer' in key:
                    self.ds_dv_plot.addItem('edc_fitter')
                if 'mdc_viewer' in key:
                    self.ds_dv_plot.addItem('mdc_fitter')
        elif isinstance(dv, MainWindow3D):
            self.ds_dv_plot.addItem('main edc')
            self.ds_dv_plot.addItem('single point edc')
            self.ds_dv_plot.addItem('vertical (analyzer)')
            self.ds_dv_plot.addItem('horizontal (scanned)')

    def add_dataset(self):

        idx = self.ds_dv.currentIndex()
        dv_lbl = self.dv_list[idx]
        dv = self.data_browser.data_viewers[dv_lbl]
        plot = self.ds_dv_plot.currentText()

        if isinstance(dv, MainWindow2D):
            x, y = self.get_data_from_2dviewer(dv_lbl, plot)
        elif isinstance(dv, MainWindow3D):
            x, y = self.get_data_from_3dviewer(dv_lbl, plot)

        data_item_lbl = dv_lbl.split('/')[-1] + ' - ' + plot

        if data_item_lbl in self.data_items.keys():
            data_item_colision_box = QMessageBox()
            data_item_colision_box.setIcon(QMessageBox.Question)
            data_item_colision_box.setWindowTitle('Ooh.')
            data_item_colision_box.setText('Curve from this data set already has been added.  Want to add another one?')
            data_item_colision_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            if data_item_colision_box.exec() == QMessageBox.Yes:
                lbl, lbl_return_val = QInputDialog.getText(self, '', 'Then add specific label:', QLineEdit.Normal, '1')
                data_item_lbl = data_item_lbl + ' (' + lbl + ')'
                if not lbl_return_val:
                    return
            else:
                return

        self.data_items[data_item_lbl] = CustonDataItem(x, y, pen=mkPen('k', width=2))
        self.plot_panel.addItem(self.data_items[data_item_lbl])
        self.ds_added.addItem(data_item_lbl)

    def remove_dataset(self):

        item = self.ds_added.currentText()
        idx = self.ds_added.currentIndex()
        self.plot_panel.removeItem(self.data_items[item])
        self.ds_added.removeItem(idx)
        del(self.data_items[item])

    def get_data_from_2dviewer(self, dv_lbl, plot):
        dv = self.data_browser.data_viewers[dv_lbl]

        if plot == 'edc':
            if dv.new_energy_axis is None:
                return dv.data_handler.axes[0], dv.edc
            else:
                return dv.new_energy_axis, dv.edc
        elif plot == 'mdc':
            if dv.new_momentum_axis is None:
                return dv.data_handler.axes[1], dv.mdc
            else:
                return dv.new_momentum_axis, dv.mdc
        elif plot == 'edc - fitter':
            edc_fitter = dv.data_viewers[dv_lbl + '_edc_viewer']
            return edc_fitter.edc_erg_ax, edc_fitter.edc
        elif plot == 'mdc - fitter':
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

    def close(self):
        self.destroy()
        self.data_browser.thread[self.title].quit()
        self.data_browser.thread[self.title].wait()
        del(self.data_browser.thread[self.title])
        del(self.data_viewer.data_viewers[self.title])


class CustonDataItem(PlotDataItem):

    def __init__(self, *args, **kwargs):
        super(CustonDataItem, self).__init__(*args, **kwargs)





