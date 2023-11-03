import pytestqt
import os
from piva.data_browser import DataBrowser
from piva.image_panels import InfoWindow
from piva.data_viewer_2d import DataViewer2D
from pyqtgraph.Qt.QtCore import Qt
from PyQt5.QtTest import QTest
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QMessageBox
import numpy as np

EXAMPLE_CUT = './tests/data/pickle_cut.p'
CMAP, CMAP_IDX = 'inferno', 27
N_SLIDER_E, N_SLIDER_K, N_BINS_E = 80, 160, 3
N_GAMMA = 10
K0_IDX = 374
# LONG_WT, SHORT_WT = 700, 5
LONG_WT, SHORT_WT = 300, 2


def change_spinBox(bot, widget, steps, key, time=SHORT_WT):
    for idx in range(steps):
        bot.keyPress(widget, key)
        bot.wait(time)


def fill_text(bot, widget, text, time=SHORT_WT*6):
    for char in text:
        QTest.keyClicks(widget, char)
        bot.wait(time)


def test_2Dviewer(qtbot):
    # initialize browser and open 2Dviewer
    browser = DataBrowser()
    qtbot.wait(QTest.qWaitForWindowExposed(browser))
    qtbot.add_widget(browser)
    qtbot.keyClicks(browser.file_explorer, 'tests')
    qtbot.keyClick(browser.file_explorer, Qt.Key_Right)
    qtbot.wait(LONG_WT)
    qtbot.keyClicks(browser.file_explorer, 'data')
    qtbot.keyClick(browser.file_explorer, Qt.Key_Right)
    qtbot.wait(LONG_WT)
    qtbot.keyClick(browser.file_explorer, Qt.Key_Down)
    assert browser.dp_anal_e0.text() is not '-'
    qtbot.wait(LONG_WT)
    browser.open_dv(EXAMPLE_CUT)
    viewer = browser.data_viewers[EXAMPLE_CUT]
    assert isinstance(viewer, DataViewer2D)
    up = viewer.util_panel
    qtbot.wait(LONG_WT)

    # move sliders
    e_start_idx, k_start_idx = up.energy_vert.value(), up.momentum_hor.value()
    change_spinBox(qtbot, up.energy_vert, N_SLIDER_E, Qt.Key_Down)
    assert up.energy_vert.value() is (e_start_idx - N_SLIDER_E)

    change_spinBox(qtbot, up.momentum_hor, N_SLIDER_K, Qt.Key_Down)
    assert up.momentum_hor.value() is (k_start_idx - N_SLIDER_K)
    qtbot.wait(LONG_WT)

    # bin one and move it back
    qtbot.mouseClick(up.bin_z, Qt.LeftButton)
    assert up.bin_z.isChecked() is True
    change_spinBox(qtbot, up.bin_z_nbins, N_BINS_E, Qt.Key_Down,
                   time=SHORT_WT * 20)
    qtbot.wait(LONG_WT)
    change_spinBox(qtbot, up.energy_vert, N_SLIDER_E, Qt.Key_Up)
    assert up.energy_vert.value() is e_start_idx
    qtbot.wait(LONG_WT)

    # move to ImageTab and change colors
    up.tabs.setCurrentIndex(1)
    assert up.tabs.currentIndex() is 1
    qtbot.wait(LONG_WT)
    qtbot.keyClicks(up.image_cmaps, CMAP)
    assert up.image_cmaps.currentIndex() is CMAP_IDX
    qtbot.wait(LONG_WT)
    qtbot.mouseClick(up.image_invert_colors, Qt.LeftButton)
    assert up.image_invert_colors.isChecked() is True
    qtbot.wait(LONG_WT)
    change_spinBox(qtbot, up.image_gamma, N_GAMMA, Qt.Key_Down,
                   time=(LONG_WT // 4))
    assert np.abs(up.image_gamma.value() - 0.5) <= 1e-3
    qtbot.wait(LONG_WT)

    # normalize data set along different directions and bring colors
    qtbot.mouseClick(up.image_normalize, Qt.LeftButton)
    assert up.image_normalize.isChecked() is True
    qtbot.wait(LONG_WT * 2)
    qtbot.keyClicks(up.image_normalize_along, 'energy')
    assert up.image_normalize_along.currentIndex() is 1
    qtbot.wait(LONG_WT * 2)
    qtbot.mouseClick(up.image_normalize, Qt.LeftButton)
    assert up.image_normalize.isChecked() is False
    qtbot.wait(LONG_WT)
    qtbot.mouseClick(up.image_invert_colors, Qt.LeftButton)
    assert up.image_invert_colors.isChecked() is False
    qtbot.wait(LONG_WT)

    # move to AxesTab, change energy scales, do k-space conversion and reset it
    up.tabs.setCurrentIndex(2)
    assert up.tabs.currentIndex() is 2
    qtbot.wait(LONG_WT)
    qtbot.keyClicks(up.axes_energy_scale, 'kinetic')
    assert up.axes_energy_scale.currentIndex() is 1
    qtbot.wait(LONG_WT * 2)
    qtbot.keyClicks(up.axes_energy_scale, 'binding')
    assert up.axes_energy_scale.currentIndex() is 0
    qtbot.wait(LONG_WT * 2)
    qtbot.mouseClick(up.axes_gamma_x, Qt.LeftButton)
    qtbot.keyPress(up.axes_gamma_x, Qt.Key_Delete)
    # QTest.keyClicks(up.axes_gamma_x, str(K0_IDX))
    fill_text(qtbot, up.axes_gamma_x, str(K0_IDX))
    qtbot.keyPress(up.axes_gamma_x, Qt.Key_Return)
    assert up.axes_gamma_x.value() == K0_IDX
    qtbot.wait(LONG_WT * 2)
    qtbot.keyClicks(up.axes_slit_orient, 'vertical')
    assert up.axes_slit_orient.currentIndex() is 1
    qtbot.wait(LONG_WT * 2)
    up.axes_do_kspace_conv.click()
    assert isinstance(viewer.k_axis, np.ndarray)
    qtbot.wait(LONG_WT * 2)

    # move to FileTab, show provenance and metadata windows, add/remove entries
    up.tabs.setCurrentIndex(3)
    assert up.tabs.currentIndex() is 3
    qtbot.wait(LONG_WT)
    qtbot.mouseClick(up.file_show_dp_button, Qt.LeftButton)
    assert isinstance(up.dp_box, InfoWindow)
    qtbot.wait(LONG_WT * 3)
    up.dp_box.close()
    qtbot.wait(LONG_WT)
    qtbot.mouseClick(up.file_show_md_button, Qt.LeftButton)
    assert isinstance(up.info_box, InfoWindow)
    qtbot.wait(QTest.qWaitForWindowExposed(up.info_box))
    qtbot.wait(LONG_WT)
    qtbot.mouseClick(up.info_box, Qt.LeftButton)
    change_spinBox(qtbot, up.info_box.central_widget, 30, Qt.Key_Down,
                   time=SHORT_WT*2)
    qtbot.wait(LONG_WT)
    up.info_box.close()

    # reset k-space conversion
    up.axes_reset_conv.click()
    assert viewer.k_axis is None

    qtbot.wait(LONG_WT * 2)
    viewer.close()
