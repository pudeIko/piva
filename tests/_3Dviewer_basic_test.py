import pytestqt
import os
from piva.data_browser import DataBrowser
from piva.imageplot import InfoWindow
from piva._3Dviewer import MainWindow3D
from piva._2Dviewer import MainWindow2D
from pyqtgraph.Qt.QtCore import Qt
from PyQt5.QtTest import QTest
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QMessageBox
import numpy as np

EXAMPLE_CUT = './tests/data/pickle_map.p'
CMAP, CMAP_IDX = 'inferno', 27
N_E, N_X, N_Y = 20, 3, 9
N_GAMMA = 10
K0_IDX = 374
LONG_WT, SHORT_WT = 700, 5


def change_spinBox(bot, widget, steps, key, time=SHORT_WT):
    for _ in range(steps):
        bot.keyPress(widget, key)
        bot.wait(time)


def fill_text(bot, widget, text, time=SHORT_WT*6):
    for char in text:
        QTest.keyClicks(widget, char)
        bot.wait(time)


def test_3Dviewer(qtbot):
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
    qtbot.keyClick(browser.file_explorer, Qt.Key_Down)
    assert browser.dp_anal_e0.text() is not '-'
    qtbot.wait(LONG_WT)
    browser.open_dv(EXAMPLE_CUT)
    viewer = browser.data_viewers[EXAMPLE_CUT]
    qtbot.wait(QTest.qWaitForWindowExposed(viewer))
    assert isinstance(viewer, MainWindow3D)
    up = viewer.util_panel
    qtbot.wait(LONG_WT)

    # move sliders
    qtbot.mouseClick(up.bin_x, Qt.LeftButton)
    qtbot.wait(LONG_WT)
    qtbot.mouseClick(up.bin_y, Qt.LeftButton)
    assert up.bin_x.isChecked() is True
    assert up.bin_y.isChecked() is True
    e0, ex0, ey0 = up.energy_main.value(), up.energy_hor.value(), \
                   up.energy_vert.value()
    x0_idx, y0_idx = up.momentum_vert.value(), up.momentum_hor.value()

    change_spinBox(qtbot, up.momentum_vert, N_X, Qt.Key_Up)
    assert up.momentum_vert.value() is (x0_idx + N_X)
    change_spinBox(qtbot, up.momentum_hor, N_Y, Qt.Key_Down, time=SHORT_WT//5)
    assert up.momentum_hor.value() == (y0_idx - N_Y)
    qtbot.wait(LONG_WT)

    for _ in range(N_E):
        qtbot.keyPress(up.energy_main, Qt.Key_Down)
        qtbot.keyPress(up.energy_hor, Qt.Key_Down)
        qtbot.keyPress(up.energy_vert, Qt.Key_Down)
        qtbot.wait(SHORT_WT // 5)
    assert up.energy_main.value() == (e0 - N_E)
    assert up.energy_hor.value() == (ex0 - N_E)
    assert up.energy_vert.value() == (ey0 - N_E)
    qtbot.wait(LONG_WT)

    # move to OrientateTab and find gamma
    up.tabs.setCurrentIndex(3)
    assert up.tabs.currentIndex() is 3
    qtbot.wait(LONG_WT)
    qtbot.mouseClick(up.orientate_copy_coords, Qt.LeftButton)
    qtbot.wait(LONG_WT)
    # qtbot.mouseClick(up.orientate_find_gamma, Qt.LeftButton)
    # assert 'Success' in up.orientate_find_gamma_message.text()
    # qtbot.wait(LONG_WT)
    qtbot.mouseClick(up.orientate_hor_line, Qt.LeftButton)
    qtbot.mouseClick(up.orientate_ver_line, Qt.LeftButton)
    assert up.orientate_hor_line.isChecked() is True
    assert up.orientate_ver_line.isChecked() is True
    change_spinBox(qtbot, up.orientate_angle, 60, Qt.Key_Up, time=SHORT_WT*2)
    qtbot.wait(LONG_WT * 2)
    qtbot.mouseClick(up.orientate_hor_line, Qt.LeftButton)
    qtbot.mouseClick(up.orientate_ver_line, Qt.LeftButton)
    assert up.orientate_hor_line.isChecked() is False
    assert up.orientate_ver_line.isChecked() is False

    # move to AxesTab, change energy scales
    up.tabs.setCurrentIndex(2)
    assert up.tabs.currentIndex() is 2
    qtbot.wait(LONG_WT)
    qtbot.keyClicks(up.axes_energy_scale, 'kinetic')
    assert up.axes_energy_scale.currentIndex() is 1
    qtbot.wait(LONG_WT * 2)
    qtbot.keyClicks(up.axes_energy_scale, 'binding')
    assert up.axes_energy_scale.currentIndex() is 0
    qtbot.wait(LONG_WT * 2)
    qtbot.mouseClick(up.axes_copy_values, Qt.LeftButton)
    qtbot.wait(LONG_WT)

    # move to ImageTab, normalize data and open 2Dviewer
    up.tabs.setCurrentIndex(1)
    assert up.tabs.currentIndex() is 1
    qtbot.wait(LONG_WT)
    qtbot.mouseClick(up.image_normalize, Qt.LeftButton)
    assert up.image_normalize.isChecked() is True
    qtbot.wait(LONG_WT * 2)
    qtbot.keyClicks(up.image_normalize_along, 'energy')
    assert up.image_normalize_along.currentIndex() is 2
    qtbot.wait(LONG_WT)
    qtbot.mouseClick(up.image_normalize, Qt.LeftButton)
    assert up.image_normalize.isChecked() is False
    qtbot.wait(LONG_WT)
    qtbot.mouseClick(up.image_2dv_button, Qt.LeftButton)
    for key in browser.data_viewers.keys():
        if ':' in key:
            new_window = key
    assert isinstance(browser.data_viewers[new_window], MainWindow2D)
    qtbot.wait(LONG_WT * 3)
    qtbot.mouseClick(browser.data_viewers[new_window].util_panel.close_button,
                     Qt.LeftButton)

    qtbot.wait(LONG_WT * 2)
    viewer.close()
