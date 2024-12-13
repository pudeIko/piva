"""
Semi-automated test for testing jupyter-lab functionalities.
"""
from piva.data_browser import DataBrowser
from piva.data_viewer_3d import DataViewer3D
from pyqtgraph.Qt.QtCore import Qt
from PyQt5.QtTest import QTest
from pkg_resources import resource_filename
from PyQt5.QtWidgets import QMessageBox
import os
import signal
from typing import Any

USER = False

PATH = os.path.join(resource_filename('piva', 'tests'), 'data')
EXAMPLE = os.path.join(PATH, 'test_map.p')
BL, BL_IDX = 'SIS', 1
PORT = '56789'
LONG_WT, SHORT_WT = 300, 1


class TestJupyterUtilities:
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
        self.browser = browser
        browser.open_dv(EXAMPLE)
        self._3dv = browser.data_viewers[EXAMPLE]
        qtbot.wait(QTest.qWaitForWindowExposed(self._3dv))
        assert isinstance(self._3dv, DataViewer3D)
        self.up = self._3dv.util_panel

    def create_logbook_file(self, qtbot: Any) -> None:
        """
        Test features creating example experimental logbook in
        :class:`data_viewer_3d.DataViewer3D`.

        :param qtbot: object emulating a user
        """

        # move to FileTab
        self.up.tabs.setCurrentIndex(4)
        assert self.up.tabs.currentIndex() == 4
        qtbot.wait(LONG_WT)
        qtbot.keyClicks(self.up.file_jl_explog, BL)
        assert self.up.file_jl_explog.currentIndex() == BL_IDX
        qtbot.wait(LONG_WT * 5)

        if USER:
            qtbot.mouseClick(self.up.file_jl_explog_button, Qt.LeftButton)
        else:
            self.up.create_experimental_logbook_file(directory=PATH)

    def create_jupyter_notebook(self, qtbot: Any) -> None:
        """
        Test features creating example jupyter notebook for loading and
        analysing data :class:`data_viewer_3d.DataViewer3D`.

        :param qtbot: object emulating a user
        """

        if USER:
            qtbot.mouseClick(self.up.file_jl_fname_button, Qt.LeftButton)
        else:
            self.up.create_jl_file(directory=PATH)

    def start_jupyter_session(self, qtbot: Any) -> None:
        """
        Test features starting new Jupyter server.

        :param qtbot: object emulating a user
        """

        if USER:
            qtbot.mouseClick(self.up.file_jl_session_button, Qt.LeftButton)
        else:
            self.jl_server_pid = self.up.open_jl_session(directory=PATH,
                                                         port=PORT)

    def test_jupyter_utilities(self, qtbot: Any) -> None:
        """
        Run the test.

        :param qtbot: object emulating a user
        """

        self.open_browser(qtbot)
        qtbot.wait(LONG_WT * 2)
        self.create_logbook_file(qtbot)
        qtbot.wait(LONG_WT * 2)
        self.create_jupyter_notebook(qtbot)
        qtbot.wait(LONG_WT * 2)
        self.start_jupyter_session(qtbot)
        qtbot.wait(LONG_WT * 2)

        kill_server_box = QMessageBox()
        kill_server_box.setIcon(QMessageBox.Information)
        kill_server_box.setText('Shut down Jupyter server?')
        kill_server_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        if kill_server_box.exec() == QMessageBox.Ok:
            try:
                os.kill(self.jl_server_pid, signal.SIGKILL)
            except Exception as e:
                pass

        del_files_box = QMessageBox()
        del_files_box.setIcon(QMessageBox.Information)
        del_files_box.setText('Delete created notebooks?')
        del_files_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        if del_files_box.exec() == QMessageBox.Cancel:
            return

        os.remove(os.path.join(PATH, 'metadata-SIS.ipynb'))
        os.remove(os.path.join(PATH, 'test_map.ipynb'))
        try:
            os.remove(os.path.join(PATH, 'Untitled.ipynb'))
        except FileNotFoundError:
            pass

        qtbot.wait(LONG_WT * 2)
        self._3dv.close()


if __name__ == "__main__":
    import pytest
    from pkg_resources import resource_filename

    path = os.path.join(resource_filename('piva', 'tests'), 'jupyter_test.py')
    pytest.main(['-v', '-s', path])
