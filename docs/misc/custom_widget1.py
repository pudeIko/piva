from __future__ import annotations
from typing import TYPE_CHECKING, Any
from pyqtgraph.Qt import QtWidgets

# from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import QLabel, QPushButton

if TYPE_CHECKING:
    from piva.data_browser import DataBrowser

app_style = """
QMainWindow{background-color: rgb(64,64,64);}
QTabWidget{background-color: rgb(64,64,64);}
QLabel{color: rgb(246, 246, 246); border:1px solid rgb(64, 64, 64);}
"""


class CustomWidget(QtWidgets.QMainWindow):
    """
    Example class for a **CustomWidgets**.
    """

    def __init__(self, data_browser: DataBrowser, index: str = None) -> None:
        """
        Initialize main window with some basic parameters.

        :param data_browser: `DataBrowser` of the current session
        :param index: title of the window and its index in the record of opened
                      **CustomWidgets**
        """

        super(CustomWidget, self).__init__()
        self.central_widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QGridLayout()
        self.setStyleSheet(app_style)
        self.setWindowTitle("Custom Widget")
        self.setGeometry(100, 100, 500, 400)
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        # create a reference to the DataBrowser
        self.db = data_browser
        self.index = index

        self.set_window()

        self.show()

    def set_window(self) -> None:
        """
        Example method for setting up the window, inserting widgets, *etc*.
        """

        # initiate basic widgets
        labell = QLabel("Here you can place your stuff.")
        button = QPushButton("Close")
        dummy1 = QLabel("")
        dummy2 = QLabel("")

        # add the widgets to the layout
        self.layout.addWidget(dummy1, 0, 0)
        self.layout.addWidget(labell, 1, 1)
        self.layout.addWidget(button, 2, 1)
        self.layout.addWidget(dummy2, 3, 3)

        # connect some actions/methods
        button.clicked.connect(self.closeEvent)

    def closeEvent(self, event: Any) -> None:
        """
        Ensure that this instance is closed and un-registered from the
        :class:`~data_browser.DataBrowser`.
        """

        del self.db.custom_plugins[self.index]
