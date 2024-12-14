from piva.data_browser import DataBrowser
import custom_widget1 as cw1
from PyQt5.QtWidgets import QAction


class PluginImporter:
    """
    Plugin importer for setting up custom plugins written for PIVA.
    """

    def __init__(self, data_browser: DataBrowser) -> None:
        """
        Initialize PluginImporter.

        :param data_browser: `DataBrowser` of the current session
        """

        self.db = data_browser
        self.db.user_menu = self.db.menu_bar.addMenu('&User plugins')

        self.import_widget1()

    def import_widget1(self) -> None:
        """
        Example method for importing and setting up custom widget.
        """

        open_widget = QAction('Custom widget', self.db)
        open_widget.setStatusTip('Custom widget')
        open_widget.setShortcut('Ctrl+D')
        open_widget.triggered.connect(self.open_widget1)
        self.db.user_menu.addAction(open_widget)

    def open_widget1(self) -> None:
        """
        Example of a separate method necessary to open CustomWidget when
        :class:`QAction` assigned to the **CustomWidget** is triggered.
        """

        idx = 'tmp'
        self.db.custom_plugins[idx] = \
            cw1.CustomWidget(data_browser=self.db, index=idx)
        print('\t', 'CustomWidget')

