import pkg_resources
import os
import sys
from PyQt5.QtWidgets import QApplication
from piva.data_browser import DataBrowser

# to fix bugs in Big Siur
os.environ['QT_MAC_WANTS_LAYER'] = '1'


def db(start_event_loop=True):
    version = pkg_resources.require('piva')[0].version
    print( '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print( '  PIVA - Photoemission Interface for Visualization and Analysis  ')
    print(f'                        Version {version}                        ')
    print( '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    app = QApplication(sys.argv)
    DataBrowser()
    sys.exit(app.exec_())
    # browser = DataBrowser()

    # if start_event_loop:
    #     sys.exit(app.exec_())
    # return browser  # allow use without blocking


if __name__ == "__main__":
    db()
