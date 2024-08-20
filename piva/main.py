import pkg_resources
import os
import sys
from PyQt5.QtWidgets import QApplication
from piva.data_browser import DataBrowser

# to fix bugs in Big Siur
os.environ['QT_MAC_WANTS_LAYER'] = '1'


def db():
    version = pkg_resources.require('piva')[0].version
    print(f'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'  PIVA - Photoemission Interface for Visualization and Analysis  ')
    print(f'                        Version {version}                        ')
    print(f'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    app = QApplication(sys.argv)
    window = DataBrowser()
    sys.exit(app.exec_())


if __name__ == "__main__":
    db()
