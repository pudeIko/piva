# main for calling files' openers

import pkg_resources
import os
import sys

from PyQt5.QtWidgets import QApplication

import piva._2Dviewer as p2d
import piva._3Dviewer as p3d
import piva.data_loader as dl
from piva.data_browser import DataBrowser

# to fix bugs in Big Siur
os.environ['QT_MAC_WANTS_LAYER'] = '1'

# TODO DOCUMANTATION !!!!!!!!!!!
# TODO glue scans together


def db():
    version = pkg_resources.require('piva')[0].version
    print(f'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'        PIVA - Python Interactive Viewer for Arpes         ')
    print(f'                    Version {version}                      ')
    print(f'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    app = QApplication(sys.argv)
    window = DataBrowser()
    sys.exit(app.exec_())


def pickle_h5():
    fname = sys.argv[1]
    data = dl.load_data(fname)
    dl.dump(data, fname[:-3])


if __name__ == "__main__" :
    db()
