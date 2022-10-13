# main for calling files' openers

import os
import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon, QPixmap
# from PyQt5.QtWidgets import

from piva.data_browser import *

# to fix bugs in Big Siur
os.environ['QT_MAC_WANTS_LAYER'] = '1'

# TODO DOCUMANTATION !!!!!!!!!!!
# TODO glue scans together


def db():
    print('++++++++++++++++++++++++++++++++++++')
    print('\tdevelopement version')
    print('++++++++++++++++++++++++++++++++++++')
    app = QApplication(sys.argv)
    window = DataBrowser()
    sys.exit(app.exec_())

###################################
########### deprecated ############
###################################


def open2D():
    app = QApplication(sys.argv)
    fname = sys.argv[1]
    # if len(sys.argv) == 3:
    #     pickle = True
    # else:
    #     pickle = False
    data_set = dl.load_data(fname)
    MainWindow2D("data_browser", data_set=data_set, title=fname)
    sys.exit(app.exec_())


def open3D():
    app = QApplication(sys.argv)
    fname = sys.argv[1]
    # if len(sys.argv) == 3:
    #     pickle = True
    # else:
    #     pickle = False
    data_set = dl.load_data(fname)
    MainWindow3D("data_browser", data_set=data_set, title=fname)
    sys.exit(app.exec_())


def pickle_h5():
    fname = sys.argv[1]
    data = dl.load_data(fname)
    dl.dump(data, fname[:-3])


def open3Djn(fname, pickle=False, ns=None):
    app = QApplication(sys.argv)
    MainWindow3D(fname=fname, from_pickle=pickle, ns=ns)
    sys.exit(app.exec_())


def reshape_pickled():
    dl.reshape_pickled(sys.argv[1])

if __name__ == "__main__" :
    db()
