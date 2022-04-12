
import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow, QWidget, QGridLayout, QLabel
from pyqtgraph import PlotWidget, PColorMeshItem, InfiniteLine
import numpy as np
from pyqtgraph.graphicsItems.ImageItem import ImageItem
from PIL import Image

BASE_LINECOLOR = (255, 255, 0, 255)
BINLINES_LINECOLOR = (168, 168, 104, 255)
HOVER_COLOR = (195, 155, 0, 255)
BGR_COLOR = (64, 64, 64)
util_panel_style = """
QFrame{margin:5px; border:1px solid rgb(150,150,150);}
QLabel{color: rgb(246, 246, 246); border:1px solid rgb(64, 64, 64);}
QCheckBox{color: rgb(246, 246, 246);}
"""
app_style = """
QMainWindow{background-color: rgb(64,64,64);}
# """
os.environ['QT_MAC_WANTS_LAYER'] = '1'


def dummy():
    app = QApplication(sys.argv)
    # MainWindow()
    mw = MainWindow()
    mw.show()
    app.exec()


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.central_widget = QWidget()
        self.layout = QGridLayout()

        # Create the 3D (main) and cut ImagePlots
        self.main_plot = ImagePlot(name='main_plot')

        self.setStyleSheet(app_style)
        self.setWindowTitle('elo')
        self.setGeometry(50, 50, 500, 500)
        # Set the loaded data in PIT
        fname = 'image0.png'
        self.title = fname
        img = np.asarray(Image.open(fname).convert("L"))

        #initUI
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        # # Align all the gui elements
        self._align()
        self.show()
        print('elo')
        self.main_plot.set_image(img)

    def _align(self):
        """ Align all the GUI elements in the QLayout::

              0   1   2   3
            +---+---+---+---+
            |utilities panel| 0
            +---+---+---+---+
            | mdc x |       | 1
            +-------+  edc  |
            | cut x |       | 2
            +-------+-------+
            |       | c | m | 3
            | main  | y | y | 4
            +---+---+---+---+

            (Units of subdivision [sd])
        """
        # subdivision
        # Get a short handle
        l = self.layout
        l.addWidget(self.main_plot, 0, 0, 2, 2)


class ImagePlot(PlotWidget):
    """
    A PlotWidget which mostly contains a single 2D image (intensity
    distribution) or a 3D array (distribution of RGB values) as well as all
    the nice pyqtgraph axes panning/rescaling/zooming functionality.

    In addition, this allows one to use custom axes scales as opposed to
    being limited to pixel coordinates.

    **Signals**

    =================  =========================================================
    sig_image_changed  emitted whenever the image is updated
    sig_axes_changed   emitted when the axes are updated
    sig_clicked        emitted when user clicks inside the imageplot
    =================  =========================================================
    """

    def __init__(self, image=None, name=None, parent=None, background=BGR_COLOR, **kwargs):
        """ Allows setting of the image upon initialization.

        **Parameters**

        ==========  ============================================================
        image       np.ndarray or pyqtgraph.ImageItem instance; the image to be
                    displayed.
        parent      QtWidget instance; parent widget of this widget.
        background  str; confer PyQt documentation
        name        str; allows giving a name for debug purposes
        ==========  ============================================================
        """
        # Initialize instance variables
        self.image_item = None

        super().__init__(parent=parent, background=background, **kwargs)

        if image is not None:
            self.set_image(image)

        # Initiliaze a crosshair and add it to this widget
        self.crosshair = Crosshair(self)
        self.crosshair.add_to(self)

        self.pos = (self.crosshair.vpos, self.crosshair.hpos)

    def set_image(self, image, *args, **kwargs):
        """ Expects either np.arrays or pg.ImageItems as input and sets them
        correctly to this PlotWidget's Image with `addItem`. Also makes sure
        there is only one Image by deleting the previous image.

        Emits :signal:`sig_image_changed`

        **Parameters**

        ========  ==============================================================
        image     np.ndarray or pyqtgraph.ImageItem instance; the image to be
                  displayed.
        emit      bool; whether or not to emit :signal:`sig_image_changed`
        (kw)args  positional and keyword arguments that are passed on to
                  :class:`pyqtgraph.ImageItem`
        ========  ==============================================================
        """
        # Convert array to ImageItem
        if isinstance(image, np.ndarray):
            if 0 not in image.shape:
                # image_item = ImageItem(image, *args, **kwargs)
                image_item = PColorMeshItem(image)
            else:
                return
        else:
            image_item = image

        # Replace the image
        self.image_item = image_item
        self.addItem(image_item)


class Crosshair:
    """ Crosshair made up of two InfiniteLines. """
    def __init__(self, image_plot, pos=(0, 0)):
        self.image_plot = image_plot

        self.hpos = 10
        self.vpos = 10

        # Initialize the InfiniteLines
        self.hline = InfiniteLine(pos[1], movable=True, angle=0)
        self.vline = InfiniteLine(pos[0], movable=True, angle=90)

        # Set the color
        self.set_color(BASE_LINECOLOR, HOVER_COLOR)

    def add_to(self, widget):
        """ Add this crosshair to a Qt widget. """
        for line in [self.hline, self.vline]:
            line.setZValue(1)
            widget.addItem(line)

    def set_color(self, linecolor=BASE_LINECOLOR, hover_color=HOVER_COLOR):
        """ Set the color and hover color of both InfiniteLines that make up
        the crosshair. The arguments can be any pyqtgraph compatible color
        specifiers.
        """
        for line in [self.hline, self.vline]:
            line.setPen(linecolor)
            line.setHoverPen(hover_color)


dummy()
