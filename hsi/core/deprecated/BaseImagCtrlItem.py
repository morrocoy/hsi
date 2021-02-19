import sys
import copy

import numpy as np
import pyqtgraph as pg

from ..qt import QtWidgets, QtCore
from ..misc import check_is_an_array, check_class

from .ColorBarItem import ColorBarItem
from .InfiniteLine import InfiniteLine

import logging

LOGGING = True
# LOGGING = False

logger = logging.getLogger(__name__)
logger.propagate = LOGGING


__all__ = ['BaseImagCtrlItem']


class BaseImagCtrlItem(pg.GraphicsWidget):

    sigCursorPositionChangeFinished = QtCore.Signal(object)
    sigCursorPositionChanged = QtCore.Signal(object)

    def __init__(self, *args, **kwargs):

        parent = kwargs.get('parent', None)
        pg.GraphicsWidget.__init__(self, parent)

        self.linkedImageControlItem = None

        self.plotItem = pg.PlotItem()

        self.imageItem = pg.ImageItem()
        self.plotItem.addItem(self.imageItem)

        self.cursorX = InfiniteLine(angle=90, movable=True, pen=(150, 150, 150), hoverPen=(255, 255, 255))
        self.cursorY = InfiniteLine(angle=0, movable=True, pen=(150, 150, 150), hoverPen=(255, 255, 255))
        self.cursorX.setPos(0)
        self.cursorY.setPos(0)
        self.cursorX.setZValue(10)
        self.cursorY.setZValue(11)
        self.cursorX.connect(self.cursorY)
        self.cursorY.connect(self.cursorX)

        self.plotItem.addItem(self.cursorX, ignoreBounds=True)
        self.plotItem.addItem(self.cursorY, ignoreBounds=True)
        self.plotViewBox = self.plotItem.getViewBox()

        self.mainLayout = QtWidgets.QGraphicsGridLayout()
        self.setLayout(self.mainLayout)
        self.mainLayout.setContentsMargins(1, 1, 1, 1)
        self.mainLayout.setSpacing(0)

        self.mainLayout.addItem(self.plotItem, 0, 0)

        # Connect signals
        self.cursorX.sigPositionChangeFinished.connect(self.cursorPositionChangeFinishedEvent)
        self.cursorY.sigPositionChangeFinished.connect(self.cursorPositionChangeFinishedEvent)
        self.cursorX.sigPositionChanged.connect(self.cursorPositionChangeEvent)
        self.cursorY.sigPositionChanged.connect(self.cursorPositionChangeEvent)


    def cursorPositionChangeFinishedEvent(self):
        # print(ev.pos())
        self.sigCursorPositionChangeFinished.emit(self)
        logger.debug("emit cursorPositionChangeFinished")


    def cursorPositionChangeEvent(self):
        # print(ev.pos())
        self.sigCursorPositionChanged.emit(self)

        # if not self.linkedImageControlItem is None:
        #     x = self.cursorX.getXPos()
        #     y = self.cursorY.getYPos()
        #     self.linkedImageControlItem.cursorX.setPos(x)
        #     self.linkedImageControlItem.cursorY.setPos(y)
        logger.debug("emit cursorPositionChanged")


    def getCursorPos(self):
        x = self.cursorX.getXPos()
        y = self.cursorY.getYPos()
        return [x, y]


    def setCursorPos(self, pos):
        self.cursorX.setPos(pos[0])
        self.cursorY.setPos(pos[1])


    def setAspectLocked(self, lock=True):
        self.plotItem.setAspectLocked(lock)


    def setImage(self, img):
        """ Sets the image data
        """
        # PyQtGraph uses the following dimension order: T, X, Y, Color.
        self.imageItem.setImage(img.T)
        nRows, nCols = img.shape
        self.plotItem.setRange(xRange=[0, nCols], yRange=[0, nRows])
        # self.colorBarItem.resetColorLevels()

        self.cursorX.setBounds((0, nCols))
        self.cursorY.setBounds((0, nRows))

        self.cursorX.setPos((nCols // 2))
        self.cursorY.setPos((nRows // 2))


    def setXYLink(self, graphicsItems):
        if isinstance(graphicsItems, pg.PlotItem):
            self.plotItem.setXLink(graphicsItems)
            self.plotItem.setYLink(graphicsItems)
            self.linkedImageControlItem = None
        elif isinstance(graphicsItems, BaseImagCtrlItem):
            self.plotItem.setXLink(graphicsItems.plotItem)
            self.plotItem.setYLink(graphicsItems.plotItem)
            self.linkedImageControlItem = None
            # self.linkedImageControlItem = graphicsItems
            # graphicsItems.linkedImageControlItem = self
            # self.plotItem.setXLink(self.linkedImageControlItem.plotItem)
            # self.plotItem.setYLink(self.linkedImageControlItem.plotItem)
        else:
            raise TypeError("Unexpected type {}, was expecting {}".format(
                type(graphicsItems), (pg.PlotItem, BaseImagCtrlItem)))



class DemoWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(DemoWindow, self).__init__(parent=parent)

        self._setupViews()


    def _setupViews(self):
        """ Creates the UI widgets.
        """
        self.mainWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.mainWidget)

        self.mainLayout = QtWidgets.QVBoxLayout()
        self.mainLayout.setContentsMargins(0, 0, 0, 0) # left, top, right, bottom
        self.mainLayout.setSpacing(0)
        self.mainWidget.setLayout(self.mainLayout)

        img1 = pg.gaussianFilter(np.random.normal(size=(300, 200)), (5, 5)) * 20
        img2 = pg.gaussianFilter(np.random.normal(size=(300, 200)), (5, 5)) * 20

        self.controlItem1 = BaseImagCtrlItem(label="Oxygenation")
        self.controlItem1.setImage(img1)
        self.controlItem2 = BaseImagCtrlItem(label="Oxygenation")
        self.controlItem2.setImage(img2)

        self.controlItem1.setXYLink(self.controlItem2)

        self.graphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self.graphicsLayoutWidget.addItem(self.controlItem1, 0, 0)
        self.graphicsLayoutWidget.addItem(self.controlItem2, 0, 1)


        self.mainLayout.addWidget(self.graphicsLayoutWidget)


def main():

    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python version: {}".format(sys.version))
    logger.info("PyQt bindings: {}".format(pg.Qt.QT_LIB))
    logger.info("PyQtGraph version: {}".format(pg.__version__))

    app = QtWidgets.QApplication([])


    cmap = pg.ColorMap([0, 0.25, 0.75, 1], [[0, 0, 0, 255], [255, 0, 0, 255], [255, 255, 0, 255], [255, 255, 255, 255]])
    win = DemoWindow()

    win.setGeometry(400, 100, 800, 500)
    win.setWindowTitle('PgColorbar Demo')
    win.show()
    app.exec_()


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s %(filename)25s:%(lineno)-4d : %(levelname)-7s: %(message)s'
    logging.basicConfig(level='DEBUG', format=LOG_FMT)

    main()