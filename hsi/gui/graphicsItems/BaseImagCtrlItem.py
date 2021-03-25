import sys
import copy

import numpy as np
import pyqtgraph as pg

from ...bindings.Qt import QtWidgets, QtCore
from ...log import logmanager
from ...misc import check_is_an_array, check_class

from .ColorBarItem import ColorBarItem
from .InfiniteLine import InfiniteLine

logger = logmanager.getLogger(__name__)


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
        logger.debug("Emit cursorPositionChangeFinished")


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
        self.plotViewBox.setAspectLocked(lock)


    def invertY(self, enable=True):
        self.plotViewBox.invertY(enable)


    def invertX(self, enable=True):
        self.plotViewBox.invertX(enable)


    def autoRange(self, *args, **kwargs):
        self.plotViewBox.autoRange(*args, **kwargs)


    def setImage(self, data):
        """ Sets the image data
        """
        if isinstance(data, list):
            data = np.array(data)
        if not isinstance(data, np.ndarray):
            raise Exception("Plot data must be ndarray.")

        if data.ndim == 2:
            nrows, ncols = data.shape
            nchan = 1
            self.imageItem.setImage(data, axisOrder='row-major')
        elif data.ndim == 3:
            nrows, ncols, nchan = data.shape
            self.imageItem.setImage(data, axisOrder='row-major')
        else:
            raise Exception("Plot data must be 2D or 3D ndarray.")

        self.cursorX.setBounds((0, ncols-1))
        self.cursorY.setBounds((0, nrows-1))

        # width = self.imageItem.width()
        # height = self.imageItem.width()
        # self.cursorX.setPos((width // 2))
        # self.cursorY.setPos((height // 2))


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




