import sys
import logging

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

import hsi
from hsi.gui import BaseImagCtrlItem
from hsi.gui import HistImagCtrlItem
from hsi.log import logmanager

logger = logmanager.getLogger(__name__)


class DemoWindow(QtWidgets.QMainWindow):

    def __init__(self, cmap, showHistogram, parent=None):
        super(DemoWindow, self).__init__(parent=parent)

        self._setupViews(cmap, showHistogram)


    def _setupViews(self, cmap, showHistogram):
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


        self.controlItem1 = HistImagCtrlItem(label="ImageHistControlItem21", cmap=cmap, showHistogram=showHistogram, cbarWidth=10)
        self.controlItem1.setData({'a': img1})
        self.controlItem2 = HistImagCtrlItem(label="ImageHistControlItem2", cmap=cmap, showHistogram=showHistogram, cbarWidth=10)
        self.controlItem2.setData({'b': img2})

        self.imagCtrlItems = []
        self.imagCtrlItems.append(self.controlItem1)
        self.imagCtrlItems.append(self.controlItem2)

        self.controlItem1.setXYLink(self.controlItem2)
        self.controlItem1.sigCursorPositionChanged.connect(
            self.updateCursorPosition)
        self.controlItem2.sigCursorPositionChanged.connect(
            self.updateCursorPosition)


        self.graphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self.graphicsLayoutWidget.addItem(self.controlItem1, 0, 0)
        self.graphicsLayoutWidget.addItem(self.controlItem2, 0, 1)

        self.mainLayout.addWidget(self.graphicsLayoutWidget)
        # self.controlItem1.sigCursorPositionChanged.connect(self.updateCursorPosition)
        self.controlItem1.setCursorPos([100, 150])

        self.mainLayout.addWidget(self.graphicsLayoutWidget)


    def updateCursorPosition(self):
        sender = self.sender()
        if not isinstance(sender, BaseImagCtrlItem):
            raise TypeError("Unexpected type {}, was expecting {}"
                            .format(type(sender), BaseImagCtrlItem))

        pos = sender.getCursorPos()
        for item in self.imagCtrlItems:  # link cursors
            if item is not sender:
                item.blockSignals(True)
                item.setCursorPos(pos)
                item.blockSignals(False)

        logger.debug("Update cursor position. Sender: {}".format(sender))



def main():

    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python version: {}".format(sys.version))
    logger.info("PyQt bindings: {}".format(pg.Qt.QT_LIB))
    logger.info("PyQtGraph version: {}".format(pg.__version__))

    app = QtWidgets.QApplication([])


    cmap = pg.ColorMap([0, 0.25, 0.75, 1], [[0, 0, 0, 255], [255, 0, 0, 255], [255, 255, 0, 255], [255, 255, 255, 255]])
    cmap0 = cmap.getLookupTable()
    cmap1 = np.array([(237,248,251), (178,226,226), (102,194,164), (35,139,69), (0, 0, 0)])
    cmap2 = np.array([(237, 248, 251), (204, 236, 230), (153, 216, 201), (102, 194, 164),
                     (65, 174, 118), (35, 139, 69), (0, 88, 36)])
    # alpha = 100
    # cmap2 = np.hstack((cmap2, alpha * np.ones((7, 1))))

    cmap = cmap2.astype(np.uint8) # Use uint8 so that the resulting image will also be of that type/
    cmap = np.flipud(cmap) # test reversed map
    win = DemoWindow(cmap=cmap, showHistogram=True)
    # win = QImageLevelsWidget(cmap=cmap, showHistogram=True)
    win.setGeometry(400, 100, 1000, 500)
    win.setWindowTitle('PgColorbar Demo')
    win.show()
    app.exec_()


if __name__ == '__main__':
    logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()