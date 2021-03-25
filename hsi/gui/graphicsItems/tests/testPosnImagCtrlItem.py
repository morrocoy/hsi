import sys
import logging
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

import hsi
from hsi.gui import PosnImagCtrlItem
from hsi.log import logmanager

logger = logmanager.getLogger(__name__)


class DemoWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        parent = kwargs.get('parent', None)
        super(DemoWindow, self).__init__(parent=parent)

        self._setupViews(*args, **kwargs)


    def _setupViews(self, *args, **kwargs):
        """ Creates the UI widgets.
        """
        self.mainWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.mainWidget)

        self.mainLayout = QtWidgets.QVBoxLayout()
        self.mainLayout.setContentsMargins(0, 0, 0, 0) # left, top, right, bottom
        self.mainLayout.setSpacing(0)
        self.mainWidget.setLayout(self.mainLayout)

        img2 = pg.gaussianFilter(np.random.normal(size=(300, 200)), (5, 5)) * 20

        self.controlItem4 = PosnImagCtrlItem(label="ImageCoordControlItem")
        self.controlItem4.setImage(img2)

        self.graphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self.graphicsLayoutWidget.addItem(self.controlItem4, 0, 0)
        self.controlItem4.setCursorPos([100,150])

        self.mainLayout.addWidget(self.graphicsLayoutWidget)



    def updateCursorPosition(self, ev):
        # print(ev.pos())
        sender = self.sender()

        # self.statusBar().showMessage(sender.text() + ' was pressed')
        # print(sender.text() + ' was pressed')
        print(sender)
        print(self.controlItem1)
        print(self.controlItem2)


        # logger.debug("emit cursorPositionChanged")


def main():

    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python version: {}".format(sys.version))
    logger.info("PyQt bindings: {}".format(pg.Qt.QT_LIB))
    logger.info("PyQtGraph version: {}".format(pg.__version__))

    app = QtWidgets.QApplication([])



    win = DemoWindow()
    win.setGeometry(400, 100, 800, 500)
    win.setWindowTitle('PgColorbar Demo')
    win.show()
    app.exec_()


if __name__ == '__main__':
    logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()