import sys
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from hsi.gui import RegnPlotCtrlItem

import logging

LOGGING = True
LOGGING = False
logger = logging.getLogger(__name__)
logger.propagate = LOGGING


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

        x = np.linspace(0, 100, 1000)
        y1 = pg.gaussianFilter(np.random.random(size=1000), 10) * 20
        y2 = pg.gaussianFilter(np.random.random(size=1000), 10) * 20
        y3 = pg.gaussianFilter(np.random.random(size=1000), 10) * 20

        self.controlItem5 = RegnPlotCtrlItem("Spectrum", xlabel="wavelength", xunits="nm")

        self.crv1 = pg.PlotCurveItem(pen=pg.mkPen(color=(255, 0, 0, 255), width=1))
        self.crv2 = pg.PlotCurveItem(pen=pg.mkPen(color=(0, 255, 0, 255), width=1))
        self.crv3 = pg.PlotCurveItem(pen=pg.mkPen(color=(0, 0, 255, 255), width=1))


        self.controlItem5.addItem(self.crv1)
        self.controlItem5.addItem(self.crv2)
        self.controlItem5.addItem(self.crv3)

        self.crv1.setData(x=x, y=y1)
        self.crv2.setData(x=x, y=y2)
        self.crv3.setData(x=x, y=y3)

        # self.controlItem5.setBounds(x[[0,999]])
        self.controlItem5.setRegion(x[[200, 800]])

        self.graphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self.graphicsLayoutWidget.addItem(self.controlItem5, 0, 0)

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
    LOG_FMT = '%(asctime)s %(filename)25s:%(lineno)-4d : %(levelname)-7s: %(message)s'
    logging.basicConfig(level='DEBUG', format=LOG_FMT)

    main()