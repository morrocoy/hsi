# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:10:02 2021

@author: kpapke

Inspector for hyperspectral images. The spectrum is plotted at user defined
corrdinated. Various filters may be applied on both, the image and spectral
directions. The RGB picture is derived from the hyperspectral data using an
in-build RGB filter is able to extract the different color channels. Besides
the raw hsformat, the hyperspectral data may be visualized as absorption,
extinction, or refraction values. In addition, the data may be visualized with
a standard normal variate correction.
"""
import sys
import logging

from pyqtgraph.Qt import QtWidgets 
import pyqtgraph as pg

import hsi

from hsi.gui import PosnImagCtrlItem
from hsi.gui import QHSImageConfigWidget

from hsi.log import logmanager

logger = logmanager.getLogger(__name__)


class QHSImageViewerWidget(QtWidgets.QWidget):

    def __init__(self, *args, **kwargs):
        QtWidgets.QWidget.__init__(self)

        # internal buffers
        self.spectra = None  # unfiltered hyperspectral data
        self.fspectra = None  # filtered hyperspectral data
        self.wavelen = None  # wavelength axis

        # image and spectral attenuation plots
        self.imageCtrlItem = PosnImagCtrlItem("RGB Image")
        self.spectralPlotItem = pg.PlotItem()

        self.curveItems = {
            'raw': pg.PlotCurveItem(
                name="raw spectral data",
                pen=pg.mkPen(color=(100, 100, 100), width=1)),
            'flt': pg.PlotCurveItem(
                name="filtered spectrum",
                pen=pg.mkPen(color=(255, 255, 255), width=1)),
        }

        # config widgets
        self.hsImageConfig = QHSImageConfigWidget()

        # set view
        self._setupViews(*args, **kwargs)
        logger.debug("HSImageViewer initialized")

    def _setupViews(self, *args, **kwargs):
        # basic layout configuration
        self.mainLayout = QtWidgets.QHBoxLayout()
        self.mainLayout.setContentsMargins(5, 0, 5, 0)  # (l, t, r, b)
        self.mainLayout.setSpacing(0)
        self.setLayout(self.mainLayout)

        # configure image plot item
        self.imageCtrlItem.setAspectLocked()
        self.imageCtrlItem.invertY()
        # self.imagCtrlItem.setMaximumWidth(440)
        # self.imagCtrlItem.setMaximumHeight(350)

        # configure spectral plot item
        self.spectralPlotItem.setLabel('bottom', "wavelength", "m")
        self.spectralPlotItem.setLabel('left', "spectral attenuation")
        for item in self.curveItems.values():
            self.spectralPlotItem.addItem(item)

        self.graphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self.graphicsLayoutWidget.addItem(self.imageCtrlItem, 0, 0)
        self.graphicsLayoutWidget.addItem(self.spectralPlotItem, 0, 1)
        qGraphicsGridLayout = self.graphicsLayoutWidget.ci.layout
        qGraphicsGridLayout.setColumnStretchFactor(0, 1)
        qGraphicsGridLayout.setColumnStretchFactor(1, 1)
        self.mainLayout.addWidget(self.graphicsLayoutWidget)

        # user config widget
        self.hsImageConfig.setMaximumWidth(200)
        # self.gLayout2.addStretch()
        self.mainLayout.addWidget(self.hsImageConfig)

        # connect signals
        self.hsImageConfig.sigValueChanged.connect(self.updateImage)
        self.imageCtrlItem.sigCursorPositionChanged.connect(
            self.updateSpectralView)

        # dark theme
        # self.setStyleSheet(
        #     "color: rgb(150,150,150);"
        #     "background-color: black;"
        #     "selection-color: white;"
        #     "selection-background-color: rgb(0,118,211);"
        #     "selection-border-color: blue;"
        #     "border-style: outset;"
        #     "border-width: 1px;"
        #     "border-radius: 2px;"
        #     "border-color: grey;"
        # )

    def updateImage(self, hsImageConfig, newFile):
        """Update hyperspectral image information."""
        self.imageCtrlItem.setData({"image": hsImageConfig.getImage()})
        if newFile:
            self.imageCtrlItem.autoRange()
            self.imageCtrlItem.setCursorPos((0, 0))

        self.spectra = self.hsImageConfig.getSpectra(filter=False)
        self.fspectra = self.hsImageConfig.getSpectra(filter=True)
        self.wavelen = self.hsImageConfig.getWavelen()

        self.updateSpectralView()

    def updateSpectralView(self):
        """Update spectral view according to the current cursor coordinates."""
        if self.spectra is None:
            return

        x, y = self.imageCtrlItem.getCursorPos()
        col = int(x)
        row = int(y)
        nwav, nrows, ncols = self.spectra.shape  # row-major
        if col < 0 or col >= ncols or row < 0 or row >= nrows:
            raise ValueError("Position outside the image {}".format([x, y]))

        x = self.wavelen
        y1 = self.spectra[:, row, col]  # raw spectral data
        y2 = self.fspectra[:, row, col]  # filtered spectral data
        self.curveItems['raw'].setData(x, y1)
        self.curveItems['flt'].setData(x, y2)


def main():
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python version: {}".format(sys.version))
    logger.info("PyQt bindings: {}".format(pg.Qt.QT_LIB))
    logger.info("PyQtGraph version: {}".format(pg.__version__))

    app = QtWidgets.QApplication([])

    win = QHSImageViewerWidget()
    win.setGeometry(300, 30, 1200, 500)
    win.setWindowTitle("Hyperspectral Image Analysis")
    win.show()
    app.exec_()

    # pg.mkQApp()
    #
    # win = QHSImageFitWidget(dir=dataPath, config=confPath)
    # win = QHSImageViewerWidget()
    # win.setGeometry(300, 30, 1200, 500)
    # win.setWindowTitle("Hyperspectral Image Analysis")
    # win.show()
    #
    # if (sys.flags.interactive != 1) or not hasattr(pg.QtCore, 'PYQT_VERSION'):
    #     pg.QtWidgets.QApplication.instance().exec_()


if __name__ == '__main__':
    logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()
