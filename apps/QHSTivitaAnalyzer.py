# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:29:59 2021

@author: kpapke

This example demonstrates the Tivita analysis applied on hyperspectral images.
The spectrum is plotted at user defined coordinated. Various filters may be
applied on both, the image and spectral directions. The RGB picture is derived
from the hyperspectral data using an in-build RGB filter is able to extract
the different color channels.
"""
import sys
import logging

import numpy as np
from pyqtgraph.Qt import QtWidgets, QtGui
import pyqtgraph as pg

import hsi

from hsi import HSAbsorption, HSIntensity

from hsi.gui import QHSImageConfigWidget
from hsi.gui import QHSComponentFitConfigWidget

from hsi.gui import BaseImagCtrlItem
from hsi.gui import HistImagCtrlItem
from hsi.gui import PosnImagCtrlItem
from hsi.gui import RegnPlotCtrlItem

# from hsi.analysis import HSOpenTivita
from hsi.analysis import HSTivita


from hsi.log import logmanager

logger = logmanager.getLogger(__name__)


class QHSTivitaAnalyzerWidget(QtGui.QWidget):

    def __init__(self, *args, **kwargs):
        QtGui.QWidget.__init__(self)

        # internal buffers
        self.spectra = None  # multidimensional array of unfiltered hs-data
        self.fspectra = None  # multidimensional array of filtered hs-data
        self.mspectra = None  # multidimensional array of modeled hs-data
        self.wavelen = None  # wavelength axis

        # image, 2D histogram and spectral attenuation plots
        self.imagCtrlItems = {
            'rgb': PosnImagCtrlItem("RGB Image", cbarWidth=10),
            'nir': HistImagCtrlItem("NIR Index", cbarWidth=10),
            'oxy': HistImagCtrlItem("StO2 Index", cbarWidth=10),
            'thi': HistImagCtrlItem("TH Index", cbarWidth=10),
            'twi': HistImagCtrlItem("TW Index", cbarWidth=10),
            # 'mel': HistImagCtrlItem("Melanin", cbarWidth=10),
        }

        self.spectViewer = RegnPlotCtrlItem(
            "spectral attenuation", xlabel="wavelength", xunits="m")

        self.curveItems = {
            'raw': pg.PlotCurveItem(
                name="raw spectral data",
                pen=pg.mkPen(color=(100, 100, 100), width=1)),
            'fil': pg.PlotCurveItem(
                name="filtered spectrum",
                pen=pg.mkPen(color=(255, 255, 255), width=1)),
            'mod': pg.PlotCurveItem(
                name="fitted spectrum",
                pen=pg.mkPen(color=(255, 0, 0), width=1))
        }

        # config widgets
        self.hsImageConfig = QHSImageConfigWidget()
        # self.hsComponentFitConfig = QHSComponentFitConfigWidget(
        #     hsformat=HSAbsorption)
        # self.hsComponentFitConfig.setEnabled(False)

        # self.hsTivitaAnalysis = HSOpenTivita(hsformat=HSAbsorption)
        self.hsTivitaAnalysis = HSTivita(hsformat=HSIntensity)

        # set view
        self._setupViews(*args, **kwargs)

        logger.debug("HSImageTivitaAnalyzer initialized")

    def _setupViews(self, *args, **kwargs):
        # basic layout configuration
        self.mainLayout = QtWidgets.QHBoxLayout()
        self.mainLayout.setContentsMargins(5, 0, 5, 0)  # (l, t, r, b)
        self.mainLayout.setSpacing(0)
        self.setLayout(self.mainLayout)

        # configure image control items
        for key, item in self.imagCtrlItems.items():
            item.setMaximumWidth(440)
            item.setAspectLocked()
            item.invertY()
            item.setMaximumHeight(440)

        # configure spectral plot item
        for item in self.curveItems.values():
            self.spectViewer.addItem(item)

        # place graphics items
        self.graphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self.graphicsLayoutWidget.addItem(self.imagCtrlItems['rgb'], 0, 0)
        self.graphicsLayoutWidget.addItem(self.imagCtrlItems['nir'], 0, 1)
        self.graphicsLayoutWidget.addItem(self.imagCtrlItems['oxy'], 0, 2)
        self.graphicsLayoutWidget.addItem(self.imagCtrlItems['thi'], 1, 1)
        self.graphicsLayoutWidget.addItem(self.imagCtrlItems['twi'], 1, 2)
        # self.graphicsLayoutWidget.addItem(self.imagCtrlItems['mel'], 1, 2)
        self.graphicsLayoutWidget.addItem(self.spectViewer, 0, 3, rowspan=2)
        # qGraphicsGridLayout = self.graphicsLayoutWidget.ci.layout
        # qGraphicsGridLayout.setColumnStretchFactor(0, 1)
        # qGraphicsGridLayout.setColumnStretchFactor(1, 1)
        self.mainLayout.addWidget(self.graphicsLayoutWidget)

        # user config widgets
        self.hsImageConfig.setMaximumWidth(220)
        # self.hsComponentFitConfig.setMaximumWidth(220)
        self.hsImageConfig.setFormat(HSAbsorption)
        self.hsImageConfig.imageFilterTypeComboBox.setCurrentIndex(0)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.hsImageConfig)

        line = QtGui.QFrame()
        line.setFrameShape(QtGui.QFrame.HLine)
        line.setFrameShadow(QtGui.QFrame.Sunken)
        layout.addWidget(line)

        # layout.addWidget(self.hsComponentFitConfig)

        # line = QtGui.QFrame()
        # line.setFrameShape(QtGui.QFrame.HLine)
        # line.setFrameShadow(QtGui.QFrame.Sunken)
        # layout.addWidget(line)

        layout.addStretch()
        self.mainLayout.addLayout(layout)

        # connect signals

        # link image items
        firstItem = next(iter(self.imagCtrlItems.values()))
        for item in self.imagCtrlItems.values():
            item.setXYLink(firstItem)
            item.sigCursorPositionChanged.connect(self.updateCursorPosition)

        self.hsImageConfig.sigValueChanged.connect(self.setHSImage)
        # self.hsComponentFitConfig.sigValueChanged.connect(self.onComponentFitChanged)
        # self.spectViewer.sigRegionChanged.connect(self.onRegionChanged)
        self.spectViewer.sigRegionChangeFinished.connect(
            self.onRegionChangeFinished)

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

    # def onRegionChanged(self, item):
    #     reg = item.getRegion()
    #     self.hsComponentFitConfig.blockSignals(True)
    #     self.hsComponentFitConfig.wavRegionWidget.blockSignals(True)
    #     self.hsComponentFitConfig.set_roi(reg)
    #     self.hsComponentFitConfig.wavRegionWidget.blockSignals(False)
    #     self.hsComponentFitConfig.blockSignals(False)

    def onRegionChangeFinished(self, item):
        reg = item.getRegion()
        # self.hsComponentFitConfig.setROI(reg)

    def updateCursorPosition(self):
        sender = self.sender()
        if not isinstance(sender, BaseImagCtrlItem):
            raise TypeError("Unexpected type {}, was expecting {}"
                            .format(type(sender), BaseImagCtrlItem))

        x, y = sender.getCursorPos()
        for item in self.imagCtrlItems.values():  # link cursors
            if item is not sender:
                item.blockSignals(True)
                item.setCursorPos([x, y])
                item.blockSignals(False)

        self.updateSpectralView()

        logger.debug("Update cursor position. Sender: {}".format(sender))

    def setHSImage(self, hsImageConfig, newFile):

        # multidimensional arrays of spectra
        self.spectra = hsImageConfig.getSpectra(filter=False)  # unfiltered
        self.fspectra = hsImageConfig.getSpectra(filter=True)  # filtered
        self.mspectra = np.zeros(self.fspectra.shape)

        # wavelength array
        self.wavelen = hsImageConfig.getWavelen()

        # set masked rgb image
        image = hsImageConfig.getImage()
        mask = hsImageConfig.getMask()

        red = image[:, :, 0]
        green = image[:, :, 1]
        blue = image[:, :, 2]

        idx = np.nonzero(mask == 0)  # gray out region out of mask
        gray = 0.2989 * red[idx] + 0.5870 * green[idx] + 0.1140 * blue[idx]
        red[idx] = gray*0
        green[idx] = gray*0
        blue[idx] = gray*0
        self.imagCtrlItems['rgb'].setData(image)  # rgb image

        # forward hsformat of hyperspectral image to the vector analyzer
        hsformat = self.hsImageConfig.getFormat()
        # self.hsComponentFitConfig.setFormat(hsformat)
        # self.hsComponentFitConfig.setMask(mask)

        # self.hsTivitaAnalysis.set_data(
        #     self.spectra, self.wavelen, hsformat=hsformat)
        self.hsTivitaAnalysis.set_data(
            self.fspectra, self.wavelen, hsformat=hsformat)
        self.hsTivitaAnalysis.evaluate(mask=mask)

        if newFile:
            # update spectra and wavelength for analysis
            # self.hsComponentFitConfig.setData(self.fspectra, self.wavelen)

            # autorange image plots and cursor reset
            self.imagCtrlItems['rgb'].autoRange()
            self.imagCtrlItems['rgb'].setCursorPos((0, 0))

            # update wavelength region and bounds in the spectral viewer
            bounds = self.wavelen[[0, -1]]
            self.spectViewer.setBounds(bounds)
            self.spectViewer.setRegion(bounds)
        else:
            # update only spectra for analysis (keep wavelength)
            # self.hsComponentFitConfig.setData(self.fspectra)
            pass

        # data = hsImageConfig.value()

        # update index plots and spectral viewer
        param = self.hsTivitaAnalysis.get_solution(unpack=True)
        keys = ['nir', 'oxy', 'thi', 'twi']
        for key in keys:
            self.imagCtrlItems[key].setData(param[key])
        self.updateSpectralView()

    def onComponentFitChanged(self, analyzer, enableTest=False):
        if self.hsImageConfig.isEmpty():
            return

        param = self.hsTivitaAnalysis.get_solution(unpack=True)
        keys = ['nir', 'oxy', 'thi', 'twi']
        for key in keys:
            self.imagCtrlItems[key].setData(param[key])
            # self.imagCtrlItems[key].setLevels([0., 1.])

        # hsformat = self.hsImageConfig.getFormat()
        self.mspectra = analyzer.getSpectra()#hsformat=hsformat)

        # update spectral viewer
        reg = analyzer.getROI()
        self.spectViewer.blockSignals(True)
        self.spectViewer.setRegion(reg)
        self.spectViewer.blockSignals(False)
        self.updateSpectralView()

    def updateSpectralView(self):
        """Retrieve hyper spectral data at current cursor position
        """
        if self.spectra is None:
            return

        x, y = self.imagCtrlItems['rgb'].getCursorPos()
        col = int(x)
        row = int(y)
        nwav, nrows, ncols = self.spectra.shape  # row-major
        if col < 0 or col >= ncols or row < 0 or row >= nrows:
            raise ValueError("Position outside the image {}".format([x, y]))

        wavelen = self.wavelen[:]
        self.curveItems['raw'].setData(wavelen, self.spectra[:, row, col])
        self.curveItems['fil'].setData(wavelen, self.fspectra[:, row, col])
        self.curveItems['mod'].setData(wavelen, self.mspectra[:, row, col])

        # self.hsComponentFitConfig.setTestMask([row, col])


def main():
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python version: {}".format(sys.version))
    logger.info("PyQt bindings: {}".format(pg.Qt.QT_LIB))
    logger.info("PyQtGraph version: {}".format(pg.__version__))

    app = QtWidgets.QApplication([])

    win = QHSTivitaAnalyzerWidget()
    # win.setGeometry(300, 30, 1200, 500)
    win.setGeometry(40, 160, 1800, 800)
    win.setWindowTitle("TIVITA Index Analysis")
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
    #     pg.QtGui.QApplication.instance().exec_()


if __name__ == '__main__':
    logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()

