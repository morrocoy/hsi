# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:25:18 2021

@author: kpapke

This example demonstrates component fitting applied on hyperspectral images.
The spectrum is plotted at user defined coordinated. Various filters may be
applied on both, the image and spectral directions. The RGB picture is derived
from the hyperspectral data using an in-build RGB filter is able to extract
the different color channels.
"""
import sys
import logging

import numpy as np
from pyqtgraph.Qt import QtWidgets
import pyqtgraph as pg

import hsi

from hsi import HSAbsorption

from hsi.gui import QHSImageConfigWidget
from hsi.gui import QHSCoFitConfigWidget

from hsi.gui import BaseImagCtrlItem
from hsi.gui import HistImagCtrlItem
from hsi.gui import PosnImagCtrlItem
from hsi.gui import RegnPlotCtrlItem

from hsi.log import logmanager

logger = logmanager.getLogger(__name__)

PARAM_CONFIG = {
    'im0': "RGB Image (original)",
    'im1': "RGB Image (with selection)",
    'cofit_blo': "Blood",
    'cofit_oxy': "Oxygenation",
    'cofit_wat': "Water",
    'cofit_fat': "Fat",
    'cofit_mel': "Melanin",
    'cofit_hhb': "Deoxyhemoglobin",
    'cofit_ohb': "Oxyhemoglobin",
    'cofit_met': "Methemoglobin",
}


class QHSCoFitAnalyzerWidget(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        QtWidgets.QWidget.__init__(self)

        # internal buffers
        self.spectra = None  # multidimensional array of unfiltered hs-data
        self.fspectra = None  # multidimensional array of filtered hs-data
        self.mspectra = None  # multidimensional array of modeled hs-data
        self.wavelen = None  # wavelength axis

        # image, 2D histogram and spectral attenuation plots
        self.imagCtrlItems = [
            PosnImagCtrlItem("Image Control Item 0", cbarWidth=10),
            HistImagCtrlItem("Image Control Item 1", cbarWidth=10),
            HistImagCtrlItem("Image Control Item 2", cbarWidth=10),
            HistImagCtrlItem("Image Control Item 3", cbarWidth=10),
            HistImagCtrlItem("Image Control Item 4", cbarWidth=10),
            HistImagCtrlItem("Image Control Item 5", cbarWidth=10),
        ]

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
        if not getattr(sys, 'frozen', False):
            import os.path
            data_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "data")
            self.hsImageConfig = QHSImageConfigWidget(dir=data_path)
        else:
            self.hsImageConfig = QHSImageConfigWidget()
        self.hsCoFitConfig = QHSCoFitConfigWidget(
            hsformat=HSAbsorption)

        # set view
        self._setupViews(*args, **kwargs)

        logger.debug("HSImageComponentAnalyzer initialized")

    def _setupViews(self, *args, **kwargs):
        # basic layout configuration
        self.mainLayout = QtWidgets.QHBoxLayout()
        self.mainLayout.setContentsMargins(5, 0, 5, 0)  # (l, t, r, b)
        self.mainLayout.setSpacing(0)
        self.setLayout(self.mainLayout)

        # configure image control items
        for item in self.imagCtrlItems:
            item.setMaximumWidth(440)
            item.setAspectLocked()
            item.invertY()
            item.setMaximumHeight(440)

        # configure spectral plot item
        for item in self.curveItems.values():
            self.spectViewer.addItem(item)

        # place graphics items
        self.graphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self.graphicsLayoutWidget.ci.layout.setHorizontalSpacing(5)
        self.graphicsLayoutWidget.ci.layout.setVerticalSpacing(20)
        for i in range(2):
            for j in range(3):
                self.graphicsLayoutWidget.addItem(
                    self.imagCtrlItems[i*3 + j], i, j)

        # self.graphicsLayoutWidget.addItem(self.imagCtrlItems[0], 0, 0)
        # self.graphicsLayoutWidget.addItem(self.imagCtrlItems[1], 0, 1)
        # self.graphicsLayoutWidget.addItem(self.imagCtrlItems[2], 0, 2)
        # self.graphicsLayoutWidget.addItem(self.imagCtrlItems[3], 1, 0)
        # self.graphicsLayoutWidget.addItem(self.imagCtrlItems[4], 1, 1)
        # self.graphicsLayoutWidget.addItem(self.imagCtrlItems[5], 1, 2)

        self.graphicsLayoutWidget.addItem(self.spectViewer, 0, 3, rowspan=2)
        # qGraphicsGridLayout = self.graphicsLayoutWidget.ci.layout
        # qGraphicsGridLayout.setColumnStretchFactor(0, 1)
        # qGraphicsGridLayout.setColumnStretchFactor(1, 1)
        self.mainLayout.addWidget(self.graphicsLayoutWidget)

        # user config widgets
        self.hsImageConfig.setMaximumWidth(220)
        self.hsImageConfig.setFormat(HSAbsorption)
        self.hsCoFitConfig.setMaximumWidth(220)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.hsImageConfig)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(line)

        layout.addWidget(self.hsCoFitConfig)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(line)

        layout.addStretch()
        self.mainLayout.addLayout(layout)

        # connect signals

        # link image items
        for item in self.imagCtrlItems[0:]:
            item.setXYLink(self.imagCtrlItems[0])
            item.sigCursorPositionChanged.connect(self.updateCursorPosition)

        self.hsImageConfig.sigValueChanged.connect(self.setHSImage)
        self.hsCoFitConfig.sigValueChanged.connect(
            self.onComponentFitChanged)
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
        self.hsCoFitConfig.setROI(reg)

    def updateCursorPosition(self):
        sender = self.sender()
        if not isinstance(sender, BaseImagCtrlItem):
            raise TypeError("Unexpected type {}, was expecting {}"
                            .format(type(sender), BaseImagCtrlItem))

        x, y = sender.getCursorPos()
        for item in self.imagCtrlItems:  # link cursors
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
        image_0 = hsImageConfig.getImage()
        mask = hsImageConfig.getMask()

        image_1 = image_0.copy()
        red = image_1[:, :, 0]
        green = image_1[:, :, 1]
        blue = image_1[:, :, 2]

        idx = np.nonzero(mask == 0)  # gray out region out of mask
        gray = 0.2989 * red[idx] + 0.5870 * green[idx] + 0.1140 * blue[idx]
        red[idx] = gray*0
        green[idx] = gray*0
        blue[idx] = gray*0

        self.imagCtrlItems[0].setData({
            'im0': image_0,
            'im1': image_1
        }, PARAM_CONFIG)  # rgb images
        self.imagCtrlItems[0].selectImage('im1')

        # forward hsformat of hyperspectral image to the component analyzer
        hsformat = self.hsImageConfig.getFormat()
        self.hsCoFitConfig.setFormat(hsformat)
        self.hsCoFitConfig.setMask(mask)

        if newFile:
            # update spectra and wavelength for analysis
            self.hsCoFitConfig.setData(self.fspectra, self.wavelen)

            # autorange image plots and cursor reset
            self.imagCtrlItems[0].autoRange()
            self.imagCtrlItems[0].resetCursor()

            # update wavelength region and bounds in the spectral viewer
            bounds = self.wavelen[[0, -1]]
            self.spectViewer.setBounds(bounds)
            self.spectViewer.setRegion(bounds)

        else:
            # update only spectra for analysis (keep wavelength)
            self.hsCoFitConfig.setData(self.fspectra)

    def onComponentFitChanged(self, analyzer, enableTest=False):
        if self.hsImageConfig.isEmpty():
            return

        # update image plots
        if not enableTest:
            prefix = "cofit_"
            param = analyzer.getSolution()
            param[prefix + 'blo'] = param[prefix + 'hhb'] + param[prefix + 'ohb']
            param[prefix + 'oxy'] = np.zeros(param[prefix + 'blo'].shape)
            idx = np.nonzero(param[prefix + 'blo'])
            param[prefix + 'oxy'][idx] = param[prefix + 'ohb'][idx] / param[prefix + 'blo'][idx]

            keys = [key for key in PARAM_CONFIG.keys() if key in param.keys()]
            nkeys = len(keys)
            for i, item in enumerate(self.imagCtrlItems[1:]):
                item.setData(param, PARAM_CONFIG)
                item.selectImage(keys[i % nkeys])

        # hsformat = self.hsImageConfig.getFormat()
        self.mspectra = analyzer.getSpectra()  # hsformat=hsformat)

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

        x, y = self.imagCtrlItems[0].getCursorPos()
        col = int(x)
        row = int(y)
        nwav, nrows, ncols = self.spectra.shape  # row-major
        if col < 0 or col >= ncols or row < 0 or row >= nrows:
            raise ValueError("Position outside the image {}".format([x, y]))

        wavelen = self.wavelen[:]
        self.curveItems['raw'].setData(wavelen, self.spectra[:, row, col])
        self.curveItems['fil'].setData(wavelen, self.fspectra[:, row, col])
        self.curveItems['mod'].setData(wavelen, self.mspectra[:, row, col])

        self.hsCoFitConfig.setTestMask([row, col])


def main():
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python version: {}".format(sys.version))
    logger.info("PyQt bindings: {}".format(pg.Qt.QT_LIB))
    logger.info("PyQtGraph version: {}".format(pg.__version__))

    app = QtWidgets.QApplication([])

    win = QHSCoFitAnalyzerWidget()
    # win.setGeometry(300, 30, 1200, 500)
    # win.setGeometry(290, 30, 1800, 800)
    # win.setGeometry(20, 30, 1900, 920)
    win.setGeometry(20, 30, 1800, 800)
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
