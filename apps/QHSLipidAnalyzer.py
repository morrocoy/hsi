# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 12:33:32 2022

@author: kpapke

This example demonstrates the Lipid index parameters analysis applied on
hyperspectral images. The spectrum is plotted at user defined coordinated.
Various filters may be applied on both, the image and spectral directions.
The RGB picture is derived from the hyperspectral data using an in-build RGB
filter is able to extract the different color channels.
"""
import sys
import logging

import numpy as np
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

import hsi

from hsi import HSAbsorption, HSIntensity

from hsi.gui import QHSImageConfigWidget
from hsi.gui import BaseImagCtrlItem
from hsi.gui import PosnImagCtrlItem
from hsi.gui import HistImagCtrlItem
from hsi.gui import RegnImagCtrlItem
from hsi.gui import RegnPlotCtrlItem

from hsi.analysis import HSLipids


from hsi.log import logmanager

logger = logmanager.getLogger(__name__)

PARAM_CONFIG = {
    'rgb': "RGB Image",
    'lipids_li0': "LPI Angle 900-915nm",  # Moussa's previous fat indices
    'lipids_li1': "LPI Ratio 925-960nm",  # Moussa's previous fat indices
    'lipids_li2': "LPI Ratio 875-925nm",  # Moussa's previous fat indices
    'lipids_li3': "LPI 2nd Derv. 925nm",  # Moussa's previous fat indices
    'lipids_li4': "LPI abs. Angle 900-920nm",  # Moussa's absolute fat index
    'lipids_li5': "LPI inv. Angle 900-920nm",  # Moussa's absolute water index
}

class QHSTivitaAnalyzerWidget(QtWidgets.QWidget):

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
            RegnImagCtrlItem("Image Control Item 1", cbarWidth=10),
            RegnImagCtrlItem("Image Control Item 2", cbarWidth=10),
            RegnImagCtrlItem("Image Control Item 3", cbarWidth=10),
            RegnImagCtrlItem("Image Control Item 4", cbarWidth=10),
            RegnImagCtrlItem("Image Control Item 5", cbarWidth=10),
            # RegnImagCtrlItem("Image Control Item 6", cbarWidth=10),
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
        # self.hsImageConfig = QHSImageConfigWidget()
        # self.hsComponentFitConfig = QHSComponentFitConfigWidget(
        #     hsformat=HSAbsorption)
        # self.hsComponentFitConfig.setEnabled(False)

        # self.hsTivitaAnalysis = HSOpenTivita(hsformat=HSAbsorption)
        self.hsLipidsAnalysis = HSLipids(hsformat=HSIntensity)

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
        for i in range(2):
            for j in range(3):
                self.graphicsLayoutWidget.addItem(
                    self.imagCtrlItems[i * 3 + j], i, j)

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

        layoutConfig = QtWidgets.QVBoxLayout()
        layoutConfig.addWidget(self.hsImageConfig)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        layoutConfig.addWidget(line)

        # output for roi paramters ............................................
        layoutROIParam1 = QtWidgets.QFormLayout()
        layoutROIParam1.setContentsMargins(5, 10, 5, 10)  # ltrb
        layoutROIParam1.setSpacing(3)

        labelROIParam = QtWidgets.QLabel("ROI Mean Values")
        labelROIParam.setStyleSheet(
            "border: 0px;"
            "font: bold;"
        )
        labelROIParam.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        labelROIParam.setMaximumWidth(220)
        layoutROIParam1.addRow(labelROIParam)
        layoutConfig.addLayout(layoutROIParam1)

        # self.textBoxROIParam = QtWidgets.QLineEdit()
        self.textBoxROIParam = QtWidgets.QTextEdit()
        # self.textBoxROIParam.setStyleSheet(
        #     "font: Monospace;"
        # )
        # self.textBoxROIParam.setFont(QtGui.QFont('Courier', 10))
        self.textBoxROIParam.setFont(QtGui.QFont('Monospace', 10))
        self.textBoxROIParam.setMaximumWidth(210)
        self.textBoxROIParam.setMinimumWidth(210)
        self.textBoxROIParam.setMinimumHeight(350)
        self.textBoxROIParam.setMaximumHeight(500)
        # layoutROIParam.addRow(self.textBoxROIParam)

        layoutROIParam2 = QtWidgets.QHBoxLayout()
        layoutROIParam2.setContentsMargins(5, 1, 3, 10)
        layoutROIParam2.addWidget(self.textBoxROIParam)
        # layoutROIParam2.setAlignment(QtCore.Qt.AlignCenter)
        layoutConfig.addLayout(layoutROIParam2)


        # line = QtWidgets.QFrame()
        # line.setFrameShape(QtWidgets.QFrame.HLine)
        # line.setFrameShadow(QtWidgets.QFrame.Sunken)
        # layout.addWidget(line)

        layoutConfig.addStretch()
        self.mainLayout.addLayout(layoutConfig)

        # connect signals

        # link image items
        for item in self.imagCtrlItems[0:]:
            item.setXYLink(self.imagCtrlItems[0])
            item.sigCursorPositionChanged.connect(self.updateCursorPosition)

        for item in self.imagCtrlItems[1:]:
            item.sigROIMaskChanged.connect(self.updateROIParams)
        self.hsImageConfig.sigValueChanged.connect(self.setHSImage)

        # self.hsComponentFitConfig.sigValueChanged.connect(self.onComponentFitChanged)
        # self.spectViewer.sigRegionChanged.connect(self.onRegionChanged)
        self.spectViewer.sigRegionChangeFinished.connect(
            self.onRegionChangeFinished)


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
        self.imagCtrlItems[0].setData({'rgb': image}, PARAM_CONFIG)  # rgb image
        self.imagCtrlItems[0].selectImage('rgb')  # rgb image

        # forward hsformat of hyperspectral image to the vector analyzer
        hsformat = self.hsImageConfig.getFormat()
        # self.hsComponentFitConfig.setFormat(hsformat)
        # self.hsComponentFitConfig.setMask(mask)

        # self.hsTivitaAnalysis.set_data(
        #     self.spectra, self.wavelen, hsformat=hsformat)
        self.hsLipidsAnalysis.set_data(
            self.fspectra, self.wavelen, hsformat=hsformat)
        self.hsLipidsAnalysis.evaluate(mask=mask)

        if newFile:
            # update spectra and wavelength for analysis
            # self.hsComponentFitConfig.setData(self.fspectra, self.wavelen)

            # autorange image plots and cursor reset
            self.imagCtrlItems[0].autoRange()
            self.imagCtrlItems[0].resetCursor()

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
        param = self.hsLipidsAnalysis.get_solution(unpack=True)
        keys = [key for key in PARAM_CONFIG.keys() if key in param.keys()]
        nkeys = len(keys)
        for i, item in enumerate(self.imagCtrlItems[1:]):
            item.setData(param, PARAM_CONFIG)
            item.selectImage(keys[i % nkeys])

        self.updateSpectralView()

    def onComponentFitChanged(self, analyzer, enableTest=False):
        if self.hsImageConfig.isEmpty():
            return

        param = self.hsLipidsAnalysis.get_solution(unpack=True)
        keys = ['li0', 'li1', 'li2', 'li3']
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

    def updateROIParams(self, pts, mask):

        sender = self.sender()
        for item in self.imagCtrlItems[1:]:
            if item != sender:
                item.blockSignals(True)
                item.setROIMask(pts)
                item.blockSignals(False)

        image_count = 1
        roi_param = {}
        param = self.hsLipidsAnalysis.get_solution(unpack=True)
        for key in param.keys():
            m = mask.reshape(-1)
            m_idx = np.ix_(range(image_count), m == 1)
            p = param[key].reshape(image_count, -1)
            roi_param[key] = np.mean(p[m_idx], axis=1)

        # print(roi_param)
        msg = "\n".join([
            "%-26s %5.3f" % (PARAM_CONFIG[key]+":", roi_param[key])
            for key in roi_param.keys()])
        self.textBoxROIParam.setText(msg)

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
    win.setWindowTitle("Lipid Index Analysis")
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

