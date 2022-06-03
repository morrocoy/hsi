# -*- coding: utf-8 -*-
"""
Created on Mon May 30 08:09:24 2022

@author: kpapke

This example demonstrates multiple analyses applied on hyperspectral images
including TIVITA index parameters, spectral fitting, and lipid index parameters.
The spectrum is plotted at user defined coordinated. Various filters may be
applied on both, the image and spectral directions. The RGB picture is derived
from the hyperspectral data using an in-build RGB filter is able to extract
the different color channels.
"""
import sys
import logging

import numpy as np
from pyqtgraph.Qt import QtWidgets, QtCore

import pyqtgraph as pg

import hsi

from hsi import HSAbsorption, HSIntensity

from hsi.gui import QHSImageConfigWidget
from hsi.gui import QHSCoFitConfigWidget

from hsi.gui import BaseImagCtrlItem
from hsi.gui import HistImagCtrlItem
from hsi.gui import PosnImagCtrlItem
from hsi.gui import RegnPlotCtrlItem
# from hsi.gui import ColorBarItem

# from hsi.analysis import HSOpenTivita
from hsi.analysis import HSTivita
from hsi.analysis import HSLipids
from hsi.analysis import HSCoFit

from hsi.log import logmanager

logger = logmanager.getLogger(__name__)

NFITS = 3

PARAM_CONFIG = {
    'im0': "RGB Image (original)",
    'im1': "RGB Image (with selection)",
    'tivita_oxy': "OXY (TIVITA)",
    'tivita_nir': "NIR (TIVITA)",
    'tivita_thi': "THI (TIVITA)",
    'tivita_twi': "TWI (TIVITA)",
    'lipids_li0': "LPI Angle 900-915nm",
    'lipids_li1': "LPI Ratio 925-960nm",
    'lipids_li2': "LPI Ratio 875-925nm",
    'lipids_li3': "LPI 2nd Derv. 925nm",
    'cofit_blo_0': "Blood (Fit 600-995nm)",
    'cofit_oxy_0': "OXY (Fit 600-995nm)  ",
    'cofit_wat_0': "Water (Fit 600-995nm)",
    'cofit_fat_0': "Fat (Fit 600-995nm)",
    'cofit_mel_0': "Melanin (Fit 600-995nm)",
    'cofit_hhb_0': "DeoxyHb (Fit 600-995nm)",
    'cofit_ohb_0': "OxyHb (Fit 600-995nm)",
    'cofit_met_0': "MetHb (Fit 600-995nm)",
    'cofit_blo_1': "Blood (Fit 520-600nm)",
    'cofit_oxy_1': "OXY (Fit 520-600nm)  ",
    'cofit_wat_1': "Water (Fit 520-600nm)",
    'cofit_fat_1': "Fat (Fit 520-600nm)",
    'cofit_mel_1': "Melanin (Fit 520-600nm)",
    'cofit_hhb_1': "DeoxyHb (Fit 520-600nm)",
    'cofit_ohb_1': "OxyHb (Fit 520-600nm)",
    # 'cofit_met_1': "MetHb (Fit 520-600nm)",
    'cofit_blo_2': "Blood (Fit 520-995nm)",
    'cofit_oxy_2': "OXY (Fit 520-995nm)  ",
    'cofit_wat_2': "Water (Fit 520-995nm)",
    'cofit_fat_2': "Fat (Fit 520-995nm)",
    'cofit_mel_2': "Melanin (Fit 520-995nm)",
    'cofit_hhb_2': "DeoxyHb (Fit 520-995nm)",
    'cofit_ohb_2': "OxyHb (Fit 520-995nm)",
    # 'cofit_met_2': "MetHb (Fit 520-995nm)",
}


class CurveViewItem(pg.GraphicsWidget):

    sigPlotChanged = QtCore.Signal(object)

    def __init__(self, *args, **kwargs):
        """
        =============== ========================================================
        **Arguments:**
        label           (str or None) label of
        xlabel, ylabel  (str or None) axes labels

        =============== ========================================================

        """

        pg.GraphicsObject.__init__(self, kwargs.get('parent', None))

        if len(args) == 0:
            kwargs['label'] = None
        elif len(args) == 1:
            kwargs['label'] = args[0]
        else:
            raise TypeError("Unexpected number of arguments {}".format(args))

        self._previousGeometry = None

        label = kwargs.get('label', None)

        self.curveItems = []
        self.shadowCurveItems = []

        self.plotItem1 = pg.PlotItem()
        # self.toolbarWidget = QRegnPlotCtrlConfigWidget(self.regionItem, label)
        # self.toolbarWidget =

        self._setupActions()
        self._setupViews(**kwargs)

    def _setupActions(self):
        pass

    def _setupViews(self, *args, **kwargs):

        if len(args) == 1:
            kwargs['xlabel'] = args[0]
        if len(args) == 2:
            kwargs['xlabel'] = args[0]
            kwargs['ylabel'] = args[1]

        xlabel = kwargs.get('xlabel', None)
        xunits = kwargs.get('xunits', None)
        ylabel = kwargs.get('ylabel', None)
        yunits = kwargs.get('yunits', None)


        if xlabel is not None:
            self.plotItem1.setLabel('bottom', xlabel, xunits)
        if ylabel is not None:
            self.plotItem1.setLabel('left', ylabel, yunits)

        # self.plotItem1.setLabel('bottom', "test")

        # self.toolbarProxy = QtWidgets.QGraphicsProxyWidget()
        # self.toolbarProxy.setWidget(self.toolbarWidget)

        self.mainLayout = QtWidgets.QGraphicsGridLayout()
        self.setLayout(self.mainLayout)
        # self.mainLayout.setContentsMargins(100, 30, 100, 30)
        # self.mainLayout.setSpacing(100)

        _spacer_toolbar = QtWidgets.QLabel()
        _spacer_toolbar.setMinimumHeight(15)
        _spacer_toolbar.setStyleSheet(
            "border-color: black;"
            "background-color: black;"
        )
        self.toolbarProxy = QtWidgets.QGraphicsProxyWidget()
        self.toolbarProxy.setWidget(_spacer_toolbar)

        # self.plotItem1.setMinimumHeight(280)
        self.plotItem1.setMaximumHeight(330)
        _spacer_statusbar = QtWidgets.QLabel()
        _spacer_statusbar.setMinimumHeight(70)
        # _spacer_statusbar.setMaximumHeight(100)
        _spacer_statusbar.setStyleSheet(
            "border-color: black;"
            "background-color: black;"
        )
        self.statusbarProxy = QtWidgets.QGraphicsProxyWidget()
        self.statusbarProxy.setWidget(_spacer_statusbar)

        # self.mainLayout.addItem(self.toolbarProxy, 0, 0)
        self.mainLayout.addItem(self.toolbarProxy, 0, 0)
        self.mainLayout.addItem(self.plotItem1, 1, 0)
        self.mainLayout.addItem(self.statusbarProxy, 2, 0)

        # self.mainLayout.setRowStretchFactor(1, 5)
        # self.mainLayout.setRowStretchFactor(2, 6)


    def addItem(self, item):

        if not isinstance(item, pg.PlotCurveItem):
            raise TypeError("Unexpected type {}, was expecting {}"
                            .format(type(item), pg.PlotCurveItem))

        x, y = item.getData()
        pen = item.opts['pen']

        shadowItem = pg.PlotCurveItem(x=x, y=y, pen=pen)

        self.plotItem1.addItem(item)
        # self.plotItem2.addItem(shadowItem)

        self.curveItems.append(item)
        self.shadowCurveItems.append(shadowItem)
        # self.updateBounds()

        item.sigPlotChanged.connect(self.plotChangedEvent)

    def plotChangedEvent(self, sender):
        for item, shadowItem in zip(self.curveItems, self.shadowCurveItems):
            if item is sender:
                x, y = item.getData()
                shadowItem.setData(x=x, y=y)

        self.updateBounds()
        self.sigPlotChanged.emit(sender)

    def setBounds(self, limits):
        pass

    def setRegion(self, limits):
        pass

    def updateBounds(self):
        pass



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
            HistImagCtrlItem("Image Control Item 1", cbarWidth=10),
            HistImagCtrlItem("Image Control Item 2", cbarWidth=10),
            HistImagCtrlItem("Image Control Item 3", cbarWidth=10),
            HistImagCtrlItem("Image Control Item 4", cbarWidth=10),
            HistImagCtrlItem("Image Control Item 5", cbarWidth=10),
            HistImagCtrlItem("Image Control Item 6", cbarWidth=10),
        ]

        # self.spectViewer = RegnPlotCtrlItem(
        #     "spectral attenuation", xlabel="wavelength", xunits="m")
        self.spectViewer = CurveViewItem(
            "spectral attenuation", xlabel="wavelength", xunits="m")

        self.spectPlotItem = pg.PlotItem()

        self.curveItems = {
            'crv0': pg.PlotCurveItem(
                name="raw spectral data",
                pen=pg.mkPen(color=(100, 100, 100), width=1)),
            'crv1': pg.PlotCurveItem(
                name="filtered spectrum",
                pen=pg.mkPen(color=(255, 255, 255), width=2)),
            'crv2': pg.PlotCurveItem(
                name="Fit 600-1000nm",
                pen=pg.mkPen(color=(255, 0, 0), width=1)),
            'crv3': pg.PlotCurveItem(
                name="Fit 500-600nm",
                pen=pg.mkPen(color=(0, 255, 0), width=1)),
            'crv4': pg.PlotCurveItem(
                name="Fit 500-1000nm",
                pen=pg.mkPen(color=(0, 255, 255), width=1))
        }

        # initiate image config widget
        import os.path
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data")
        self.hsImageConfig = QHSImageConfigWidget(dir=data_path)
        # self.hsImageConfig = QHSImageConfigWidget()

        # initiate Tivita analysis module
        # self.hsTivitaAnalysis = HSOpenTivita(hsformat=HSAbsorption)
        self.hsTivitaAnalysis = HSTivita(hsformat=HSIntensity)

        # initiate Moussa's lipid index analysis module
        self.hsLipidsAnalysis = HSLipids(hsformat=HSIntensity)

        # initiate component fit analysis module
        self.hsCoFitAnalysis = HSCoFit(hsformat=HSAbsorption)
        self.hsCoFitAnalysis.loadtxt("basevectors_2_17052022.txt", mode='all')
        self.hsCoFitAnalysis.set_var_bounds("hhb", [0, 0.1])
        self.hsCoFitAnalysis.set_var_bounds("ohb", [0, 0.1])
        self.hsCoFitAnalysis.set_var_bounds("wat", [0, 2.00])
        self.hsCoFitAnalysis.set_var_bounds("met", [0, 0.10])
        self.hsCoFitAnalysis.set_var_bounds("mel", [0, 0.20])

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
                    self.imagCtrlItems[i*3 + j], i, j)
                self.imagCtrlItems[i*3 + j].setMinimumHeight(380)

        # self.graphicsLayoutWidget2 = pg.GraphicsLayoutWidget()

        self.graphicsLayoutWidget.addItem(self.spectViewer, 0, 3)
        # self.graphicsLayoutWidget2.addItem(self.spectViewer, 0, 0, rowspan=1)
        # self.spectViewer.setMinimumWidth(400)
        self.spectViewer.setMinimumHeight(380)

        self.graphicsLayoutWidget.addItem(
            self.imagCtrlItems[6], 1, 3)
        self.imagCtrlItems[6].setMinimumHeight(380)

        # qGraphicsGridLayout = self.graphicsLayoutWidget.ci.layout
        # qGraphicsGridLayout.setColumnStretchFactor(0, 1)
        # qGraphicsGridLayout.setColumnStretchFactor(1, 1)

        self.mainLayout.addWidget(self.graphicsLayoutWidget)
        # self.mainLayout.addWidget(self.graphicsLayoutWidget2)

        # user config widgets
        self.hsImageConfig.setMaximumWidth(220)
        # self.hsComponentFitConfig.setMaximumWidth(220)
        self.hsImageConfig.setFormat(HSAbsorption)
        # self.hsImageConfig.imageFilterTypeComboBox.setCurrentIndex(0)
        self.hsImageConfig.imageFilterTypeComboBox.setCurrentIndex(1)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.hsImageConfig)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(line)

        # layout.addWidget(self.hsComponentFitConfig)

        # line = QtWidgets.QFrame()
        # line.setFrameShape(QtWidgets.QFrame.HLine)
        # line.setFrameShadow(QtWidgets.QFrame.Sunken)
        # layout.addWidget(line)

        layout.addStretch()
        self.mainLayout.addLayout(layout)

        # connect signals

        # link image items
        for item in self.imagCtrlItems[0:]:
            item.setXYLink(self.imagCtrlItems[0])
            item.sigCursorPositionChanged.connect(self.updateCursorPosition)

        self.hsImageConfig.sigValueChanged.connect(self.setHSImage)

        # self.spectViewer.sigRegionChanged.connect(self.onRegionChanged)
        # self.spectViewer.sigRegionChangeFinished.connect(
        #     self.onRegionChangeFinished)

    # def onRegionChanged(self, item):
    #     reg = item.getRegion()
    #     self.hsComponentFitConfig.blockSignals(True)
    #     self.hsComponentFitConfig.wavRegionWidget.blockSignals(True)
    #     self.hsComponentFitConfig.set_roi(reg)
    #     self.hsComponentFitConfig.wavRegionWidget.blockSignals(False)
    #     self.hsComponentFitConfig.blockSignals(False)

    # def onRegionChangeFinished(self, item):
    #     reg = item.getRegion()

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
        self.mspectra = np.zeros((NFITS, ) + self.fspectra.shape)

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
        red[idx] = gray * 0
        green[idx] = gray * 0
        blue[idx] = gray * 0

        self.imagCtrlItems[0].setData({
            'im0': image_0,
            'im1': image_1
        }, PARAM_CONFIG)  # rgb images
        self.imagCtrlItems[0].selectImage('im1')

        # forward hsformat of hyperspectral image to the vector analyzer
        hsformat = self.hsImageConfig.getFormat()
        # self.hsComponentFitConfig.setFormat(hsformat)
        # self.hsComponentFitConfig.setMask(mask)

        self.hsTivitaAnalysis.set_data(
            self.spectra, self.wavelen, hsformat=hsformat)
        self.hsTivitaAnalysis.evaluate(mask=mask)

        self.hsLipidsAnalysis.set_data(
            self.spectra, self.wavelen, hsformat=hsformat)
        self.hsLipidsAnalysis.evaluate(mask=mask)

        self.hsCoFitAnalysis.set_data(
            self.fspectra, self.wavelen, hsformat=hsformat)
        self.hsCoFitAnalysis.prepare_ls_problem()
        self.hsCoFitAnalysis.freeze_component("met")
        self.hsCoFitAnalysis.set_roi([520e-9, 995e-9])
        self.hsCoFitAnalysis.fit(method='bvls_f', mask=mask)
        self.hsCoFitAnalysis.set_roi([520e-9, 600e-9])
        self.hsCoFitAnalysis.fit(method='bvls_f', mask=mask)
        self.hsCoFitAnalysis.unfreeze_component("met")
        self.hsCoFitAnalysis.set_roi([600e-9, 995e-9])
        self.hsCoFitAnalysis.fit(method='bvls_f', mask=mask)

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
        param = {}
        param.update(self.hsTivitaAnalysis.get_solution(unpack=True))
        param.update(self.hsLipidsAnalysis.get_solution(unpack=True))
        param.update(
            self.hsCoFitAnalysis.get_solution(which="all", unpack=True))

        # additional parameter combinations
        prefix = self.hsCoFitAnalysis.prefix
        for i in range(2):
            param["%sblo_%d" % (prefix, i)] = \
                param["%shhb_%d" % (prefix, i)] + param[
                    "%sohb_%d" % (prefix, i)]
            param["%soxy_%d" % (prefix, i)] = np.zeros(
                param["%sblo_%d" % (prefix, i)].shape)
            idx = np.nonzero(param["%sblo_%d" % (prefix, i)])
            param["%soxy_%d" % (prefix, i)][idx] = \
                param["%sohb_%d" % (prefix, i)][idx] / param[
                    "%sblo_%d" % (prefix, i)][idx]

        keys = [key for key in PARAM_CONFIG.keys() if key in param.keys()]
        nkeys = len(keys)
        for i, item in enumerate(self.imagCtrlItems[1:]):
            item.setData(param, PARAM_CONFIG)
            item.selectImage(keys[i % nkeys])

        self.mspectra = self.hsCoFitAnalysis.model(which="all")
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
        self.curveItems['crv0'].setData(wavelen, self.spectra[:, row, col])
        self.curveItems['crv1'].setData(wavelen, self.fspectra[:, row, col])

        for i in range(NFITS):
            self.curveItems['crv%d' % (2+i)].setData(
                wavelen, self.mspectra[i, :, row, col])

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
    #     pg.QtWidgets.QApplication.instance().exec_()


if __name__ == '__main__':
    logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()

