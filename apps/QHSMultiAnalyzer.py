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
import json

import numpy as np
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

import pyqtgraph as pg

import hsi

from hsi import HSAbsorption, HSExtinction, HSIntensity, convert

from hsi.gui import QHSImageConfigWidget
from hsi.gui import QHSCoFitConfigWidget

from hsi.gui import BaseImagCtrlItem
from hsi.gui import PosnImagCtrlItem
from hsi.gui import HistImagCtrlItem
from hsi.gui import RegnImagCtrlItem
from hsi.gui import RegnPlotCtrlItem
# from hsi.gui import ColorBarItem

# from hsi.analysis import HSOpenTivita
from hsi.analysis import HSTivita
from hsi.analysis import HSLipids
from hsi.analysis import HSOxygen
from hsi.analysis import HSCoFit

from hsi.log import logmanager

logger = logmanager.getLogger(__name__)

NFITS = 3

MAINZ= True

# PARAM_CONFIG = {
#     'im0': "RGB Image (original)",
#     'im1': "RGB Image (with selection)",
#     'tivita_oxy': "OXY (TIVITA)",
#     'tivita_nir': "NIR (TIVITA)",
#     'tivita_thi': "THI (TIVITA)",
#     'tivita_twi': "TWI (TIVITA)",
#     'oxygen_ox0': "OXY Angle 630-710nm",  # Moussa's oxygen index
#     'lipids_li0': "LPI Angle 900-915nm",  # Moussa's previous fat indices
#     'lipids_li1': "LPI Ratio 925-960nm",  # Moussa's previous fat indices
#     'lipids_li2': "LPI Ratio 875-925nm",  # Moussa's previous fat indices
#     'lipids_li3': "LPI 2nd Derv. 925nm",  # Moussa's previous fat indices
#     'lipids_li4': "LPI abs. Angle 900-920nm",  # Moussa's absolute fat index
#     'lipids_li5': "LPI inv. Angle 900-920nm",  # Moussa's absolute water index
#     'cofit_blo_0': "Blood (Fit 600-995nm)",
#     'cofit_oxy_0': "OXY (Fit 600-995nm)",
#     'cofit_wat_0': "Water (Fit 600-995nm)",
#     'cofit_wob_0': "Wat/Blo (Fit 600-995nm)",
#     # 'cofit_fat_0': "Fat (Fit 600-995nm)",
#     'cofit_mel_0': "Melanin (Fit 600-995nm)",
#     'cofit_hhb_0': "DeoxyHb (Fit 600-995nm)",
#     'cofit_ohb_0': "OxyHb (Fit 600-995nm)",
#     'cofit_met_0': "MetHb (Fit 600-995nm)",
#     'cofit_blo_1': "Blood (Fit 520-600nm)",
#     'cofit_oxy_1': "OXY (Fit 520-600nm)  ",
#     'cofit_wat_1': "Water (Fit 520-600nm)",
#     # 'cofit_wob_1': "Wat/Blo (Fit 520-600nm)",
#     # 'cofit_fat_1': "Fat (Fit 520-600nm)",
#     'cofit_mel_1': "Melanin (Fit 520-600nm)",
#     'cofit_hhb_1': "DeoxyHb (Fit 520-600nm)",
#     'cofit_ohb_1': "OxyHb (Fit 520-600nm)",
#     # 'cofit_met_1': "MetHb (Fit 520-600nm)",
#     'cofit_blo_2': "Blood (Fit 520-995nm)",
#     'cofit_oxy_2': "OXY (Fit 520-995nm)  ",
#     'cofit_wat_2': "Water (Fit 520-995nm)",
#     # 'cofit_wob_2': "Wat/Blo (Fit 520-995nm)",
#     # 'cofit_fat_2': "Fat (Fit 520-995nm)",
#     'cofit_mel_2': "Melanin (Fit 520-995nm)",
#     'cofit_hhb_2': "DeoxyHb (Fit 520-995nm)",
#     'cofit_ohb_2': "OxyHb (Fit 520-995nm)",
#     # 'cofit_met_2': "MetHb (Fit 520-995nm)",
# }

PARAM_CONFIG = {
    'im0': "RGB Image (original)",
    'im1': "RGB Image (with selection)",
    'tivita_oxy': "OXY (TIVITA)",
    'oxygen_ox0': "OXY Angle 630-710nm",  # Moussa's oxygen index
    'cofit_oxy_0': "OXY (Fit 600-995nm)",
    'cofit_oxy_1': "OXY (Fit 520-600nm)",
    'cofit_oxy_2': "OXY (Fit 520-995nm)",
    'tivita_nir': "NIR (TIVITA)",
    'tivita_thi': "THI (TIVITA)",
    'cofit_blo_0': "Blood (Fit 600-995nm)",
    'cofit_blo_1': "Blood (Fit 520-600nm)",
    'cofit_blo_2': "Blood (Fit 520-995nm)",
    'cofit_hhb_0': "DeoxyHb (Fit 600-995nm)",
    'cofit_hhb_1': "DeoxyHb (Fit 520-600nm)",
    'cofit_hhb_2': "DeoxyHb (Fit 520-995nm)",
    'cofit_ohb_0': "OxyHb (Fit 600-995nm)",
    'cofit_ohb_1': "OxyHb (Fit 520-600nm)",
    'cofit_ohb_2': "OxyHb (Fit 520-995nm)",
    'tivita_twi': "TWI (TIVITA)",
    'cofit_wat_0': "Water (Fit 600-995nm)",
    'cofit_wat_1': "Water (Fit 520-600nm)",
    'cofit_wat_2': "Water (Fit 520-995nm)",
    'cofit_wob_0': "Wat/Blo (Fit 600-995nm)",
    # 'cofit_wob_1': "Wat/Blo (Fit 520-600nm)",
    # 'cofit_wob_2': "Wat/Blo (Fit 520-995nm)",
    # 'cofit_fat_0': "Fat (Fit 600-995nm)",
    # 'cofit_fat_1': "Fat (Fit 520-600nm)",
    # 'cofit_fat_2': "Fat (Fit 520-995nm)",
    'lipids_li0': "LPI Angle 900-915nm",  # Moussa's previous fat indices
    'lipids_li1': "LPI Ratio 925-960nm",  # Moussa's previous fat indices
    'lipids_li2': "LPI Ratio 875-925nm",  # Moussa's previous fat indices
    'lipids_li3': "LPI 2nd Derv. 925nm",  # Moussa's previous fat indices
    'lipids_li4': "LPI abs. Angle 900-920nm",  # Moussa's absolute fat index
    'lipids_li5': "LPI inv. Angle 900-920nm",  # Moussa's absolute water index
    'cofit_mel_0': "Melanin (Fit 600-995nm)",
    # 'cofit_mel_1': "Melanin (Fit 520-600nm)",
    'cofit_mel_2': "Melanin (Fit 520-995nm)",
    'cofit_met_0': "MetHb (Fit 600-995nm)",
    # 'cofit_met_1': "MetHb (Fit 520-600nm)",
    # 'cofit_met_2': "MetHb (Fit 520-995nm)",
}

if MAINZ:

    PARAM_CONFIG = {
        'im0': "RGB Image (original)",
        'im1': "RGB Image (with selection)",
        'tivita_oxy': "OXY (TIVITA)",
        'tivita_nir': "NIR (TIVITA)",
        'tivita_thi': "THI (TIVITA)",
        'tivita_twi': "TWI (TIVITA)",
        # 'oxygen_ox0': "OXY Angle 630-710nm",  # Moussa's oxygen index
        'cofit_oxy_0': "OXY (Fit 600-995nm)",
        'cofit_oxy_1': "OXY (Fit 520-600nm)",
        # 'cofit_oxy_2': "OXY (Fit 520-995nm)",
        # 'cofit_blo_0': "Blood (Fit 600-995nm)",
        # 'cofit_blo_1': "Blood (Fit 520-600nm)",
        'cofit_blo_2': "Blood (Fit 520-995nm)",
        # 'cofit_hhb_0': "DeoxyHb (Fit 600-995nm)",
        # 'cofit_hhb_1': "DeoxyHb (Fit 520-600nm)",
        # 'cofit_hhb_2': "DeoxyHb (Fit 520-995nm)",
        # 'cofit_ohb_0': "OxyHb (Fit 600-995nm)",
        # 'cofit_ohb_1': "OxyHb (Fit 520-600nm)",
        # 'cofit_ohb_2': "OxyHb (Fit 520-995nm)",
        'cofit_wat_0': "Water (Fit 600-995nm)",
        # 'cofit_wat_1': "Water (Fit 520-600nm)",
        # 'cofit_wat_2': "Water (Fit 520-995nm)",
        'cofit_wob_0': "Wat/Blo (Fit 600-995nm)",
        # 'cofit_wob_1': "Wat/Blo (Fit 520-600nm)",
        # 'cofit_wob_2': "Wat/Blo (Fit 520-995nm)",
        # 'cofit_fat_0': "Fat (Fit 600-995nm)",
        # 'cofit_fat_1': "Fat (Fit 520-600nm)",
        # 'cofit_fat_2': "Fat (Fit 520-995nm)",
        # 'lipids_li0': "LPI Angle 900-915nm",  # Moussa's previous fat indices
        # 'lipids_li1': "LPI Ratio 925-960nm",  # Moussa's previous fat indices
        # 'lipids_li2': "LPI Ratio 875-925nm",  # Moussa's previous fat indices
        'lipids_li3': "LPI 2nd Derv. 925nm",  # Moussa's previous fat indices
        # 'lipids_li4': "LPI abs. Angle 900-920nm",  # Moussa's absolute fat index
        # 'lipids_li5': "LPI inv. Angle 900-920nm",  # Moussa's absolute water index
        # 'cofit_mel_0': "Melanin (Fit 600-995nm)",
        # 'cofit_mel_1': "Melanin (Fit 520-600nm)",
        'cofit_mel_2': "Melanin (Fit 520-995nm)",
        'cofit_met_0': "MetHb (Fit 600-995nm)",
        # 'cofit_met_1': "MetHb (Fit 520-600nm)",
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
        self.plotItem1.addLegend(offset=(0, -100))
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

        self.plotItem1.addItem(item, name="Hallo")
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


class QHSROIParamWidget(QtWidgets.QWidget):
    """ Config widget for hyper spectral images
    """
    sigValueChanged = QtCore.Signal(object, str)

    def __init__(self, *args, **kwargs):
        """ Constructor
        """
        parent = kwargs.get('parent', None)
        super(QHSROIParamWidget, self).__init__(parent=parent)

        if len(args) == 1:
            kwargs['dir'] = args[0]
        elif len(args) > 1:
            raise TypeError("To many arguments {}".format(args))

        self.dir = kwargs.get('dir', None)
        self.filePath = None

        self.clearButton = QtWidgets.QToolButton(self)
        self.exportButton = QtWidgets.QToolButton(self)
        self.dataTextEdit = QtWidgets.QTextEdit(self)

        # configure actions
        self._setupActions()

        # configure widget views
        self._setupViews(*args, **kwargs)

    def _setupActions(self):
        self.clearAction = QtWidgets.QAction(self)
        self.clearAction.setIconText("Clear")
        self.clearAction.triggered.connect(self.clearData)
        self.addAction(self.clearAction)

        self.exportAction = QtWidgets.QAction(self)
        self.exportAction.setIconText("Export")
        self.exportAction.triggered.connect(self.exportData)
        self.addAction(self.exportAction)

        self.clearButton.setDefaultAction(self.clearAction)
        self.exportButton.setDefaultAction(self.exportAction)

    def _setupViews(self, *args, **kwargs):
        # self.mainLayout = QtWidgets.QVBoxLayout()
        self.mainLayout = QtWidgets.QFormLayout()
        self.mainLayout.setContentsMargins(5, 10, 5, 10) # left, top, right, bottom
        self.mainLayout.setSpacing(3)
        self.setLayout(self.mainLayout)

        # file load ..........................................................
        label = QtWidgets.QLabel("ROI Mean Values")
        label.setStyleSheet(
            "border: 0px;"
            "font: bold;"
        )
        label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.mainLayout.addRow(label)


        # self.dataTextEdit.setFont(QtGui.QFont('Courier', 7))
        if sys.platform == "win32":
            self.dataTextEdit.setFont(QtGui.QFont('Monospace', 8))
        else:
            self.dataTextEdit.setFont(QtGui.QFont('Monospace', 7))
        self.dataTextEdit.setMinimumHeight(260)
        self.dataTextEdit.setMaximumHeight(600)
        self.mainLayout.addRow(self.dataTextEdit)

        # self.fileLineEdit.setReadOnly(True)
        # layout = QtWidgets.QHBoxLayout()
        # layout.addWidget(self.fileLineEdit)
        # layout.addWidget(self.exportButton)
        # self.mainLayout.addRow(layout)

        # filter and reset controls
        label = QtWidgets.QLabel(self)
        label.setMinimumHeight(20)
        label.setStyleSheet("border: 0px;")
        self.clearButton.setText("Clear")
        self.exportButton.setText("Export")

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.clearButton)
        layout.addWidget(self.exportButton)
        self.mainLayout.addRow(layout)

    def clear(self):
        self.dataTextEdit.setText("")

    def finalize(self):
        """ Should be called manually before object deletion
        """
        logger.debug("Finalizing: {}".format(self))
        super(QHSROIParamWidget, self).finalize()

    def clearData(self):
        """Load hyper spectral image file using a dialog box
        """
        logger.debug("Clear ROI Parameter")
        self.dataTextEdit.setText("")
        self.sigValueChanged.emit(self, "")

    def exportData(self):
        """Export content of textbox
        """
        filePath, filter = QtWidgets.QFileDialog.getSaveFileName(
            None, 'Select file:', self.dir)

        logger.debug("Export ROI Parameter to file: {}".format(filePath))

        self.filePath = filePath
        str = self.dataTextEdit.toPlainText()
        with open(filePath, "w") as file:
            file.write(str)

    def setText(self, str):
        self.dataTextEdit.setText(str)
        self.sigValueChanged.emit(self, str)


class QHSMultiAnalyzerViewConfigWidget(QtWidgets.QWidget):
    """ Config widget for view selection
    """
    sigCurrentParamViewChanged = QtCore.Signal(list)

    # 'im0': "RGB Image (original)",
    # 'im1': "RGB Image (with selection)",
    # 'tivita_oxy': "OXY (TIVITA)",
    # 'tivita_nir': "NIR (TIVITA)",
    # 'tivita_thi': "THI (TIVITA)",
    # 'tivita_twi': "TWI (TIVITA)",
    # 'oxygen_ox0': "OXY Angle 630-710nm",  # Moussa's oxygen index
    # 'lipids_li0': "LPI Angle 900-915nm",  # Moussa's previous fat indices
    # 'lipids_li1': "LPI Ratio 925-960nm",  # Moussa's previous fat indices
    # 'lipids_li2': "LPI Ratio 875-925nm",  # Moussa's previous fat indices
    # 'lipids_li3': "LPI 2nd Derv. 925nm",  # Moussa's previous fat indices
    # 'lipids_li4': "LPI abs. Angle 900-920nm",  # Moussa's absolute fat index
    # 'lipids_li5': "LPI inv. Angle 900-920nm",  # Moussa's absolute water index
    # 'cofit_blo_0': "Blood (Fit 600-995nm)",
    # 'cofit_oxy_0': "OXY (Fit 600-995nm)",
    # 'cofit_wat_0': "Water (Fit 600-995nm)",
    # 'cofit_wob_0': "Wat/Blo (Fit 600-995nm)",
    # # 'cofit_fat_0': "Fat (Fit 600-995nm)",
    # 'cofit_mel_0': "Melanin (Fit 600-995nm)",
    # 'cofit_hhb_0': "DeoxyHb (Fit 600-995nm)",
    # 'cofit_ohb_0': "OxyHb (Fit 600-995nm)",
    # 'cofit_met_0': "MetHb (Fit 600-995nm)",
    # 'cofit_blo_1': "Blood (Fit 520-600nm)",
    # 'cofit_oxy_1': "OXY (Fit 520-600nm)  ",
    # 'cofit_wat_1': "Water (Fit 520-600nm)",
    # # 'cofit_wob_1': "Wat/Blo (Fit 520-600nm)",
    # # 'cofit_fat_1': "Fat (Fit 520-600nm)",
    # 'cofit_mel_1': "Melanin (Fit 520-600nm)",
    # 'cofit_hhb_1': "DeoxyHb (Fit 520-600nm)",
    # 'cofit_ohb_1': "OxyHb (Fit 520-600nm)",
    # # 'cofit_met_1': "MetHb (Fit 520-600nm)",
    # 'cofit_blo_2': "Blood (Fit 520-995nm)",
    # 'cofit_oxy_2': "OXY (Fit 520-995nm)  ",
    # 'cofit_wat_2': "Water (Fit 520-995nm)",
    # # 'cofit_wob_2': "Wat/Blo (Fit 520-995nm)",
    # # 'cofit_fat_2': "Fat (Fit 520-995nm)",
    # 'cofit_mel_2': "Melanin (Fit 520-995nm)",
    # 'cofit_hhb_2': "DeoxyHb (Fit 520-995nm)",
    # 'cofit_ohb_2': "OxyHb (Fit 520-995nm)",
    # # 'cofit_met_2': "MetHb (Fit 520-995nm)",

    views = {
        "TIVITA": [
            'im1',  # "RGB Image (with selection)"
            'tivita_oxy',  # "OXY (TIVITA)"
            'tivita_nir',  # "NIR (TIVITA)"
            'oxygen_ox0',  # Moussa's oxygen index
            'tivita_thi',  # "THI (TIVITA)"
            'tivita_twi',  # "TWI (TIVITA)"
            'cofit_oxy_1', # "OXY (Fit 520-600nm)"
        ],
        "Oxygenation": [
            'im1',  # "RGB Image (with selection)"
            'tivita_oxy',  # "OXY (TIVITA)"
            'oxygen_ox0',  # Moussa's oxygen index
            'cofit_oxy_1',  # "OXY (Fit 520-600nm)"
            'cofit_ohb_1',  # "OxyHb (Fit 520-600nm)",
            'cofit_oxy_2',  #"OXY (Fit 520-995nm)",
            'cofit_ohb_2',  # "OxyHb (Fit 520-995nm)",
        ],
        "Lipids": [
            'im1',  # "RGB Image (with selection)"
            'lipids_li0',  # "LPI Angle 900-915nm"
            'lipids_li1',  # "LPI Ratio 925-960nm"
            'lipids_li2',  # "LPI Ratio 875-925nm"
            'lipids_li3',  # "LPI 2nd Derv. 925nm"
            'lipids_li4',  # "LPI abs. Angle 900-920nm"
            'lipids_li5',  # "LPI inv. Angle 900-920nm"
        ],
        "User defined": [
            'im1',  # "RGB Image (with selection)"
            'tivita_oxy',  # "OXY (TIVITA)"
            'tivita_nir',  # "NIR (TIVITA)"
            'oxygen_ox0',  # Moussa's oxygen index
            'tivita_thi',  # "THI (TIVITA)"
            'tivita_twi',  # "TWI (TIVITA)"
            'cofit_oxy_1',  # "OXY (Fit 520-600nm)"
        ],
    }

    if MAINZ:
        views = {
            "TIVITA": [
                'im1',  # "RGB Image (with selection)"
                'tivita_oxy',  # "OXY (TIVITA)"
                'tivita_nir',  # "NIR (TIVITA)"
                'tivita_thi',  # "THI (TIVITA)"
                'tivita_thi',  # "THI (TIVITA)"
                'tivita_twi',  # "TWI (TIVITA)"
                'tivita_twi',  # "TWI (TIVITA)"
            ],
            "HSWismar": [
                'im1',  # "RGB Image (with selection)"
                'cofit_oxy_1',  # "OXY (Fit 520-600nm)",
                'cofit_oxy_0',  # "OXY (Fit 600-995nm)",
                'cofit_blo_2',  # "Blood (Fit 520-995nm)"
                'cofit_wat_0',  # "Water (Fit 600-995nm)",
                'cofit_met_0',  # "MetHb (Fit 600-995nm)",
                'cofit_mel_2',  # "Melanin (Fit 520-995nm)",
            ],
            "Oxygenation": [
                'im1',  # "RGB Image (with selection)"
                'tivita_oxy',  # "OXY (TIVITA)"
                'tivita_nir',  # "NIR (TIVITA)"
                'cofit_oxy_1',  #"OXY (Fit 520-600nm)",
                'cofit_oxy_0',  #"OXY (Fit 600-995nm)",
                'cofit_blo_2',  # "Blood (Fit 520-995nm)"
                'tivita_thi',  # "THI (TIVITA)"
            ],
            "Water and Fat": [
                'im1',  # "RGB Image (with selection)"
                'lipids_li3',  # "LPI 2nd Derv. 925nm"
                'cofit_wat_0', # "Water (Fit 600-995nm)",
                'cofit_wob_0',  # "Water/Blood(Fit 600-995nm)",
                'lipids_li3',  # "LPI 2nd Derv. 925nm"
                'cofit_wat_0', # "Water (Fit 600-995nm)",
                'cofit_wob_0',  # "Water/Blood(Fit 600-995nm)",
            ],
            "User defined": [
                'im1',  # "RGB Image (with selection)"
                'tivita_oxy',  # "OXY (TIVITA)"
                'tivita_nir',  # "NIR (TIVITA)"
                'tivita_thi',  # "THI (TIVITA)"
                'tivita_thi',  # "THI (TIVITA)"
                'tivita_twi',  # "TWI (TIVITA)"
                'tivita_twi',  # "TWI (TIVITA)"
            ],
        }

    def __init__(self, *args, **kwargs):
        """ Constructor
        """
        parent = kwargs.get('parent', None)
        super(QHSMultiAnalyzerViewConfigWidget, self).__init__(parent=parent)

        if len(args) == 1:
            kwargs['dir'] = args[0]
        elif len(args) > 1:
            raise TypeError("To many arguments {}".format(args))

        self.dir = kwargs.get('dir', None)
        self.filePath = None

        self.addParamViewButton = QtWidgets.QToolButton(self)
        self.saveParamViewButton = QtWidgets.QToolButton(self)
        self.loadParamViewButton = QtWidgets.QToolButton(self)
        self.viewComboBox = QtWidgets.QComboBox(self)

        # configure actions
        self._setupActions()

        # configure widget views
        self._setupViews(*args, **kwargs)

    def _setupActions(self):
        self.addParamViewAction = QtWidgets.QAction(self)
        self.addParamViewAction.setIconText("Add")
        self.addParamViewAction.triggered.connect(self.addCurrentParamView)
        self.addAction(self.addParamViewAction)

        self.saveParamViewAction = QtWidgets.QAction(self)
        self.saveParamViewAction.setIconText("Save")
        self.saveParamViewAction.triggered.connect(self.saveParamViewConfig)
        self.addAction(self.saveParamViewAction)

        self.loadParamViewAction = QtWidgets.QAction(self)
        self.loadParamViewAction.setIconText("Load")
        self.loadParamViewAction.triggered.connect(self.loadParamViewConfig)
        self.addAction(self.loadParamViewAction)

        self.addParamViewButton.setDefaultAction(self.addParamViewAction)
        self.saveParamViewButton.setDefaultAction(self.saveParamViewAction)
        self.loadParamViewButton.setDefaultAction(self.loadParamViewAction)

    def _setupViews(self, *args, **kwargs):
        # self.mainLayout = QtWidgets.QVBoxLayout()
        self.mainLayout = QtWidgets.QFormLayout()
        self.mainLayout.setContentsMargins(5, 10, 5, 10) # left, top, right, bottom
        self.mainLayout.setSpacing(3)
        self.setLayout(self.mainLayout)

        label = QtWidgets.QLabel("View Config")
        label.setStyleSheet(
            "border: 0px;"
            "font: bold;"
        )
        label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.mainLayout.addRow(label)

        for key in self.views.keys():
            self.viewComboBox.addItem(key)
        # self.viewComboBox.setReadOnly(True)
        # self.viewComboBox.setMinimumHeight(350)
        # self.viewComboBox.setMaximumHeight(600)
        self.mainLayout.addRow(self.viewComboBox)

        label = QtWidgets.QLabel(self)
        label.setMinimumHeight(20)
        label.setStyleSheet("border: 0px;")
        self.addParamViewButton.setText("Add")
        self.saveParamViewButton.setText("Save")
        self.loadParamViewButton.setText("Load")

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.addParamViewButton)
        layout.addWidget(self.saveParamViewButton)
        layout.addWidget(self.loadParamViewButton)
        self.mainLayout.addRow(layout)

        # connect signals
        self.viewComboBox.currentTextChanged.connect(
            self.onCurrentParamViewChanged)

    def addCurrentParamView(self):
        """Add view to current configuration
        """
        key, ok = QtWidgets.QInputDialog.getText(
            self, 'Add view', 'Define name for the current view')

        if ok:
            view = self.views[self.viewComboBox.currentText()]
            self.viewComboBox.addItem(key)
            logger.debug("Set new view '{}': {}".format(key, view))
            self.views.update({key: view})

    def clear(self):
        self.viewComboBox.setText("")

    def finalize(self):
        """ Should be called manually before object deletion
        """
        logger.debug("Finalizing: {}".format(self))
        super(QHSMultiAnalyzerViewConfigWidget, self).finalize()

    def getParamView(self):
        key = self.viewComboBox.currentText()
        if key in self.views.keys():
            return self.views[key]

    def loadParamViewConfig(self):
        """Load view configuration
        """
        filter = "Json (*.json)"
        filePath, filter = QtWidgets.QFileDialog.getOpenFileName(
            None, 'Select file:', filter=filter)

        logger.debug("Import view configuration from {}".format(filePath))
        with open(filePath) as file:
            self.views = json.load(file)

        self.viewComboBox.blockSignals(True)
        self.viewComboBox.clear()
        for key in self.views.keys():
            self.viewComboBox.addItem(key)
        self.viewComboBox.setCurrentIndex(0)
        self.viewComboBox.blockSignals(False)


    def onCurrentParamViewChanged(self, key):
        if key in self.views.keys():
            view = self.views[key]
            self.sigCurrentParamViewChanged.emit(view)

    def saveParamViewConfig(self):
        """Export view configuration
        """
        filter = "Json (*.json)"
        filePath, filter = QtWidgets.QFileDialog.getSaveFileName(
            None, 'Save file:', filter=filter)

        logger.debug("Export view configuration to {}".format(filePath))

        views = json.dumps(self.views, sort_keys=False, indent=4)
        with open(filePath +  ".json", 'w') as file:
            file.write(views)

    def setParamView(self, key, view=None):
        """Add view to current configuration
        """
        if key not in self.views.keys():
            return

        if view is None:
            view = self.views[key]
            logger.debug("Set parameter view '{}': {}".format(key, view))
            self.viewComboBox.setCurrentText(key)
            self.sigCurrentParamViewChanged.emit(view)
        else:
            logger.debug("Set new view '{}': {}".format(key, view))
            self.views[key] = view
            self.viewComboBox.setCurrentText(key)
            self.sigCurrentParamViewChanged.emit(view)



class QHSMultiAnalyzerWidget(QtWidgets.QWidget):

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
            RegnImagCtrlItem("Image Control Item 6", cbarWidth=10),
        ]

        # self.spectViewer = RegnPlotCtrlItem(
        #     "spectral attenuation", xlabel="wavelength", xunits="m")
        self.spectViewer = CurveViewItem(
            "Spectral attenuation", xlabel="Wavelength", xunits="m")

        self.spectPlotItem = pg.PlotItem()

        self.curveItems = {
            'crv0': pg.PlotCurveItem(
                name="Raw spectral data",
                pen=pg.mkPen(color=(100, 100, 100), width=1)),
            'crv1': pg.PlotCurveItem(
                name="Filtered spectrum",
                pen=pg.mkPen(color=(255, 255, 255), width=2)),
            'crv2': pg.PlotCurveItem(
                name="Fit 600-995nm",
                pen=pg.mkPen(color=(255, 0, 0), width=1)),
            'crv3': pg.PlotCurveItem(
                name="Fit 500-600nm",
                pen=pg.mkPen(color=(0, 255, 0), width=1)),
            'crv4': pg.PlotCurveItem(
                name="Fit 500-995nm",
                pen=pg.mkPen(color=(0, 255, 255), width=1),
            )
        }

        if MAINZ:
            self.curveItems = {
                'crv0': pg.PlotCurveItem(
                    name="Raw spectral data",
                    pen=pg.mkPen(color=(100, 100, 100), width=1)),
                'crv1': pg.PlotCurveItem(
                    name="Filtered spectrum",
                    pen=pg.mkPen(color=(255, 255, 255), width=2)),
            }

        # initiate image config widget
        if not getattr(sys, 'frozen', False):
            import os.path
            data_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "data")
            self.hsImageConfig = QHSImageConfigWidget(dir=data_path)
        else:
            self.hsImageConfig = QHSImageConfigWidget()

        # initiate Tivita analysis module
        # self.hsTivitaAnalysis = HSOpenTivita(hsformat=HSAbsorption)
        self.hsTivitaAnalysis = HSTivita(hsformat=HSIntensity)

        # initiate Moussa's lipid index analysis module
        self.hsLipidsAnalysis = HSLipids(hsformat=HSIntensity)

        # initiate Moussa's oxygen index analysis module
        self.hsOxygenAnalysis = HSOxygen(hsformat=HSIntensity)

        # view configuration widget
        self.hsViewConfig = QHSMultiAnalyzerViewConfigWidget()

        # Widget to output mean values over region of interest
        self.hsROIParamView = QHSROIParamWidget()

        # initiate component fit analysis module
        self.hsCoFitAnalysis = HSCoFit(hsformat=HSAbsorption)
        self.hsCoFitAnalysis.loadtxt("basevectors_2_17052022.txt", mode='all')
        self.hsCoFitAnalysis.set_var_bounds("hhb", [0, 0.1])
        self.hsCoFitAnalysis.set_var_bounds("ohb", [0, 0.1])
        self.hsCoFitAnalysis.set_var_bounds("wat", [0, 2.00])
        self.hsCoFitAnalysis.set_var_bounds("met", [0, 0.10])
        self.hsCoFitAnalysis.set_var_bounds("mel", [0, 0.20])

        self.param = {}
        self.roiparam = {}

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
        for item in self.imagCtrlItems:
            item.setMinimumHeight(380)
        self.spectViewer.setMinimumHeight(380)

        self.graphicsLayoutWidget.addItem(self.imagCtrlItems[0], 0, 0)
        self.graphicsLayoutWidget.addItem(self.imagCtrlItems[1], 0, 1)
        self.graphicsLayoutWidget.addItem(self.imagCtrlItems[2], 0, 2)
        self.graphicsLayoutWidget.addItem(self.spectViewer, 0, 3)
        self.graphicsLayoutWidget.addItem(self.imagCtrlItems[3], 2, 0)
        self.graphicsLayoutWidget.addItem(self.imagCtrlItems[4], 2, 1)
        self.graphicsLayoutWidget.addItem(self.imagCtrlItems[5], 2, 2)
        self.graphicsLayoutWidget.addItem(self.imagCtrlItems[6], 2, 3)

        infoLabel = QtWidgets.QLabel(
            " --- ".join(["Not for clinical use" for _ in range(7)]).upper())
        infoLabel.setStyleSheet(
            "border: 0px;"
            "font: bold 14px;"
            "color: rgb(200,0,0);"
            "background-color: black;"
            "border-style: outset;"
            "border-width: 10px;"
        )
        infoLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        infoLabelProxy = QtWidgets.QGraphicsProxyWidget()
        infoLabelProxy.setWidget(infoLabel)
        self.graphicsLayoutWidget.addItem(infoLabelProxy, 1, 0, colspan=4)

        self.mainLayout.addWidget(self.graphicsLayoutWidget)
        # self.mainLayout.addWidget(self.graphicsLayoutWidget2)

        # user config widgets
        self.hsImageConfig.setMaximumWidth(220)
        # self.hsComponentFitConfig.setMaximumWidth(220)
        # self.hsImageConfig.setFormat(HSAbsorption)
        self.hsImageConfig.setFormat(HSExtinction)
        # self.hsImageConfig.imageFilterTypeComboBox.setCurrentIndex(0)
        self.hsImageConfig.imageFilterTypeComboBox.setCurrentIndex(1)
        self.hsROIParamView.setMaximumWidth(220)
        layoutConfig = QtWidgets.QVBoxLayout()
        layoutConfig.addWidget(self.hsImageConfig)

        # separation line
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        layoutConfig.addWidget(line)

        # view configuration widget
        layoutConfig.addWidget(self.hsViewConfig)

        # separation line
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        layoutConfig.addWidget(line)

        # output for roi paramters
        layoutConfig.addWidget(self.hsROIParamView)

        layoutConfig.addStretch()
        self.mainLayout.addLayout(layoutConfig)

        # connect signals
        for item in self.imagCtrlItems[0:]:
            item.setXYLink(self.imagCtrlItems[0])
            item.sigCursorPositionChanged.connect(self.updateCursorPosition)

        for item in self.imagCtrlItems[1:]:
            item.sigROIMaskChanged.connect(self.updateROIParams)

        self.hsImageConfig.sigValueChanged.connect(self.setHSImage)
        self.hsROIParamView.sigValueChanged.connect(self.onROIParamViewChanged)
        self.hsViewConfig.sigCurrentParamViewChanged.connect(self.updateParamView)

        for item in self.imagCtrlItems:
            item.sigSelectedImageChanged.connect(self.onSelectedImageChanged)

    def onSelectedImageChanged(self, key):
        view = [item.currentImage() for item in self.imagCtrlItems]
        # print(view)
        self.hsViewConfig.blockSignals(True)
        self.hsViewConfig.setParamView("User defined", view)
        self.hsViewConfig.blockSignals(False)

    def onROIParamViewChanged(self, textEdit, str):
        if str=="":
            for item in self.imagCtrlItems[1:]:
                item.clearROIMask()

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

        self.hsOxygenAnalysis.set_data(
            self.spectra, self.wavelen, hsformat=hsformat)
        self.hsOxygenAnalysis.evaluate(mask=mask)

        method = 'bvls_f'
        self.hsCoFitAnalysis.set_data(
            self.fspectra, self.wavelen, hsformat=hsformat)
        self.hsCoFitAnalysis.prepare_ls_problem()
        self.hsCoFitAnalysis.freeze_component("met")
        self.hsCoFitAnalysis.set_roi([520e-9, 995e-9])
        self.hsCoFitAnalysis.fit(method=method, mask=mask)
        self.hsCoFitAnalysis.set_roi([520e-9, 600e-9])
        self.hsCoFitAnalysis.fit(method=method, mask=mask)
        self.hsCoFitAnalysis.unfreeze_component("met")
        self.hsCoFitAnalysis.set_roi([600e-9, 995e-9])
        self.hsCoFitAnalysis.fit(method=method, mask=mask)

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
        param.update(self.hsOxygenAnalysis.get_solution(unpack=True))
        param.update(self.hsLipidsAnalysis.get_solution(unpack=True))
        param.update(
            self.hsCoFitAnalysis.get_solution(which="all", unpack=True))

        # additional parameter combinations
        prefix = self.hsCoFitAnalysis.prefix
        for i in range(3):
            param["%sblo_%d" % (prefix, i)] = \
                param["%shhb_%d" % (prefix, i)] + param[
                    "%sohb_%d" % (prefix, i)]
            idx = np.nonzero(param["%sblo_%d" % (prefix, i)])
            # oxy = ohb / blood
            param["%soxy_%d" % (prefix, i)] = np.zeros(
                param["%sblo_%d" % (prefix, i)].shape)
            param["%soxy_%d" % (prefix, i)][idx] = \
                param["%sohb_%d" % (prefix, i)][idx] / param[
                    "%sblo_%d" % (prefix, i)][idx]
            # water / blood
            param["%swob_%d" % (prefix, i)] = np.zeros(
                param["%sblo_%d" % (prefix, i)].shape)
            param["%swob_%d" % (prefix, i)][idx] = \
                param["%swat_%d" % (prefix, i)][idx] / param[
                    "%sblo_%d" % (prefix, i)][idx]

        keys = [key for key in PARAM_CONFIG.keys() if key in param.keys()]
        nkeys = len(keys)
        self.param = dict([(key, param[key]) for key in keys])
        self.roiparam = dict([(key, 0) for key in keys])
        for i, item in enumerate(self.imagCtrlItems[1:]):
            item.blockSignals(True)
            item.setData(param, PARAM_CONFIG)
            item.clearROIMask()
            item.blockSignals(False)

        self.mspectra = convert(
            self.hsImageConfig.getFormat(), HSAbsorption,
            self.hsCoFitAnalysis.model(which="all"), self.wavelen
        )

        self.updateSpectralView()
        self.updateParamView()
        self.hsROIParamView.clear()

    def updateParamView(self, view=None):
        if view is None:
            view = self.hsViewConfig.getParamView()

        for i, item in enumerate(self.imagCtrlItems):
            item.blockSignals(True)
            item.selectImage(view[i])
            item.blockSignals(False)


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

    def updateROIParams(self, pts, roimask):
        sender = self.sender()
        for item in self.imagCtrlItems[1:]:
            if item != sender:
                item.blockSignals(True)
                item.setROIMask(pts)
                item.blockSignals(False)

        image_count = 1
        roi_param = {}
        mask = roimask * self.hsImageConfig.getMask()
        for key in self.param.keys():
            m = mask.reshape(-1)
            m_idx = np.ix_(range(image_count), m == 1)
            p = self.param[key].reshape(image_count, -1)
            roi_param[key] = np.mean(p[m_idx], axis=1)

        self.roi_param = roi_param
        # print(roi_param)

        msg = "\n".join([
            "%-25s %8.5f" % (PARAM_CONFIG[key]+":", roi_param[key])
            for key in roi_param.keys()])
        self.hsROIParamView.setText(msg)

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

        if not MAINZ:
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

    win = QHSMultiAnalyzerWidget()
    # win.setGeometry(300, 30, 1200, 500)
    win.setGeometry(40, 160, 1820, 800)
    win.setWindowTitle("Multi Index Analysis")
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

