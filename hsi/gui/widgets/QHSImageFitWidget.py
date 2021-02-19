# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:46:58 2020

@author: papkai
"""
import sys
import os

import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap

from pyqtgraph.Qt import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.parametertree as pgtree

# from myqt.datavisualization import BaseImagCtrlItem
# from myqt.datavisualization import HistImagCtrlItem
# from myqt.datavisualization import PosnImagCtrlItem
# from myqt.datavisualization import RegnPlotCtrlItem


from ...core.HSImage import HSImage
from ...analysis.SpectralTissueCompound import SpectralTissueCompound

from ..graphicsItems.BaseImagCtrlItem import BaseImagCtrlItem
from ..graphicsItems.HistImagCtrlItem import HistImagCtrlItem
from ..graphicsItems.PosnImagCtrlItem import PosnImagCtrlItem
from ..graphicsItems.RegnPlotCtrlItem import RegnPlotCtrlItem

from .QHSImageConfigWidget import QHSImageConfigWidget
from .QHSVectorConfigWidget import QHSVectorConfigWidget

import logging

LOGGING = True
# LOGGING = False

logger = logging.getLogger(__name__)
logger.propagate = LOGGING


__all__ = ['QHSImageFitWidget']



def loadBaseModel(filePath):
    '''
    load basis spectra to describe tissue compound from file
    :return:
    '''

    data = np.load(filePath, allow_pickle=True)
    wavelen = data['wavelen']
    baseSpec = data['baseSpec']
    basePara = data['basePara'][()]

    return SpectralTissueCompound(wavelen, [], baseSpec, basePara)



class QHSImageFitWidget(QtGui.QWidget):

    def __init__(self, **kwargs):


        # general information
        self.dir = kwargs.get('dir', os.getcwd())  # initial directory
        self.dirConfig = kwargs.get('config', ".")  # hyper spectral image

        # information on base spectra to describe tissue compound model
        # self.wavelen = np.linspace(500., 1000., 100, endpoint=False)
        # self.baseModel = None  # fitting model using basis spectra
        self.baseModel = loadBaseModel(
            os.path.join(self.dirConfig, "Versuch11_Spectra.npz"))
        self.wavelen = self.baseModel.wavelen

        # hyper spectral image ...............................................
        self.hsi = HSImage()

        self.hsiData = {}
        self.hsiData['raw'] = np.array([[[]]])
        self.hsiData['fil'] = np.array([[[]]])
        self.hsiData['fit'] = np.array([[[]]])

        self.imgData = {
            'rgb': np.array([[]]),
            'blo': np.array([[]]),
            'oxy': np.array([[]]),
            'wat': np.array([[]]),
            'fat': np.array([[]]),
            'mel': np.array([[]]),
        }

        # fit configuration ..................................................
        self.fitMethod = kwargs.get('fitmethod', 'gesv')
        colors = np.loadtxt(os.path.join(self.dirConfig, "cmap_tivita.txt"))
        positions = np.linspace(0, 1, len(colors))
        # self.gColorMap = pg.ColorMap(positions, colors)
        # self.gColorMap = pg.colormaps.Viridis()
        # self.colormap = ListedColormap(colors)


        # pltMap = plt.get_cmap('nipy_spectral')
        # colors = pltMap.colors
        # colors = [c + [1.] for c in colors]
        # positions = np.linspace(0, 1, len(colors))
        # self.gColorMap = pg.ColorMap(positions, colors)

        # colormap = plt.cm.get_cmap("nipy_spectral")  # cm.get_cmap("CMRmap")
        # colormap._init()
        # self.gColorMap = (colormap._lut * 255).view(np.ndarray)
        cmap = (colors * 255).view(np.ndarray).astype(np.uint8)


        # background mask configuration ......................................
        self.maskSegX = 10  # number of vertical segmentation lines
        self.maskSegY = 10  # number of vertical segmentation lines

        # graphics layout ....................................................
        QtGui.QWidget.__init__(self)

        # image and spectral attenuation plots ...............................
        self.imagCtrlItems = {
            'rgb': PosnImagCtrlItem("RGB Image"),
            'blo': HistImagCtrlItem(cmap, label="blood", cbarWidth=10),
            'oxy': HistImagCtrlItem(cmap, label="oxigenation", cbarWidth=10),
            'wat': HistImagCtrlItem(cmap, label="water", cbarWidth=10),
            'fat': HistImagCtrlItem(cmap, label="fat", cbarWidth=10),
            'mel': HistImagCtrlItem(cmap, label="melanin", cbarWidth=10),
        }

        self.spectViewer = RegnPlotCtrlItem(
            "spectral attenuation", xlabel="wavelength", xunits="m")

        self.curveItems = {
            'raw': pg.PlotCurveItem(
                name="raw spectral data",
                pen=pg.mkPen(color=(100, 100, 100), width=1)),
            'flt': pg.PlotCurveItem(
                name="filtered spectrum",
                pen=pg.mkPen(color=(255, 255, 255), width=1)),
            'fit': pg.PlotCurveItem(
                name="fitted spectrum",
                pen=pg.mkPen(color=(255, 0, 0), width=1))
        }

        # config widgets
        self.hsImageConfig = QHSImageConfigWidget()
        self.hsFitConfig = QHSVectorConfigWidget()

        self._setupViews()

        logger.debug("init HSIAnalysis")


    def _setupViews(self, *args, **kwargs):

        # basic layout configuration
        self.gLayout = QtGui.QHBoxLayout()
        self.gLayout.setSpacing(0)
        # self.layout.setMargin(0)                  # pyqt
        self.gLayout.setContentsMargins(0, 0, 0, 0)  # pyside
        self.setLayout(self.gLayout)

        self.gLayout1 = QtGui.QVBoxLayout()
        self.gLayout.addLayout(self.gLayout1)
        self.gLayout2 = QtGui.QVBoxLayout()
        self.gLayout.addLayout(self.gLayout2)

        self.gLayoutPlots = pg.GraphicsLayoutWidget()
        # self.gLayout1.addWidget(self.gLayoutPlots)

        self.gImagePlots = {}
        self.gImages = {}

        # graphics layout widget
        self.graphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self.gLayout1.addWidget(self.graphicsLayoutWidget)

        # place image control items
        for key, item in self.imagCtrlItems.items():
            item.setMaximumWidth(440)
            item.setAspectLocked()
            item.invertY()
            if key == 'rgb':
                item.setMaximumHeight(350)
            else:
                item.setMaximumHeight(440)

        self.graphicsLayoutWidget.addItem(self.imagCtrlItems['rgb'], 0, 0)
        self.graphicsLayoutWidget.addItem(self.imagCtrlItems['blo'], 0, 1)
        self.graphicsLayoutWidget.addItem(self.imagCtrlItems['oxy'], 0, 2)
        self.graphicsLayoutWidget.addItem(self.imagCtrlItems['wat'], 1, 0)
        self.graphicsLayoutWidget.addItem(self.imagCtrlItems['fat'], 1, 1)
        self.graphicsLayoutWidget.addItem(self.imagCtrlItems['mel'], 1, 2)

        # link image control items
        firstItem = next(iter(self.imagCtrlItems.values()))
        for item in self.imagCtrlItems.values():
            item.setXYLink(firstItem)
            item.sigCursorPositionChanged.connect(self.updateCursorPosition)

        # place plot control items
        for item in self.curveItems.values():
            self.spectViewer.addItem(item)

        self.graphicsLayoutWidget.addItem(self.spectViewer, 0, 3, rowspan=2)

        # Console window for stdout and stderr ...............................
        # self.gConsole = QtGui.QTextBrowser()
        # self.gConsole.setMaximumHeight(132)
        # self.gLayout1.addWidget(self.gConsole)

        # self.gConsole.setStyleSheet(
        #     "QTextBrowser { "
        #     "border: 500px; "
        #     "background-color : rgb(0, 0, 0); "
        #     # "background-color : rgb(46, 52, 54); "
        #     "color : rgb(211, 215, 196) "
        #     "}")

        # config widget for hsimage ..........................................
        self.hsImageConfig.setMaximumWidth(200)
        self.gLayout2.addWidget(self.hsImageConfig)
        self.hsFitConfig.setMaximumWidth(200)
        self.gLayout2.addWidget(self.hsFitConfig)
        self.gLayout2.addStretch()

        # Configuration tree .................................................
        # self.gTree = pgtree.ParameterTree(showHeader=False)
        # self.gTree.setMaximumSize(QtCore.QSize(200, 1400))
        # self.gTree.setMinimumSize(QtCore.QSize(200, 570))
        # self.gLayout2.addWidget(self.gTree)
        #
        # parLimits = []
        # for key, val in self.baseModel.compound.items():
        #     sval = "%g ... %g" % (val[0], val[1])
        #     parLimits.append({'name': key, 'type': 'str', 'value': sval})
        #
        # params = [
        #     {'name': 'Coordinates', 'type': 'group', 'children': [
        #         {'name': 'x', 'type': 'int', 'value': 0},
        #         {'name': 'y', 'type': 'int', 'value': 0},
        #     ]},
        #     {'name': 'Fit', 'type': 'group', 'children': [
        #         {'name': 'wmin', 'type': 'float', 'value': self.wavelen[0], 'limits': self.wavelen[[0, -1]]},
        #         {'name': 'wmax', 'type': 'float', 'value': self.wavelen[-1], 'limits': self.wavelen[[0, -1]]},
        #         {'name': 'method', 'type': 'list', 'values': self.baseModel.methods,
        #          'value': self.baseModel.methods[0]},
        #         {'name': 'quadratic', 'type': 'bool', 'value': True},
        #         {'name': 'single', 'type': 'bool', 'value': False},
        #         {'name': 'fit', 'type': 'action'},
        #         {'name': 'limits', 'type': 'group', 'children': parLimits, 'expanded': False},
        #     ]},
        # ]
        # self.gTreeParam = pgtree.Parameter.create(name='params', type='group', children=params)
        # self.gTree.setParameters(self.gTreeParam, showTop=False)

        # self.gTreeParam.param('Fit').sigTreeStateChanged.connect(self.onFitSettings)

        # table = pg.TableWidget()
        # self.gLayout2.addWidget(table)
        # a = np.zeros([2,2])
        # table.setData(a)
        # table.setHorizontalHeaderLabels(["min", "max"])
        # par = self.baseModel.compound.keys()
        # table.setVerticalHeaderLabels(par)

        # main the menu box with all the control buttons .....................
        self.gMenu = QtGui.QVBoxLayout()
        self.gMenu.setSpacing(5)
        self.gMenu.setContentsMargins(20, 10, 20, 20)
        self.gLayout2.addLayout(self.gMenu)

        self.gBtnQuit = QtGui.QPushButton('Quit')
        self.gBtnQuit.setMaximumHeight(30)
        self.gBtnQuit.setMinimumHeight(30)
        self.gMenu.addWidget(self.gBtnQuit)

        # connect signals
        self.hsImageConfig.sigValueChanged.connect(self.updateImage)


       #  self.setStyleSheet(
       #      "color: rgb(150,150,150);"
       #      "background-color: black;"
       #      "selection-color: white;"
       #      "selection-background-color: rgb(0,118,211);"
       #      "selection-border-color: blue;"
       #      "border-style: outset;"
       #      "border-width: 1px;"
       #      "border-radius: 2px;"
       #      "border-color: grey;"
       # )



    def updateCursorPosition(self):
        sender = self.sender()
        if not isinstance(sender, BaseImagCtrlItem):
            raise TypeError("Unexpected type {}, was expecting {}"
                            .format(type(sender), BaseImagCtrlItem))

        pos = sender.getCursorPos()
        for item in self.imagCtrlItems.values():  # link cursors
            if item is not sender:
                item.blockSignals(True)
                item.setCursorPos(pos)
                item.blockSignals(False)

        self.updateSpectViewer()

        logger.debug("Update cursor position. Sender: {}".format(sender))


    def updateImage(self, hsImageConfig, newFile):
        self.imagCtrlItems['rgb'].setImage(hsImageConfig.rgbValue())
        if newFile:
            self.imagCtrlItems['rgb'].autoRange()
            self.imagCtrlItems['rgb'].setCursorPos((0,0))

        self.updateModel()
        self.updateSpectViewer()

        # data = hsImageConfig.value()


    def updateModel(self):
        if self.hsImageConfig.isEmpty():
            return

        data = self.hsImageConfig.value()
        k, m, n = data.shape
        self.baseModel.setSpectra(data.reshape((k, m * n)))

        xmin, xmax = self.spectViewer.getRegion()
        # xmin = 500
        # xmax = 990
        xmin *= 1e9 # rescale from m to nm
        xmax *= 1e9
        self.baseModel.setFittingRange(xmin, xmax)
        self.baseModel.fit(method='gesv')#,quadratic=True)
        # self.baseModel.fitLinear(method='gesv')  # ,quadratic=True)
        # self.baseModel.fitLinear(method=self.fitMethod)
        # self.baseModel.fitLinearConstraint(method='bvls', quadratic=True)

        param = self.baseModel.getParameter(unpack=True)
        for key in ['blo', 'oxy', 'wat', 'fat', 'mel']:
            self.imagCtrlItems[key].setImage(param[key].reshape((m, n)))


    def updateSpectViewer(self):
        """Retrieve hyper spectral data at current cursor position
        """
        if self.hsImageConfig.isEmpty():
            return

        x, y = self.imagCtrlItems['rgb'].getCursorPos()
        pos = [int(x), int(y)]
        x = self.wavelen * 1e-9
        y1 = self.hsImageConfig.value(pos, filter=False)
        y2 = self.hsImageConfig.value(pos, filter=True)
        self.curveItems['raw'].setData(x, y1)
        self.curveItems['flt'].setData(x, y2)


    # def onUpdateRegionWavelen(self):
    #     xmin, xmax = self.gRegionWavelen.getRegion()
    #     self.gSpectralPlots['plot2'].setXRange(xmin, xmax, padding=0)
    #
    #     self.gTreeParam.param('Fit').blockSignals(True)
    #     self.gTreeParam['Fit', 'wmin'] = xmin
    #     self.gTreeParam['Fit', 'wmax'] = xmax
    #     self.gTreeParam.param('Fit').blockSignals(False)
    #
    #
    # def onUpdateWavelen(self):
    #     # self.regWavelen.setZValue(10)
    #     # xmin, xmax = self.gSpectralPlots['plot1'].getRegion()
    #     # self.gSpectralPlots['plot2'].setXRange(xmin, xmax, padding=0)
    #     pass
    #
    #
    # def onUpdateImagePlots(self, auto=False):
    #     # self.gImages['rgb'].setImage(self.hsi.getRGBImage())
    #
    #     x = self.gTreeParam['Coordinates', 'x']
    #     y = self.gTreeParam['Coordinates', 'y']
    #
    #     for key, line in self.gVLines.items():
    #         line.setPos(x)
    #     for key, line in self.gHLines.items():
    #         line.setPos(y)
    #
    #     for key, gimg in self.gImages.items():
    #         if key == 'rgb':
    #             gimg.setImage(self.imgData[key])
    #         else:
    #             levels = self.baseModel.compound[key][:2]
    #             gimg.setImage(self.imgData[key], levels=levels)
    #
    #     if auto:
    #         self.gImagePlots['rgb'].autoRange()
    #
    #
    # def onUpdateSpectralPlots(self):
    #
    #     for ckey, cobj in self.curveItemsConfig.items():
    #         for pkey in cobj.plotkeys:
    #             if cobj.enabled:
    #                 x = self.wavelen
    #                 y = self.hsiData[ckey][:, 0, 0]
    #             else:
    #                 x = np.array([])
    #                 y = np.array([])
    # #             self.gSpectralCurves[ckey][pkey].setData(x=x, y=y)
    #
    #
    # def onFitSettings(self, param, changes):
    #     (param, _, value) = changes[0]
    #     param = self.gTreeParam.childPath(param)[-1]
    #
    #     self.gTreeParam.param('Fit').blockSignals(True)
    #     self.gRegionWavelen.blockSignals(True)
    #
    #     if param == 'method':
    #         if value != 'gesv' and value != 'bvls_f':
    #             self.gTreeParam['Fit', 'single'] = True
    #     elif param == 'wmin' or param == 'wmax':
    #         xmin = self.gTreeParam['Fit', 'wmin']
    #         xmax = self.gTreeParam['Fit', 'wmax']
    #         self.gRegionWavelen.setRegion([xmin, xmax])
    #
    #     self.gTreeParam.param('Fit').blockSignals(False)
    #     self.gRegionWavelen.blockSignals(False)


    def saveState(self):
        pass
        # colors = np.loadtxt(os.path.join(self.dirConfig, "cmap_tivita.txt"))
        # cmap = ListedColormap(colors)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # # pos = plt.imshow(par, cmap='gray', vmin=0, vmax=1)
        # pos = plt.imshow(self.imgData['oxy'], cmap=cmap, vmin=0, vmax=1)
        # fig.colorbar(pos, ax=ax)
        #
        # filePath = os.path.join(pict_path, baseName +
        #                         "_StO2_%d-%dnm_fit" % tuple(wavelen_limits))
        # plt.savefig(filePath + ".png", format="png", dpi=900)
        # plt.show()

        # if len(self.m_data):
        #     _idx = self.params.param('Particle', 'No.').value()
        #     _filename = os.path.join(self.m_directory, '..', '..', '..', 'pictures', 'Particle_%d_%d.png'
        #                              % (self.m_data[_idx, 0], self.m_data[_idx, 1]))
        #
        #     pixmap = QtGui.QPixmap.grabWindow(self.winId())
        #     pixmap.save(_filename, format='png')
        #     print('%s | Results written to %s' % (strftime("%H:%M:%S", gmtime()), _filename))



# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':


    pg.mkQApp()

    confPath = os.path.join(os.getcwd(), "..", "config")
    dataPath =  os.path.join(os.getcwd(), '..', 'data')
    # dataPath = os.path.join(os.getcwd(), '..', 'studies')

    win = QHSImageFitWidget(dir=dataPath, config=confPath)
    win.setWindowTitle("Hyperspectral Image Analysis")
    win.resize(900, 900)
    win.show()

    if (sys.flags.interactive != 1) or not hasattr(pg.QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.instance().exec_()
