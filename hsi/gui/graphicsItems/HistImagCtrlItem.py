import sys
import os.path
import numpy as np
import pyqtgraph as pg

from ...bindings.Qt import QtWidgets, QtGui, QtCore
from ...log import logmanager
from ...misc import getPkgDir
from ...core.hs_cm import cm

from .ColorBarItem import ColorBarItem
from .BaseImagCtrlItem import BaseImagCtrlItem

logger = logmanager.getLogger(__name__)

__all__ = ['HistImagCtrlItem']


class QHistImagCtrlConfigWidget(QtWidgets.QWidget):

    sigToggleHistogramChanged = QtCore.Signal(object)

    """ Config widget with two spinboxes that control the image levels.
    """
    def __init__(self, colorBarItem, label=None, labels=None, parent=None):
        """ Constructor
        """
        super(QHistImagCtrlConfigWidget, self).__init__(parent=parent)

        self.colorBarItem = colorBarItem

        self._setupActions()
        self._setupViews(label, labels)


    def _setupActions(self):

        self.resetAction = QtWidgets.QAction("Reset", self)
        self.resetAction.triggered.connect(self.colorBarItem.resetColorLevels)
        # self.resetAction.setShortcut("Ctrl+0")
        self.addAction(self.resetAction)

        self.selectImageComboBox.currentTextChanged.connect(
            self._updateImageSelection)
        # self.toggleHistogramAction = QtWidgets.QAction("Hist", self)
        # self.toggleHistogramAction.setCheckable(True)
        # self.toggleHistogramAction.setChecked(self.colorBarItem.histogramIsVisible)
        # self.toggleHistogramAction.triggered.connect(self.colorBarItem.showHistogram)
        # # self.toggleHistogramAction.setShortcut("Ctrl+H")
        # self.addAction(self.toggleHistogramAction)


    def _setupViews(self, label=None, labels=None):

        self.mainLayout = QtWidgets.QHBoxLayout()
        self.mainLayout.setContentsMargins(5, 0, 5, 0) # left, top, right, bottom
        self.mainLayout.setSpacing(3)
        self.setLayout(self.mainLayout)

        self.selectImageComboBox = QtGui.QComboBox(self)
        if labels is not None:
            self.selectImageComboBox.addItems(labels)
            self.selectImageComboBox.setCurrentText(labels[0])
        self.selectImageComboBox.setMinimumWidth(120)
        self.mainLayout.addWidget(self.selectImageComboBox)
        self.mainLayout.addStretch()

        # if label is None:
        #     self.label = None
        # else:
        #     self.label = QtWidgets.QLabel(label, self)
        #     self.label.setStyleSheet(
        #         "border-color: black;"
        #         "font: bold 14px;"
        #     )
        #     self.label.setMinimumWidth(120)
        #     self.mainLayout.addStretch()
        #     self.mainLayout.addWidget(self.label)

        self.mainLayout.addStretch()
        self.label = QtWidgets.QLabel("Limits", self)
        self.label.setStyleSheet("border-color: black;")
        self.mainLayout.addWidget(self.label)

        self.minLevelSpinBox = QtWidgets.QDoubleSpinBox(self)
        self.minLevelSpinBox.setKeyboardTracking(False)
        self.minLevelSpinBox.setMinimum(-1000)
        self.minLevelSpinBox.setMaximum(1000)
        self.minLevelSpinBox.setSingleStep(0.1)
        self.minLevelSpinBox.setDecimals(3)
        self.minLevelSpinBox.setMaximumWidth(60)
        self.mainLayout.addWidget(self.minLevelSpinBox)

        self.maxLevelSpinBox = QtWidgets.QDoubleSpinBox(self)
        self.maxLevelSpinBox.setKeyboardTracking(False)
        self.maxLevelSpinBox.setMinimum(-1000)
        self.maxLevelSpinBox.setMaximum(1000)
        self.maxLevelSpinBox.setSingleStep(0.1)
        self.maxLevelSpinBox.setDecimals(3)
        self.maxLevelSpinBox.setMaximumWidth(60)
        self.mainLayout.addWidget(self.maxLevelSpinBox)

        self.minLevelSpinBox.valueChanged.connect(lambda val: self.setLevels((val, None)))
        self.maxLevelSpinBox.valueChanged.connect(lambda val: self.setLevels((None, val)))
        self.colorBarItem.sigLevelsChanged.connect(self._updateSpinBoxLevels)

        self.resetButton = QtWidgets.QToolButton(self)
        self.resetButton.setDefaultAction(self.resetAction)
        self.mainLayout.addWidget(self.resetButton)

        # self.histogramButton = QtWidgets.QToolButton(self)
        # self.histogramButton.setDefaultAction(self.toggleHistogramAction)
        # self.mainLayout.addWidget(self.histogramButton)

        self.setStyleSheet(
            "color: rgb(150,150,150);"
            "background-color: black;"
            "selection-color: white;"
            "selection-background-color: rgb(0,118,211);"
            "selection-border-color: blue;"
            "border-style: outset;"
            "border-width: 1px;"
            "border-radius: 2px;"
            "border-color: grey;"
       )

    def finalize(self):
        """ Should be called manually before object deletion
        """
        logger.debug("Finalizing: {}".format(self))
        super(QHistImagCtrlConfigWidget, self).finalize()

    def setLabels(self, labels):
        self.labels = labels
        self.selectImageComboBox.clear()
        self.selectImageComboBox.addItems(list(labels.values()))

    def setLevels(self, levels):
        """ Sets plot levels
            :param levels: (vMin, vMax) tuple
        """
        logger.debug("Setting image levels: {}".format(levels))
        minLevel, maxLevel = levels

        # Replace Nones by the current level
        oldMin, oldMax = self.colorBarItem.getLevels()
        logger.debug("Old levels: {}".format(levels))

        if minLevel is None: # Only maxLevel was set.
            minLevel = oldMin
            if maxLevel <= minLevel:
                minLevel = maxLevel - 1

        if maxLevel is None: # Only minLevel was set
            maxLevel = oldMax
            if maxLevel <= minLevel:
                maxLevel = minLevel + 1

        self.colorBarItem.setLevels((minLevel, maxLevel))

    def _updateSpinBoxLevels(self, levels):
        """ Updates the spinboxes given the levels
        """
        minLevel, maxLevel = levels
        logger.debug("_updateSpinBoxLevels: {}".format(levels))
        self.minLevelSpinBox.setValue(minLevel)
        self.maxLevelSpinBox.setValue(maxLevel)

    def _updateImageSelection(self, label):

        for key, val in self.labels.items():
            if val == label:
                self.self.selectImageComboBox.text


class HistImagCtrlItem(BaseImagCtrlItem):

    def __init__(self,
                 label=None,
                 cmap=None,
                 showHistogram=True,
                 cbarOrientation='bottom',
                 cbarWidth=20,
                 cbarHistogramHeight=30,
                 parent=None):
        BaseImagCtrlItem.__init__(self, parent=parent)

        self.cbarWidth = cbarWidth
        self.cbarHistogramHeight = cbarHistogramHeight
        self.cbarOrientation = cbarOrientation
        self.cbarHistogramIsVisible = showHistogram

        # set color map
        if cmap is None:
            # colors = np.loadtxt(
            #     os.path.join(getPkgDir(), "data", "cmap_tivita.txt"))
            cmap = cm.tivita()
            colors = cmap(range(256))
            cmap = (colors * 255).view(np.ndarray).astype(np.uint8)

        self.imageItem.setLookupTable(cmap)

        self._setupActions()
        self._setupViews(label)

        # self.toolbarWidget.toggleHistogramAction.setChecked(
        #     self.cbarHistogramIsVisible)


    def _setupActions(self):
        """ Creates the UI actions.
        """
        # self.noiseImgAction = QtWidgets.QAction("Noise", self)
        # self.noiseImgAction.setToolTip("Sets the image data to noise.")
        # self.noiseImgAction.triggered.connect(self._setTestData)
        # self.noiseImgAction.setShortcut("Ctrl+N")
        # self.addAction(self.noiseImgAction)
        pass


    def _setupViews(self, label):

        self.colorBarItem = ColorBarItem(
            imageItem=self.imageItem,
            showHistogram=self.cbarHistogramIsVisible,
            width=self.cbarWidth,
            orientation=self.cbarOrientation)
        self.colorBarItem.setMinimumHeight(60)

        self.toolbarWidget = QHistImagCtrlConfigWidget(
            self.colorBarItem,
            label=label)
        self.toolbarProxy = QtGui.QGraphicsProxyWidget()
        self.toolbarProxy.setWidget(self.toolbarWidget)

        self.mainLayout = QtWidgets.QGraphicsGridLayout()
        self.setLayout(self.mainLayout)
        self.mainLayout.setContentsMargins(1, 1, 1, 1)
        self.mainLayout.setSpacing(10)

        # self.graphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self.mainLayout.addItem(self.toolbarProxy, 0, 0)
        self.mainLayout.addItem(self.plotItem, 1, 0)
        self.mainLayout.addItem(self.colorBarItem, 2, 0)


    def setData(self, data, labels=None):
        """ Sets the image data
        """
        # if not isinstance(data, dict):
        #     raise Exception("Plot data must be dictionary of 2D or 3D ndarray.")
        #
        # for key, img in data.items():
        #     if isinstance(img, list):
        #         data[key] = np.array(data[key])
        #     if not isinstance(img, np.ndarray):
        #         raise Exception("Plot data must be ndarray.")
        #
        # if labels is None or not isinstance(data, dict):
        #     self.labels = data.keys()
        # else:
        #     self.labels = labels
        # self.data = data

        super(HistImagCtrlItem, self).setData(data, labels)
        self.toolbarWidget.setLabels(self.labels)


    #     self.selectImageComboBox.setCurrentIndex(0)

    # def setData(self, data):
    #     """ Sets the image data
    #     """
    #     # PyQtGraph uses the following dimension order: T, X, Y, Color.
    #     if not isinstance(data, dict):
    #         raise Exception("Plot data must be dictionary of 2D or 3D ndarray.")
    #
    #     self.data = data

        # self.imageItem.setImage(img.T)
        # nRows, nCols = img.shape
        # self.plotItem.setRange(xRange=[0, nCols], yRange=[0, nRows])
        # self.colorBarItem.resetColorLevels()

        # self.cursorX.setBounds((0, nCols))
        # self.cursorY.setBounds((0, nRows))
        #
        # self.cursorX.setPos((nCols // 2))
        # self.cursorY.setPos((nRows // 2))

    def selectImage(self, key):
        """ Sets the image data
        """
        if not key in self.data.keys():
            return

        data = self.data[key]

        if data.ndim == 2:
            nRows, nCols = data.shape
            nChan = 1
            self.imageItem.setImage(data, axisOrder='row-major')
        elif data.ndim == 3:
            nRows, nCols, nChan = data.shape
            self.imageItem.setImage(data, axisOrder='row-major')
        else:
            raise Exception("Plot data must be 2D or 3D ndarray.")

        self.plotItem.setRange(xRange=[0, nCols], yRange=[0, nRows])
        self.colorBarItem.resetColorLevels()

        self.cursorX.setBounds((0, nCols-1))
        self.cursorY.setBounds((0, nRows-1))

        self.cursorX.setPos((nCols // 2))
        self.cursorY.setPos((nRows // 2))

    # def selectImage(self, key):
    #     """ Select the image data to be visualized
    #     """
    #     # PyQtGraph uses the following dimension order: T, X, Y, Color.
    #
    #     if not key in self.data.keys:
    #         return
    #
    #     data = self.data[key]
    #
    #     if data.ndim == 2:
    #         nRows, nCols = data.shape
    #         nChan = 1
    #         self.imageItem.setImage(data, axisOrder='row-major')
    #     elif data.ndim == 3:
    #         nRows, nCols, nChan = data.shape
    #         self.imageItem.setImage(data, axisOrder='row-major')
    #     else:
    #         raise Exception("Plot data must be 2D or 3D ndarray.")
    #
    #     # self.imageItem.setImage(img.T)
    #     # nRows, nCols = img.shape
    #
    #     self.plotItem.setRange(xRange=[0, nCols], yRange=[0, nRows])
    #     self.colorBarItem.resetColorLevels()
    #
    #     self.cursorX.setBounds((0, nCols))
    #     self.cursorY.setBounds((0, nRows))
    #
    #     self.cursorX.setPos((nCols // 2))
    #     self.cursorY.setPos((nRows // 2))

    def setLevels(self, levels):
        self.colorBarItem.setLevels(levels)

