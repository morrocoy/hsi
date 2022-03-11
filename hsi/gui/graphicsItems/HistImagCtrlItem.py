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
    def __init__(self, colorBarItem, label=None, parent=None):
        """ Constructor
        """
        super(QHistImagCtrlConfigWidget, self).__init__(parent=parent)

        self.colorBarItem = colorBarItem

        self._setupActions()
        self._setupViews(label)


    def _setupActions(self):

        self.resetAction = QtWidgets.QAction("reset", self)
        self.resetAction.triggered.connect(self.colorBarItem.resetColorLevels)
        # self.resetAction.setShortcut("Ctrl+0")
        self.addAction(self.resetAction)

        self.toggleHistogramAction = QtWidgets.QAction("hist", self)
        self.toggleHistogramAction.setCheckable(True)
        self.toggleHistogramAction.setChecked(self.colorBarItem.histogramIsVisible)
        self.toggleHistogramAction.triggered.connect(self.colorBarItem.showHistogram)
        # self.toggleHistogramAction.setShortcut("Ctrl+H")
        self.addAction(self.toggleHistogramAction)


    def _setupViews(self, label=None):

        self.mainLayout = QtWidgets.QHBoxLayout()
        self.mainLayout.setContentsMargins(5, 0, 5, 0) # left, top, right, bottom
        self.mainLayout.setSpacing(3)
        self.setLayout(self.mainLayout)

        if label is None:
            self.label = None
        else:
            self.label = QtWidgets.QLabel(label, self)
            self.label.setStyleSheet(
                "border-color: black;"
                "font: bold 14px;"
            )
            self.label.setMinimumWidth(120)
            self.mainLayout.addStretch()
            self.mainLayout.addWidget(self.label)

        self.mainLayout.addStretch()
        # self.label = QtWidgets.QLabel("limits", self)
        # self.label.setStyleSheet("border-color: black;")
        # self.mainLayout.addWidget(self.label)

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

        self.histogramButton = QtWidgets.QToolButton(self)
        self.histogramButton.setDefaultAction(self.toggleHistogramAction)
        self.mainLayout.addWidget(self.histogramButton)

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

        self.toolbarWidget.toggleHistogramAction.setChecked(
            self.cbarHistogramIsVisible)


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
        self.mainLayout.setSpacing(0)

        # self.graphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self.mainLayout.addItem(self.toolbarProxy, 0, 0)
        self.mainLayout.addItem(self.plotItem, 1, 0)
        self.mainLayout.addItem(self.colorBarItem, 2, 0)


    def setImage(self, img):
        """ Sets the image data
        """
        # PyQtGraph uses the following dimension order: T, X, Y, Color.
        self.imageItem.setImage(img.T)
        nRows, nCols = img.shape
        self.plotItem.setRange(xRange=[0, nCols], yRange=[0, nRows])
        self.colorBarItem.resetColorLevels()

        self.cursorX.setBounds((0, nCols))
        self.cursorY.setBounds((0, nRows))

        self.cursorX.setPos((nCols // 2))
        self.cursorY.setPos((nRows // 2))


    def setLevels(self, levels):
        self.colorBarItem.setLevels(levels)

