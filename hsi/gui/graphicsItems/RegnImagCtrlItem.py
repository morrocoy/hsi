# import sys
# import os.path
import numpy as np
from matplotlib.path import Path
import pyqtgraph as pg

from ...bindings.Qt import QtWidgets, QtCore
from ...log import logmanager
from ...misc import getPkgDir
from ...core.hs_cm import cm

from .ColorBarItem import ColorBarItem
from .BaseImagCtrlItem import BaseImagCtrlItem

logger = logmanager.getLogger(__name__)

__all__ = ['RegnImagCtrlItem']


class QRegnImagCtrlConfigWidget(QtWidgets.QWidget):

    sigToggleHistogramChanged = QtCore.Signal(object)
    sigSelectedImageChanged = QtCore.Signal(object)

    """ Config widget with two spinboxes that control the image levels.
    """
    def __init__(self, colorBarItem, label=None, labels=None, parent=None):
        """ Constructor
        """
        super(QRegnImagCtrlConfigWidget, self).__init__(parent=parent)


        self.colorBarItem = colorBarItem

        self.isHistAutoLevel = True
        self._setupActions()
        self._setupViews(label, labels)

    def _setupActions(self):

        # self.resetAction = QtWidgets.QAction("Reset", self)
        # self.resetAction.triggered.connect(self.colorBarItem.resetColorLevels)
        # # self.resetAction.setShortcut("Ctrl+0")
        # self.addAction(self.resetAction)

        self.toggleHistAutoLevelAction = QtWidgets.QAction("Auto", self)
        self.toggleHistAutoLevelAction.setCheckable(True)
        self.toggleHistAutoLevelAction.setChecked(self.isHistAutoLevel)
        self.toggleHistAutoLevelAction.triggered.connect(
            self._triggerResetColorLevels)

        # self.toggleHistogramAction = QtWidgets.QAction("Hist", self)
        # self.toggleHistogramAction.setCheckable(True)
        # self.toggleHistogramAction.setChecked(
        # self.colorBarItem.histogramIsVisible)
        # self.toggleHistogramAction.triggered.connect(
        # self.colorBarItem.showHistogram)
        # # self.toggleHistogramAction.setShortcut("Ctrl+H")
        # self.addAction(self.toggleHistogramAction)


    def _setupViews(self, label=None, labels=None):

        self.mainLayout = QtWidgets.QHBoxLayout()
        self.mainLayout.setContentsMargins(5, 0, 5, 0) # ltrb
        self.mainLayout.setSpacing(3)
        self.setLayout(self.mainLayout)

        self.selectImageComboBox = QtWidgets.QComboBox(self)
        if labels is not None:
            self.selectImageComboBox.addItems(labels)
            self.selectImageComboBox.setCurrentText(labels[0])
        # self.selectImageComboBox.setMinimumWidth(120)
        self.selectImageComboBox.setMinimumWidth(150)
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

        # connect signals
        self.minLevelSpinBox.valueChanged.connect(
            lambda val: self.setLevels((val, None)))
        self.maxLevelSpinBox.valueChanged.connect(
            lambda val: self.setLevels((None, val)))
        self.colorBarItem.sigLevelsChanged.connect(self._updateSpinBoxLevels)
        self.selectImageComboBox.currentTextChanged.connect(
            self._triggerSelectedImageChanged)

        # self.resetButton = QtWidgets.QToolButton(self)
        # self.resetButton.setDefaultAction(self.resetAction)
        # self.mainLayout.addWidget(self.resetButton)

        self.histAutoLevelButton = QtWidgets.QToolButton(self)
        self.histAutoLevelButton.setDefaultAction(
            self.toggleHistAutoLevelAction)
        self.mainLayout.addWidget(self.histAutoLevelButton)

        self.histAutoLevelButton.setStyleSheet(
            "QToolButton:checked { background-color: gray }"
        )

        # self.histogramButton = QtWidgets.QToolButton(self)
        # self.histogramButton.setDefaultAction(self.toggleHistogramAction)
        # self.mainLayout.addWidget(self.histogramButton)

        self.setStyleSheet(
            "color: rgb(150,150,150);"
            "background-color: black;"
            "selection-color: white;"
            "selection-background-color: rgb(0,118,211);"
            # "selection-border-color: blue;"
            "border-style: outset;"
            "border-width: 1px;"
            "border-radius: 2px;"
            "border-color: grey;"
       )

    def finalize(self):
        """ Should be called manually before object deletion
        """
        logger.debug("Finalizing: {}".format(self))
        super(QRegnImagCtrlConfigWidget, self).finalize()

    def setLabels(self, labels):
        self.labels = labels
        self.selectImageComboBox.clear()
        self.selectImageComboBox.addItems(list(labels.values()))

    def selectImage(self, key):
        if not key in self.labels.keys():
            return
        self.selectImageComboBox.setCurrentText(self.labels[key])

    def setLevels(self, levels, autoscale=True):
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

        if not autoscale:
            self.isHistAutoLevel = False
            self.toggleHistAutoLevelAction.setChecked(False)

    def _triggerResetColorLevels(self, auto):
        if auto:
            self.colorBarItem.resetColorLevels()

    def _triggerSelectedImageChanged(self, label):
        for key, val in self.labels.items():
            if val == label:
                self.sigSelectedImageChanged.emit(key)
                break

    def _updateSpinBoxLevels(self, levels=[None, None]):
        """ Updates the spinboxes given the levels
        """
        if self.isHistAutoLevel:
            minLevel, maxLevel = levels
            logger.debug("_updateSpinBoxLevels: {}".format(levels))
            self.minLevelSpinBox.setValue(minLevel)
            self.maxLevelSpinBox.setValue(maxLevel)


class RegnImagCtrlItem(BaseImagCtrlItem):

    sigROIMaskChanged = QtCore.Signal(object, np.ndarray)

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

        # roi drawing functionality ...........................................
        self.kern_template = np.array([
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ])

        # radius = 5
        # kern_template = np.zeros((2 * radius + 1, 2 * radius + 1))
        # y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        # kern_template[x ** 2 + y ** 2 <= radius ** 2] = 1

        self.roimask = None
        self.roiGraphItem = pg.GraphItem()
        self.plotItem.addItem(self.roiGraphItem)

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

        self.toolbarWidget = QRegnImagCtrlConfigWidget(
            self.colorBarItem,
            label=label)
        self.toolbarProxy = QtWidgets.QGraphicsProxyWidget()
        self.toolbarProxy.setWidget(self.toolbarWidget)

        self.mainLayout = QtWidgets.QGraphicsGridLayout()
        self.setLayout(self.mainLayout)
        self.mainLayout.setContentsMargins(1, 1, 1, 1)
        self.mainLayout.setSpacing(10)

        # self.graphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self.mainLayout.addItem(self.toolbarProxy, 0, 0)
        self.mainLayout.addItem(self.plotItem, 1, 0)
        self.mainLayout.addItem(self.colorBarItem, 2, 0)

        self.toolbarWidget.sigSelectedImageChanged.connect(
            self.updateSelectedImage)

        self.imageItem.sigROISelectionFinished.connect(self.createROIMask)

    def createROIMask(self, imageItem, pts):
        image = imageItem.image

        if image.ndim == 2:
            nRows, nCols = image.shape
            nChan = 1
        elif image.ndim == 3:
            nRows, nCols, nChan = image.shape

        # image_contour = np.zeros([nRows, nCols], dtype=np.uint8)
        # image_contour[pts[:, 1], pts[:, 0]] = 255

        pts = np.clip(pts, [0, 0], [nCols, nRows])
        contours = [pts]

        # create mask by filling contour
        poly_path = Path(pts)
        # x, y = np.mgrid[:nRows, :nCols]
        # coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        x, y = np.mgrid[:nRows, :nCols]
        coors = np.hstack((y.reshape(-1, 1), x.reshape(-1, 1)))
        mask = poly_path.contains_points(coors)
        mask = mask.reshape(nRows, nCols)
        mask = np.asarray(mask, dtype=np.uint8)

        # mask2 = np.zeros((nRows, nCols), dtype=np.uint8)
        # cv2.fillPoly(mask2, pts=contours, color=1)  # set 1 within the contour

        self.roimask = mask

        rng = np.arange(0, len(pts))
        adj = np.column_stack([rng, np.roll(rng, 1)])
        pen = pg.mkPen(color=(255, 0, 255), width=2)
        self.roiGraphItem.setData(pos=pts, adj=adj, pen=pen, size=2,
                  pxMode=False, symbol=None)

        # emit signal to indicate roi update
        self.sigROIMaskChanged.emit(self, mask)

        # draw contour and mask for testing
        # import cv2
        # cv2.drawContours(image, contours, -1, (255, 0, 255), 3)

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # plt.imshow(image_contour, cmap="gray", origin='lower', vmin=0, vmax=1)
        # # plt.imshow(mask, cmap="gray", vmin=0, vmax=1)
        # plt.show()
        # plt.close
        #
        # fig = plt.figure()
        # # plt.imshow(image_mask2, cmap="gray", vmin=0, vmax=1)
        # plt.imshow(mask, cmap="gray", origin='lower', vmin=0, vmax=1)
        # plt.show()
        # plt.close
        #
        # fig = plt.figure()
        # # plt.imshow(image_mask2, cmap="gray", vmin=0, vmax=1)
        # plt.imshow(mask2, cmap="gray", origin='lower', vmin=0, vmax=1)
        # plt.show()
        # plt.close

    def onCursorHovered(self, ev, state):
        pass
    #     if state:
    #     self.imageItem.setDrawEnabled(False)
    #     self.imageItem.blockSignals(True)
    #     else:
    #         self.imageItem.setDrawEnabled(True)
    #         self.imageItem.blockSignals(True)

    def setData(self, data, labels=None):
        """ Sets the image data
        """
        super(RegnImagCtrlItem, self).setData(data, labels)
        self.toolbarWidget.setLabels(self.labels)

    def selectImage(self, key):
        """ Sets the image data
        """
        self.toolbarWidget.selectImage(key)

    def clearROIMask(self):
        logger.debug("Clear ROI mask.")
        self.roimask = None

        pos = np.array([[0, 0]])
        adj = np.array([[0, 0]])
        pen = pg.mkPen(color=(0, 0, 0), width=0)
        self.roiGraphItem.setData(pos=pos, adj=adj, pen=pen, pxMode=False, symbol=None)

    def updateSelectedImage(self, key):
        if self.data is None or not isinstance(
                self.data, dict) or not key in self.data.keys():
            return

        data = self.data[key].copy()
        self.selectedImage = data

        if data.ndim == 2:
            nRows, nCols = data.shape
            nChan = 1
            self.imageItem.setImage(data, axisOrder='row-major')
            brush = 2.5
            kern = self.kern_template * brush
            self.imageItem.setDrawKernel(kern, mask=kern, center=(1, 1),
                                         mode='add')
            self.imageItem.setLevels([0, 1])
            self.roimask = None
            self.imageItem.setDrawEnabled(True)
        elif data.ndim == 3:
            nRows, nCols, nChan = data.shape
            self.imageItem.setImage(data, axisOrder='row-major')
            brush = 2.5 * np.array([1., 0, 1.])
            kern = np.outer(self.kern_template, brush).reshape(
                self.kern_template.shape + (len(brush),))
            self.imageItem.setDrawKernel(kern, mask=kern, center=(1, 1),
                                         mode='add')
            self.imageItem.setLevels([[0, 1], [0, 1], [0, 1]])
            self.roimask = None
            self.imageItem.setDrawEnabled(True)
        else:
            raise Exception("Plot data must be 2D or 3D ndarray.")

        self.plotItem.setRange(xRange=[0, nCols], yRange=[0, nRows])

        if self.toolbarWidget.toggleHistAutoLevelAction.isChecked():
            self.colorBarItem.resetColorLevels()

        self.cursorX.setBounds((0, nCols-1))
        self.cursorY.setBounds((0, nRows-1))

    def setLevels(self, levels):
        self.colorBarItem.setLevels(levels)
