import sys

import numpy as np
import pyqtgraph as pg

from ...bindings.Qt import QtWidgets, QtCore
from ...log import logmanager
from ...misc import check_is_an_array, check_class

logger = logmanager.getLogger(__name__)

__all__ = ['RegnPlotCtrlItem']


class QRegnPlotCtrlConfigWidget(QtWidgets.QWidget):
    """ Config widget with two spinboxes that control the region limits.
    """
    def __init__(self, *args, **kwargs):
        """ Constructor
        """
        parent = kwargs.get('parent', None)
        super(QRegnPlotCtrlConfigWidget, self).__init__(parent=parent)

        if len(args) == 1:
            regionItem = args[0]
        elif len(args) == 2:
            regionItem = args[0]
            kwargs['label'] = args[1]
        else:
            raise TypeError("Unexpected number of arguments {}".format(args))

        check_class(regionItem, pg.LinearRegionItem)

        self.regionItem = regionItem

        if 'label' in kwargs:
            self.label = QtWidgets.QLabel(kwargs.get('label'))
        else:
            self.label = None
        self.minLevelSpinBox = QtWidgets.QDoubleSpinBox()
        self.maxLevelSpinBox = QtWidgets.QDoubleSpinBox()
        self.resetButton = QtWidgets.QToolButton()

        # configure actions
        self._setupActions()

        # configure widget views
        self._setupViews()

        # connect signals
        self.minLevelSpinBox.valueChanged.connect(lambda val: self.setLimits((val, None)))
        self.maxLevelSpinBox.valueChanged.connect(lambda val: self.setLimits((None, val)))
        self.regionItem.sigRegionChanged.connect(self._updateSpinBoxLimits)
        # self.regionItem.sigRegionChangeFinished.connect(self._updateSpinBoxLimits)


    def _setupActions(self):
        self.resetAction = QtWidgets.QAction("reset", self)
        self.resetAction.triggered.connect(self.resetRegion)
        # self.resetAction.setShortcut("Ctrl+0")
        self.addAction(self.resetAction)


    def _setupViews(self):
        self.mainLayout = QtWidgets.QHBoxLayout()
        self.mainLayout.setContentsMargins(5, 0, 5, 0) # left, top, right, bottom
        self.mainLayout.setSpacing(3)
        self.setLayout(self.mainLayout)

        if self.label is not None:
            self.label.setStyleSheet(
                "border-color: black;"
                "font: bold 14px;"
            )
            self.mainLayout.addStretch()
            self.mainLayout.addWidget(self.label)


        self.mainLayout.addStretch()
        self.label = QtWidgets.QLabel("limits")
        self.label.setStyleSheet("border-color: black;")
        self.mainLayout.addWidget(self.label)

        self.minLevelSpinBox.setKeyboardTracking(False)
        self.minLevelSpinBox.setMinimum(-1000)
        self.minLevelSpinBox.setMaximum(1000)
        self.minLevelSpinBox.setSingleStep(0.1)
        self.minLevelSpinBox.setDecimals(3)
        self.minLevelSpinBox.setMaximumWidth(60)
        self.mainLayout.addWidget(self.minLevelSpinBox)

        self.maxLevelSpinBox.setKeyboardTracking(False)
        self.maxLevelSpinBox.setMinimum(-1000)
        self.maxLevelSpinBox.setMaximum(1000)
        self.maxLevelSpinBox.setSingleStep(0.1)
        self.maxLevelSpinBox.setDecimals(3)
        self.maxLevelSpinBox.setMaximumWidth(60)
        self.mainLayout.addWidget(self.maxLevelSpinBox)

        self.resetButton.setDefaultAction(self.resetAction)
        self.mainLayout.addWidget(self.resetButton)

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


    def _updateSpinBoxLimits(self):
        """ Updates the spinboxes given the levels
        """
        limits = self.regionItem.getRegion()
        logger.debug("_updateSpinBoxLimits: {}".format(limits))

        self.minLevelSpinBox.blockSignals(True)
        self.maxLevelSpinBox.blockSignals(True)
        self.minLevelSpinBox.setValue(limits[0])
        self.maxLevelSpinBox.setValue(limits[1])
        self.minLevelSpinBox.blockSignals(False)
        self.maxLevelSpinBox.blockSignals(False)


    def finalize(self):
        """ Should be called manually before object deletion
        """
        logger.debug("Finalizing: {}".format(self))
        super(QRegnPlotCtrlConfigWidget, self).finalize()


    def resetRegion(self):
        limits = self.regionItem.lines[0].maxRange
        self.regionItem.setRegion(limits)


    def setLimits(self, limits):
        """ Sets plot levels
            :param limits: (vMin, vMax) tuple
        """
        logger.debug("Setting image levels: {}".format(limits))
        minLevel, maxLevel = limits

        # Replace Nones by the current level
        oldMin, oldMax = self.regionItem.getRegion()
        logger.debug("Old levels: {}".format(limits))

        if minLevel is None: # Only maxLevel was set.
            minLevel = oldMin
            if maxLevel <= minLevel:
                minLevel = maxLevel - 1

        if maxLevel is None: # Only minLevel was set
            maxLevel = oldMax
            if maxLevel <= minLevel:
                maxLevel = minLevel + 1

        self.regionItem.setRegion((minLevel, maxLevel))




class RegnPlotCtrlItem(pg.GraphicsWidget):

    sigPlotChanged = QtCore.Signal(object)
    sigRegionChanged = QtCore.Signal(object)
    sigRegionChangeFinished = QtCore.Signal(object)


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
        self.plotItem2 = pg.PlotItem()
        self.regionItem = pg.LinearRegionItem()
        self.toolbarWidget = QRegnPlotCtrlConfigWidget(self.regionItem, label)


        self._setupActions()
        self._setupViews(**kwargs)

        # Connect signals
        self.regionItem.sigRegionChanged.connect(self.regionChangeEvent)
        # self.regionItem.sigRegionChangeFinished.connect(self.regionChangeEvent)
        self.regionItem.sigRegionChangeFinished.connect(self.regionChangeFinishedEvent)
        self.plotItem2.sigRangeChanged.connect(self.rangeChangeEvent)


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


        self.regionItem.setZValue(10)
        self.plotItem1.addItem(self.regionItem, ignoreBounds=True)
        # self.plotItem2.setAutoVisible(y=True)

        if xlabel is not None:
            self.plotItem1.setLabel('bottom', xlabel, xunits)
            self.plotItem2.setLabel('bottom', xlabel, xunits)
        if ylabel is not None:
            self.plotItem1.setLabel('left', ylabel, yunits)
            self.plotItem2.setLabel('left', ylabel, yunits)

        # self.plotItem1.setLabel('bottom', "test")

        self.toolbarProxy = QtWidgets.QGraphicsProxyWidget()
        self.toolbarProxy.setWidget(self.toolbarWidget)

        self.mainLayout = QtWidgets.QGraphicsGridLayout()
        self.setLayout(self.mainLayout)
        self.mainLayout.setContentsMargins(1, 1, 1, 1)
        self.mainLayout.setSpacing(0)

        self.mainLayout.addItem(self.toolbarProxy, 0, 0)
        self.mainLayout.addItem(self.plotItem2, 1, 0)
        self.mainLayout.addItem(self.plotItem1, 2, 0)

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
        self.plotItem2.addItem(shadowItem)

        self.curveItems.append(item)
        self.shadowCurveItems.append(shadowItem)
        self.updateBounds()

        item.sigPlotChanged.connect(self.plotChangedEvent)

    def getRegion(self):
        xmin, xmax = self.regionItem.getRegion()
        return xmin , xmax

    def plotChangedEvent(self, sender):
        for item, shadowItem in zip(self.curveItems, self.shadowCurveItems):
            if item is sender:
                x, y = item.getData()
                shadowItem.setData(x=x, y=y)

        self.updateBounds()
        self.sigPlotChanged.emit(sender)

    def rangeChangeEvent(self, window, viewRange):
        rgn = viewRange[0]
        self.regionItem.setRegion(rgn)

    def regionChangeEvent(self):
        self.regionItem.setZValue(10)
        minX, maxX = self.regionItem.getRegion()
        self.plotItem2.setXRange(minX, maxX, padding=0)

        self.sigRegionChanged.emit(self)

    def regionChangeFinishedEvent(self):
        self.sigRegionChangeFinished.emit(self)

    def updateBounds(self):
        if len(self.curveItems):
            min = []
            max = []

            for i, item in enumerate(self.curveItems):
                x, y = item.getData()
                if len(x):
                    min.append(np.nanmin(x))
                    max.append(np.nanmax(x))

            if len(min):
                min = np.array(min)
                max = np.array(max)
                limits = (np.nanmin(min), np.nanmax(max))
            else:
                limits = (0, 1)
        else:
            limits = (0, 1)

        self.setBounds(limits)

    def setBounds(self, limits):
        lbnd = limits[0]
        ubnd = limits[1]
        self.regionItem.setBounds([lbnd, ubnd])

    def setRegion(self, limits):
        xmin = limits[0]
        xmax = limits[1]
        self.regionItem.setRegion([xmin, xmax])


class DemoWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(DemoWindow, self).__init__(parent=parent)

        self._setupViews()

    def _setupViews(self):
        """ Creates the UI widgets.
        """
        self.mainWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.mainWidget)

        self.mainLayout = QtWidgets.QVBoxLayout()
        self.mainLayout.setContentsMargins(0, 0, 0, 0) # left, top, right, bottom
        self.mainLayout.setSpacing(0)
        self.mainWidget.setLayout(self.mainLayout)

        x = np.linspace(0, 100, 1000)
        y1 = pg.gaussianFilter(np.random.random(size=1000), 10) * 20
        y2 = pg.gaussianFilter(np.random.random(size=1000), 10) * 20
        y3 = pg.gaussianFilter(np.random.random(size=1000), 10) * 20

        self.controlItem1 = RegnPlotCtrlItem(label="Spectrum")

        self.crv1 = pg.PlotCurveItem(pen=pg.mkPen(color=(255, 0, 0, 255), width=1))
        self.crv2 = pg.PlotCurveItem(pen=pg.mkPen(color=(0, 255, 0, 255), width=1))
        self.crv3 = pg.PlotCurveItem(pen=pg.mkPen(color=(0, 0, 255, 255), width=1))
        self.crv1.setData(x=x, y=y1)
        self.crv2.setData(x=x, y=y2)
        self.crv3.setData(x=x, y=y3)

        self.controlItem1.addItem(self.crv1)
        self.controlItem1.addItem(self.crv2)
        self.controlItem1.addItem(self.crv3)

        self.graphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self.graphicsLayoutWidget.addItem(self.controlItem1, 0, 0)

        self.mainLayout.addWidget(self.graphicsLayoutWidget)
