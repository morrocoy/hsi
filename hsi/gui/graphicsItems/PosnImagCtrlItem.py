import sys
import numpy as np
import pyqtgraph as pg

from ...bindings.Qt import QtWidgets, QtCore
from ...log import logmanager
from ...misc import check_is_an_array, check_class
from ...core.hs_cm import cm

from .BaseImagCtrlItem import BaseImagCtrlItem
from .ColorBarItem import ColorBarItem

logger = logmanager.getLogger(__name__)

__all__ = ['PosnImagCtrlItem']


class QPosnImagCtrlConfigWidget(QtWidgets.QWidget):

    sigToggleHistogramChanged = QtCore.Signal(object)
    sigSelectedImageChanged = QtCore.Signal(object)

    """ Config widget with two spinboxes that control the image levels.
    """
    def __init__(self, controlItem, label=None, parent=None):
        """ Constructor
        """
        super(QPosnImagCtrlConfigWidget, self).__init__(parent=parent)

        self.controlItem = controlItem
        self._setupActions()
        self._setupViews(label)


    def _setupActions(self):
        pass


    def _setupViews(self, label, labels=None,  **kwargs):

        self.mainLayout = QtWidgets.QHBoxLayout()
        self.mainLayout.setContentsMargins(5, 0, 5, 0) # left, top, right, bottom
        self.mainLayout.setSpacing(3)
        self.setLayout(self.mainLayout)

        if labels is None:
            self.labels = {}

        self.selectImageComboBox = QtWidgets.QComboBox(self)
        for l in self.labels.values():
            self.selectImageComboBox.addItems(l)
        if len(self.labels):
            self.selectImageComboBox.setCurrentIndex(0)
        # self.selectImageComboBox.setMinimumWidth(120)
        self.selectImageComboBox.setMinimumWidth(150)
        self.mainLayout.addWidget(self.selectImageComboBox)
        self.mainLayout.addStretch()

        # if label is None:
        #     self.label = None
        # else:
        #     self.label = QtWidgets.QLabel(label)
        #     self.label.setStyleSheet(
        #         "border-color: black;"
        #         "font: bold 14px;"
        #     )
        #     self.label.setMinimumWidth(160)
        #     self.mainLayout.addStretch()
        #     self.mainLayout.addWidget(self.label)
        #
        # self.mainLayout.addStretch()
        # self.label = QtWidgets.QLabel("Position")
        # self.label.setStyleSheet("border-color: black;")
        self.label = QtWidgets.QLabel("+", self)
        self.label.setStyleSheet("border-color: black; font: 14px;")
        self.mainLayout.addWidget(self.label)


        self.cursorXSpinBox = QtWidgets.QDoubleSpinBox()
        self.cursorXSpinBox.setKeyboardTracking(False)
        self.cursorXSpinBox.setMinimum(0)
        self.cursorXSpinBox.setMaximum(1000)
        self.cursorXSpinBox.setSingleStep(1)
        self.cursorXSpinBox.setDecimals(0)
        self.cursorXSpinBox.setMinimumWidth(60)
        self.cursorXSpinBox.setMaximumWidth(60)
        self.mainLayout.addWidget(self.cursorXSpinBox)

        self.cursorYSpinBox = QtWidgets.QDoubleSpinBox()
        self.cursorYSpinBox.setKeyboardTracking(False)
        self.cursorYSpinBox.setMinimum(0)
        self.cursorYSpinBox.setMaximum(1000)
        self.cursorYSpinBox.setSingleStep(1)
        self.cursorYSpinBox.setDecimals(0)
        self.cursorYSpinBox.setMinimumWidth(60)
        self.cursorYSpinBox.setMaximumWidth(60)
        self.mainLayout.addWidget(self.cursorYSpinBox)

        # connect signals
        self.cursorXSpinBox.valueChanged.connect(self.setCursorPos)
        self.cursorYSpinBox.valueChanged.connect(self.setCursorPos)
        self.controlItem.sigCursorPositionChanged.connect(self._updateSpinBoxPosition)
        self.selectImageComboBox.currentTextChanged.connect(
            self._triggerSelectedImageChanged)

        # self.resetButton = QtWidgets.QToolButton()
        # self.resetButton.setDefaultAction(self.resetAction)
        # self.mainLayout.addWidget(self.resetButton)
        #
        # self.histogramButton = QtWidgets.QToolButton()
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
        super(QPosnImagCtrlConfigWidget, self).finalize()

    def setLabels(self, labels):
        self.labels = labels
        self.selectImageComboBox.clear()
        self.selectImageComboBox.addItems(list(labels.values()))

    def selectImage(self, key):
        if not key in self.labels.keys():
            return
        self.selectImageComboBox.setCurrentText(self.labels[key])

    def setCursorPos(self, pos):
        """ Sets plot levels
            :param levels: (x, y) tuple
        """
        logger.debug("Setting cursor position: {}".format(pos))

        x = self.cursorXSpinBox.value()
        y = self.cursorYSpinBox.value()

        self.controlItem.blockSignals(True)
        self.controlItem.cursorX.setPos(x)
        self.controlItem.cursorY.setPos(y)
        self.controlItem.blockSignals(False)

    def _triggerSelectedImageChanged(self, label):
        for key, val in self.labels.items():
            if val == label:
                self.sigSelectedImageChanged.emit(key)
                break

    def _updateSpinBoxPosition(self):
        """ Updates the spinboxes given the levels
        """
        x = self.controlItem.cursorX.getXPos()
        y = self.controlItem.cursorY.getYPos()

        # logger.debug("_updateSpinBoxPosition: {}".hsformat(tuple(x,y)))

        self.cursorXSpinBox.blockSignals(True)
        self.cursorYSpinBox.blockSignals(True)
        self.cursorXSpinBox.setValue(x)
        self.cursorYSpinBox.setValue(y)
        self.cursorXSpinBox.blockSignals(False)
        self.cursorYSpinBox.blockSignals(False)


class PosnImagCtrlItem(BaseImagCtrlItem):

    def __init__(self,
                 label=None,
                 cmap=None,
                 showHistogram=True,
                 cbarOrientation='bottom',
                 cbarWidth=20,
                 cbarHistogramHeight=30,
                 parent=None):
        BaseImagCtrlItem.__init__(self)

        self.cbarWidth = cbarWidth
        self.cbarHistogramHeight = cbarHistogramHeight
        self.cbarOrientation = cbarOrientation
        self.cbarHistogramIsVisible = showHistogram

        if cmap is None:
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
        pass

    def _setupViews(self, label):
        self.colorBarItem = ColorBarItem(
            imageItem=self.imageItem,
            showHistogram=self.cbarHistogramIsVisible,
            width=self.cbarWidth,
            orientation=self.cbarOrientation)
        self.colorBarItem.setMinimumHeight(60)

        self.toolbarWidget = QPosnImagCtrlConfigWidget(self, label=label)
        self.toolbarProxy = QtWidgets.QGraphicsProxyWidget()
        self.toolbarProxy.setWidget(self.toolbarWidget)

        self.mainLayout = QtWidgets.QGraphicsGridLayout()
        self.setLayout(self.mainLayout)
        self.mainLayout.setContentsMargins(1, 1, 1, 1)
        self.mainLayout.setSpacing(10)
        # self.mainLayout.setVerticalSpacing()
        self.mainLayout.addItem(self.toolbarProxy, 0, 0)
        self.mainLayout.addItem(self.plotItem, 1, 0)
        # self.mainLayout.addStretch()
        self.mainLayout.addItem(self.colorBarItem, 2, 0)

        self.toolbarWidget.sigSelectedImageChanged.connect(
            self.updateSelectedImage)

    def currentImage(self):
        label = self.toolbarWidget.selectImageComboBox.currentText()
        for key, val in self.labels.items():
            if val == label:
                return key

    def selectImage(self, key):
        """ Sets the image data
        """
        self.toolbarWidget.selectImage(key)

    def setData(self, data, labels=None):
        """ Sets the image data
        """
        super(PosnImagCtrlItem, self).setData(data, labels)
        self.toolbarWidget.setLabels(self.labels)

    def updateSelectedImage(self, key):
        if self.data is None or not isinstance(
                self.data, dict) or not key in self.data.keys():
            return

        data = self.data[key]
        self.selectedImage = data

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

        self.cursorX.setBounds((0, nCols-1))
        self.cursorY.setBounds((0, nRows-1))


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

        img1 = pg.gaussianFilter(np.random.normal(size=(300, 200)), (5, 5)) * 20
        img2 = pg.gaussianFilter(np.random.normal(size=(300, 200)), (5, 5)) * 20

        self.controlItem1 = PosnImagCtrlItem(label="ImageCoordControlItem1")
        self.controlItem1.setData(img1)
        self.controlItem2 = PosnImagCtrlItem(label="ImageCoordControlItem2")
        self.controlItem2.setData(img2)

        self.controlItem1.setXYLink(self.controlItem2)

        self.graphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self.graphicsLayoutWidget.addItem(self.controlItem1, 0, 0)
        self.graphicsLayoutWidget.addItem(self.controlItem2, 0, 1)


        self.mainLayout.addWidget(self.graphicsLayoutWidget)


def main():

    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python version: {}".format(sys.version))
    logger.info("PyQt bindings: {}".format(pg.Qt.QT_LIB))
    logger.info("PyQtGraph version: {}".format(pg.__version__))

    app = QtWidgets.QApplication([])

    win = DemoWindow()
    win.setGeometry(400, 100, 800, 500)
    win.setWindowTitle('PgColorbar Demo')
    win.show()
    app.exec_()


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s %(filename)25s:%(lineno)-4d : %(levelname)-7s: %(message)s'
    logging.basicConfig(level='DEBUG', format=LOG_FMT)

    main()