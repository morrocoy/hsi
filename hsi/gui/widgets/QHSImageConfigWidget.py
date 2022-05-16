import os

from ...bindings.Qt import QtWidgets, QtGui, QtCore
from ...log import logmanager
from ...core.hs_image import HSImage

from ...core.hs_formats import HSFormatFlag, HSFormatDefault
from ...core import hs_functions as fn

from .QParamRegionWidget import QParamRegionWidget

logger = logmanager.getLogger(__name__)


__all__ = ['QHSImageConfigWidget']


class QHSImageConfigWidget(QtWidgets.QWidget):
    """ Config widget for hyper spectral images
    """

    sigValueChanged = QtCore.Signal(object, bool)

    def __init__(self, *args, **kwargs):
        """ Constructor
        """
        parent = kwargs.get('parent', None)
        super(QHSImageConfigWidget, self).__init__(parent=parent)

        if len(args) == 1:
            kwargs['dir'] = args[0]
        elif len(args) > 1:
            raise TypeError("To many arguments {}".format(args))

        self.dir = kwargs.get('dir', None)
        self.filePath = None
        self.hsImage = HSImage()

        self._data = None  # internal storage for the spectral data
        self._rawValue = None  # internal storage for the spectral raw data

        self.loadButton = QtWidgets.QToolButton(self)
        self.resetButton = QtWidgets.QToolButton(self)
        self.filterButton = QtWidgets.QToolButton(self)

        self.fileLineEdit = QtGui.QLineEdit(self)
        self.spectSNVCheckBox = QtGui.QCheckBox(self)
        self.spectFormatComboBox = QtGui.QComboBox(self)
        self.maskThreshRegionWidget = QParamRegionWidget("mask", self)
        self.imageFilterTypeComboBox = QtGui.QComboBox(self)
        self.spectFilterTypeComboBox = QtGui.QComboBox(self)

        self.imageFilterSizeSpinBox = QtWidgets.QDoubleSpinBox(self)
        self.imageFilterSigmaSpinBox = QtWidgets.QDoubleSpinBox(self)
        self.spectFilterSizeSpinBox = QtWidgets.QDoubleSpinBox(self)
        self.spectFilterSigmaSpinBox = QtWidgets.QDoubleSpinBox(self)
        self.spectFilterOrderSpinBox = QtWidgets.QDoubleSpinBox(self)
        self.spectFilterDerivSpinBox = QtWidgets.QDoubleSpinBox(self)

        # configure actions
        self._setupActions()

        # configure widget views
        self._setupViews(*args, **kwargs)

        # connect signals
        self.spectFormatComboBox.currentTextChanged.connect(self.updateFormat)
        self.spectSNVCheckBox.stateChanged.connect(self.updateFormat)
        self.imageFilterTypeComboBox.currentTextChanged.connect(
            self._updateImageFilterSettings)
        self.spectFilterTypeComboBox.currentTextChanged.connect(
            self._updateSpectFilterSettings)

        # self.imageFilterSizeSpinBox.valueChanged.connect(self.updateFilter)
        # self.imageFilterSigmaSpinBox.valueChanged.connect(self.updateFilter)
        # self.spectFilterSizeSpinBox.valueChanged.connect(self.updateFilter)
        # self.spectFilterSigmaSpinBox.valueChanged.connect(self.updateFilter)
        # self.spectFilterOrderSpinBox.valueChanged.connect(self.updateFilter)
        # self.spectFilterDerivSpinBox.valueChanged.connect(self.updateFilter)


    def _setupActions(self):
        self.loadAction = QtWidgets.QAction(self)
        self.loadAction.setIconText("...")
        self.loadAction.triggered.connect(self.loadFile)
        self.addAction(self.loadAction)

        self.filterAction = QtWidgets.QAction("Update", self)
        self.filterAction.triggered.connect(self.updateFilter)
        self.addAction(self.filterAction)

        self.resetAction = QtWidgets.QAction("Reset", self)
        self.resetAction.triggered.connect(self.resetFilter)
        self.addAction(self.resetAction)

        self.loadButton.setDefaultAction(self.loadAction)
        self.resetButton.setDefaultAction(self.resetAction)
        self.filterButton.setDefaultAction(self.filterAction)


    def _setupViews(self, *args, **kwargs):
        # self.mainLayout = QtWidgets.QVBoxLayout()
        self.mainLayout = QtWidgets.QFormLayout()
        self.mainLayout.setContentsMargins(5, 10, 5, 10) # left, top, right, bottom
        self.mainLayout.setSpacing(3)
        self.setLayout(self.mainLayout)

        # file load ..........................................................
        label = QtGui.QLabel("Image file")
        label.setStyleSheet(
            "border: 0px;"
            "font: bold;"
        )
        label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.mainLayout.addRow(label)

        self.fileLineEdit.setReadOnly(True)
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.fileLineEdit)
        layout.addWidget(self.loadButton)
        self.mainLayout.addRow(layout)

        label = QtGui.QLabel("Format")
        label.setStyleSheet("border: 0px;")
        label.setIndent(5)
        label.setMinimumWidth(50)

        keys = [flag.key for flag in HSFormatFlag.get_flags()]  # available fmts
        self.spectFormatComboBox.addItems(keys)
        self.spectFormatComboBox.setCurrentText(HSFormatDefault.key)
        self.spectFormatComboBox.setMinimumWidth(90)
        self.spectSNVCheckBox.setChecked(False)
        self.spectSNVCheckBox.setText("SNV")

        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.spectFormatComboBox)
        layout.addStretch()
        layout.addWidget(self.spectSNVCheckBox)
        self.mainLayout.addRow(label, layout)

        self.maskThreshRegionWidget.setLabel("Mask Th.")
        self.maskThreshRegionWidget.setDecimals(2)
        self.maskThreshRegionWidget.setSingleStep(0.1)
        self.maskThreshRegionWidget.setBounds([0., 1.])
        self.maskThreshRegionWidget.setValueDefault([0.1, 0.9])
        self.maskThreshRegionWidget.setValue([0.1, 0.9])
        self.mainLayout.addRow(self.maskThreshRegionWidget)

        # image filter .......................................................
        label = QtGui.QLabel("Image smoothening filter")
        label.setStyleSheet(
            "border: 0px;"
            "font: bold;"
        )
        label.setMinimumHeight(20)
        label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.mainLayout.addRow(label)

        label = QtGui.QLabel("Type")
        label.setStyleSheet("border: 0px;")
        label.setIndent(5)
        label.setMinimumWidth(50)
        self.imageFilterTypeComboBox.addItems(
            ['none', 'gauss', 'mean', 'median'])
        self.imageFilterTypeComboBox.setCurrentIndex(1)
        self.mainLayout.addRow(label, self.imageFilterTypeComboBox)

        label = QtGui.QLabel("Size")
        label.setStyleSheet("border: 0px;")
        label.setIndent(5)
        label.setMinimumWidth(50)
        self.imageFilterSizeSpinBox.setKeyboardTracking(False)
        self.imageFilterSizeSpinBox.setMinimum(0)
        self.imageFilterSizeSpinBox.setMaximum(50)
        self.imageFilterSizeSpinBox.setSingleStep(1)
        self.imageFilterSizeSpinBox.setDecimals(0)
        self.imageFilterSizeSpinBox.setValue(5)
        self.imageFilterSizeSpinBox.setEnabled(True)
        self.mainLayout.addRow(label, self.imageFilterSizeSpinBox)

        label = QtGui.QLabel("Sigma")
        label.setStyleSheet("border: 0px;")
        label.setIndent(5)
        label.setMinimumWidth(50)
        self.imageFilterSigmaSpinBox.setKeyboardTracking(False)
        self.imageFilterSigmaSpinBox.setMinimum(0)
        self.imageFilterSigmaSpinBox.setMaximum(10)
        self.imageFilterSigmaSpinBox.setSingleStep(0.1)
        self.imageFilterSigmaSpinBox.setDecimals(2)
        self.imageFilterSigmaSpinBox.setValue(1)
        self.imageFilterSigmaSpinBox.setEnabled(True)
        self.mainLayout.addRow(label, self.imageFilterSigmaSpinBox)

        # spectral attenuation filter ........................................
        label = QtGui.QLabel("Spectral attenuation filter")
        label.setStyleSheet(
            "border: 0px;"
            "font: bold;"
        )
        label.setMinimumHeight(20)
        label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.mainLayout.addRow(label)

        label = QtGui.QLabel("Type")
        label.setStyleSheet("border: 0px;")
        label.setIndent(5)
        label.setMinimumWidth(50)
        self.spectFilterTypeComboBox.addItems(
            ['none', 'gauss', 'mean', 'median', 'savgol'])
        self.spectFilterTypeComboBox.setCurrentIndex(0)
        self.mainLayout.addRow(label, self.spectFilterTypeComboBox)

        label = QtGui.QLabel("Size")
        label.setStyleSheet("border: 0px;")
        label.setIndent(5)
        label.setMinimumWidth(50)
        self.spectFilterSizeSpinBox.setKeyboardTracking(False)
        self.spectFilterSizeSpinBox.setMinimum(3)
        self.spectFilterSizeSpinBox.setMaximum(50)
        self.spectFilterSizeSpinBox.setSingleStep(1)
        self.spectFilterSizeSpinBox.setDecimals(0)
        self.spectFilterSizeSpinBox.setValue(7)
        self.spectFilterSizeSpinBox.setEnabled(False)
        self.mainLayout.addRow(label, self.spectFilterSizeSpinBox)

        label = QtGui.QLabel("Sigma")
        label.setStyleSheet("border: 0px;")
        label.setIndent(5)
        label.setMinimumWidth(50)
        self.spectFilterSigmaSpinBox.setKeyboardTracking(False)
        self.spectFilterSigmaSpinBox.setMinimum(0)
        self.spectFilterSigmaSpinBox.setMaximum(10)
        self.spectFilterSigmaSpinBox.setSingleStep(0.1)
        self.spectFilterSigmaSpinBox.setDecimals(2)
        self.spectFilterSigmaSpinBox.setValue(1)
        self.spectFilterSigmaSpinBox.setEnabled(False)
        self.mainLayout.addRow(label, self.spectFilterSigmaSpinBox)

        label = QtGui.QLabel("Order")
        label.setStyleSheet("border: 0px;")
        label.setIndent(5)
        label.setMinimumWidth(50)
        self.spectFilterOrderSpinBox.setKeyboardTracking(False)
        self.spectFilterOrderSpinBox.setMinimum(2)
        self.spectFilterOrderSpinBox.setMaximum(10)
        self.spectFilterOrderSpinBox.setSingleStep(1)
        self.spectFilterOrderSpinBox.setDecimals(0)
        self.spectFilterOrderSpinBox.setValue(2)
        self.spectFilterOrderSpinBox.setEnabled(False)
        self.mainLayout.addRow(label, self.spectFilterOrderSpinBox)

        label = QtGui.QLabel("Deriv")
        label.setStyleSheet("border: 0px;")
        label.setIndent(5)
        label.setMinimumWidth(50)
        self.spectFilterDerivSpinBox.setKeyboardTracking(False)
        self.spectFilterDerivSpinBox.setMinimum(0)
        self.spectFilterDerivSpinBox.setMaximum(2)
        self.spectFilterDerivSpinBox.setSingleStep(1)
        self.spectFilterDerivSpinBox.setDecimals(0)
        self.spectFilterDerivSpinBox.setValue(0)
        self.spectFilterDerivSpinBox.setEnabled(False)
        self.mainLayout.addRow(label, self.spectFilterDerivSpinBox)

        # filter and reset controls
        label = QtGui.QLabel(self)
        label.setMinimumHeight(20)
        label.setStyleSheet("border: 0px;")
        self.filterButton.setText("Update")
        self.resetButton.setText("Reset")

        layout = QtGui.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.resetButton)
        layout.addWidget(self.filterButton)
        self.mainLayout.addRow(layout)


    def _updateImageFilterSettings(self, type):
        """Disables and enables individual options for the image filter
        """
        logger.debug("Change image filter type to '{}'".format(type))

        if type == 'gauss':
            self.imageFilterSizeSpinBox.setEnabled(True)
            self.imageFilterSigmaSpinBox.setEnabled(True)
        elif type == 'mean':
            self.imageFilterSizeSpinBox.setEnabled(True)
            self.imageFilterSigmaSpinBox.setEnabled(False)
        elif type == 'median':
            self.imageFilterSizeSpinBox.setEnabled(True)
            self.imageFilterSigmaSpinBox.setEnabled(False)
        else:
            self.imageFilterSizeSpinBox.setEnabled(False)
            self.imageFilterSigmaSpinBox.setEnabled(False)


    def _updateSpectFilterSettings(self, type):
        """Disables and enables individual options for the spectral filter
        """
        logger.debug("Change spectral filter type to '{}'".format(type))

        if type == 'gauss':
            self.spectFilterSizeSpinBox.setEnabled(True)
            self.spectFilterSigmaSpinBox.setEnabled(True)
            self.spectFilterOrderSpinBox.setEnabled(False)
            self.spectFilterDerivSpinBox.setEnabled(False)
        elif type == 'mean':
            self.spectFilterSizeSpinBox.setEnabled(True)
            self.spectFilterSigmaSpinBox.setEnabled(False)
            self.spectFilterOrderSpinBox.setEnabled(False)
            self.spectFilterDerivSpinBox.setEnabled(False)
        elif type == 'median':
            self.spectFilterSizeSpinBox.setEnabled(True)
            self.spectFilterSigmaSpinBox.setEnabled(False)
            self.spectFilterOrderSpinBox.setEnabled(False)
            self.spectFilterDerivSpinBox.setEnabled(False)
        elif type == 'savgol':
            self.spectFilterSizeSpinBox.setEnabled(True)
            self.spectFilterSigmaSpinBox.setEnabled(False)
            self.spectFilterOrderSpinBox.setEnabled(True)
            self.spectFilterDerivSpinBox.setEnabled(True)
        else:
            self.spectFilterSizeSpinBox.setEnabled(False)
            self.spectFilterSigmaSpinBox.setEnabled(False)
            self.spectFilterOrderSpinBox.setEnabled(False)
            self.spectFilterDerivSpinBox.setEnabled(False)


    def finalize(self):
        """ Should be called manually before object deletion
        """
        logger.debug("Finalizing: {}".format(self))
        super(QHSImageConfigWidget, self).finalize()


    def isEmpty(self):

        if self.hsImage.shape is None:
            return True
        else:
            return False


    def loadFile(self):
        """Load hyper spectral image file using a dialog box
        """
        filter = "data cube (*.dat)"
        filePath, filter = QtGui.QFileDialog.getOpenFileName(
            None, 'Select file:', self.dir, filter)

        if not os.path.isfile(filePath):
            return

        logger.debug("Load hyper spectral image file: {}".format(filePath))

        self.filePath = filePath
        self.fileLineEdit.setText(filePath)

        self.hsImage.load(self.filePath)
        self.updateFilter(newFile=True)


    def resetFilter(self):
        """Set and apply the default filter configuration
        """
        self.imageFilterTypeComboBox.setCurrentIndex(1)
        self.imageFilterSizeSpinBox.setValue(4)
        self.imageFilterSigmaSpinBox.setValue(1)

        self.spectFilterTypeComboBox.setCurrentIndex(0)
        self.spectFilterSizeSpinBox.setValue(7)
        self.spectFilterSigmaSpinBox.setValue(1)
        self.spectFilterOrderSpinBox.setValue(2)
        self.spectFilterDerivSpinBox.setValue(0)

        self.updateFilter()


    def updateFilter(self, newFile=False):
        """Apply current filter settings to hyper spectral image
        """
        if self.hsImage.shape is None:
            return

        self.hsImage.clear_filter()

        mode = 'image'
        type =  self.imageFilterTypeComboBox.currentText()
        if type != 'none':
            size = int(self.imageFilterSizeSpinBox.value())
            sigma = self.imageFilterSigmaSpinBox.value()
            self.hsImage.add_filter(mode, type, size=size, sigma=sigma)

            logger.debug("Update image filter with arguments {}"
                         .format((type, size, sigma)))

        mode = 'spectra'
        type = self.spectFilterTypeComboBox.currentText()
        if type != 'none':
            size = int(self.spectFilterSizeSpinBox.value())
            sigma = self.spectFilterSigmaSpinBox.value()
            order = int(self.spectFilterOrderSpinBox.value())
            deriv = int(self.spectFilterDerivSpinBox.value())
            self.hsImage.add_filter(mode, type, size=size, sigma=sigma,
                                    order=order, deriv=deriv)

            logger.debug("Update spectral filter with arguments {}"
                         .format((type, size, sigma, order, deriv)))

        self.sigValueChanged.emit(self, newFile)


    def updateFormat(self):
        """Retrieve spectral data according to the current hsformat setting
        """
        sformat = self.spectFormatComboBox.currentText()
        format = HSFormatFlag.from_str(sformat)
        self.hsImage.set_format(format)

        logger.debug("Change spectral hsformat to '{}'.".format(sformat))

        if self.hsImage.shape is not None:
            self.sigValueChanged.emit(self, False)

    def getMask(self):
        thresh = self.maskThreshRegionWidget.value()
        return self.hsImage.get_tissue_mask(thresholds=thresh)

    def getImage(self, *args, **kwargs):
        """Retrieve the rgb image from hyperspectral data.

        Forwards all arguments to
        :func:`as_rgb <hsi.core.HSImage.as_rgb>`.
        """
        return self.hsImage.as_rgb(*args, **kwargs)


    def getSpectra(self, filter=True):
        """Get the filtered or unfiltered hyperspectral data of the image.

        Parameters
        ----------
        filter : boolean
            A flag to select between filtered or unfiltered data
        """
        norm = self.spectSNVCheckBox.isChecked()
        if filter and norm:
            return fn.snv(self.hsImage.fspectra)
        elif not filter and norm:
            return fn.snv(self.hsImage.spectra)
        elif filter and not norm:
            return self.hsImage.fspectra
        else:
            return self.hsImage.spectra


    def getWavelen(self):
        """Get the wavelength axis."""
        return self.hsImage.wavelen


    def getFormat(self):
        """Get the hsformat of hyperspectral data."""
        return self.hsImage.hsformat


    def setFormat(self, format):
        """Set the hsformat for the hyperspectral data.

        Parameters
        ----------
        format :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The hsformat for the hyperspectral data. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`


        """
        # check hsformat, if not previously defined also set the hsformat
        if not HSFormatFlag.has_flag(format):
            logger.debug("Unknown hsformat '{}'.".format(format))
            return

        self.spectFormatComboBox.setCurrentText(format.key)