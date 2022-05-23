import os.path
from timeit import default_timer as timer

from ... import CONFIG_OPTIONS

from ...bindings.Qt import QtWidgets, QtCore
from ...log import logmanager
from ...misc import getPkgDir

from ...analysis.hs_cofit import HSCoFit

from .QParamRegionWidget import QParamRegionWidget

logger = logmanager.getLogger(__name__)

__all__ = ['QHSCoFitConfigWidget']


if CONFIG_OPTIONS['enableBVLS']:
    _methods = (
        ("bvls", "Bounded value least square algorithm"),
        ("bvls_f", "Bounded value least square algorithm (fortran)."),
        ("gesv", "Linear matrix equation (unconstrained)."),
        ("lstsq", "Least square algorithm (unconstrained)."),
        ("nnls", "Non-negative least squares."),
        ("trf", "Trust region reflective algorithm."),
        ("cg", "Conjugate gradient algorithm (unconstrained)."),
        ("l-bfgs-b", "Constrained BFGS algorithm."),
        ("nelder-mead", "Nelder-Mead algorithm (unconstrained)."),
        ("powell", "Powell algorithm."),
        ("slsqp", "Sequential least squares Programming."),
        ("tnc", "Truncated Newton (TNC) algorithm."),
    )
else:
    _methods = (
        ("bvls", "Bounded value least square algorithm"),
        ("gesv", "Linear matrix equation (unconstrained)."),
        ("lstsq", "Least square algorithm (unconstrained)."),
        ("nnls", "Non-negative least squares."),
        ("trf", "Trust region reflective algorithm."),
        ("cg", "Conjugate gradient algorithm (unconstrained)."),
        ("l-bfgs-b", "Constrained BFGS algorithm."),
        ("nelder-mead", "Nelder-Mead algorithm (unconstrained)."),
        ("powell", "Powell algorithm."),
        ("slsqp", "Sequential least squares Programming."),
        ("tnc", "Truncated Newton (TNC) algorithm."),
    )



class QHSCoFitConfigWidget(QtWidgets.QWidget):
    """ Config widget for hyper spectral images

    Attributes
    ----------
    hsformat :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
        The output hsformat for the hyperspectral data. Should be one of:

            - :class:`HSIntensity<hsi.HSIntensity>`
            - :class:`HSAbsorption<hsi.HSAbsorption>`
            - :class:`HSExtinction<hsi.HSExtinction>`
            - :class:`HSRefraction<hsi.HSRefraction>`

    filePath :  str
        The input file providing the base vectors.
    hsVectorAnalysis :  HSCoFit
        A wrapper for the component fit analysis.
    testMask : numpy.ndarray
        An image mask applied for test pixel-wise analysis.
    mask : numpy.ndarray
        An image mask applied for the complete image analysis.
    nroi : int
        The number of regions of interest.
    """
    sigValueChanged = QtCore.Signal(object, bool)

    def __init__(self, *args, **kwargs):
        """ Constructor
        """
        if len(args) == 1:
            kwargs['parent'] = args[0]
        elif len(args) == 2:
            kwargs['filePath'] = args[0]
            kwargs['parent'] = args[1]
        elif len(args) == 3:
            kwargs['filePath'] = args[0]
            kwargs['analysis'] = args[1]
            kwargs['parent'] = args[2]

        parent = kwargs.get('parent', None)
        super(QHSCoFitConfigWidget, self).__init__(parent=parent)

        # file providing the base vectors
        # filePath = kwargs.get('filePath', "basevectors_1.txt")
        filePath = kwargs.get('filePath', "basevectors_2.txt")
        # filePath = kwargs.get('filePath', "basevectors_2_17052022.txt")

        # analysis object
        self.hsVectorAnalysis = kwargs.get('analysis', HSCoFit())
        self.nroi = self.hsVectorAnalysis.nroi

        # mask for testwise fit
        self.testMask = None
        self.mask = None

        self.loadButton = QtWidgets.QToolButton(self)
        self.resetParamButton = QtWidgets.QToolButton(self)
        self.toggleParamButton = QtWidgets.QToolButton(self)
        self.resetButton = QtWidgets.QToolButton(self)
        self.testButton = QtWidgets.QToolButton(self)
        self.updateButton = QtWidgets.QToolButton(self)

        self.fileLineEdit = QtWidgets.QLineEdit(self)
        self.normalCheckBox = QtWidgets.QCheckBox(self)
        self.methodComboBox = QtWidgets.QComboBox(self)
        self.wavRegionWidgets = [
            QParamRegionWidget("roi%d" % i, self) for i in range(self.nroi)]
        self.lsVarRegionWidgets = []

        # configure actions
        self._setupActions()

        # configure widget views
        self._setupViews(*args, **kwargs)

        # load base vectors and update filePath
        self.loadFile(filePath)

        # connect signals
        self.methodComboBox.currentTextChanged.connect(self._updateSettings)
        self.normalCheckBox.stateChanged.connect(self._updateSettings)

        for i, item in enumerate(self.wavRegionWidgets):
            item.sigValueChanged.connect(self._updateROI)

        # self.imageFilterTypeComboBox.currentTextChanged.connect(
        # self._updateImageFilterSettings)
        # self.spectFilterTypeComboBox.currentTextChanged.connect(
        # self._updateSpectFilterSettings)

        # self.imageFilterSigmaSpinBox.valueChanged.connect(self.updateLSFit)
        # self.spectFilterSizeSpinBox.valueChanged.connect(self.updateLSFit)
        # self.spectFilterSigmaSpinBox.valueChanged.connect(self.updateLSFit)
        # self.spectFilterOrderSpinBox.valueChanged.connect(self.updateLSFit)
        # self.spectFilterDerivSpinBox.valueChanged.connect(self.updateLSFit)

    def _setupActions(self):
        self.loadAction = QtWidgets.QAction(self)
        self.loadAction.setIconText("...")
        self.loadAction.triggered.connect(self.onLoadFile)
        self.addAction(self.loadAction)

        self.resetParamAction = QtWidgets.QAction("Reset", self)
        self.resetParamAction.triggered.connect(self.resetParam)
        self.addAction(self.resetParamAction)

        self.toggleParamAction = QtWidgets.QAction("View", self)
        self.toggleParamAction.setCheckable(True)
        self.toggleParamAction.setChecked(True)
        self.toggleParamAction.triggered.connect(self.toggleParamView)
        self.addAction(self.toggleParamAction)

        self.resetAction = QtWidgets.QAction("Reset", self)
        self.resetAction.triggered.connect(self.reset)
        self.addAction(self.resetAction)

        self.testAction = QtWidgets.QAction("Test", self)
        self.testAction.triggered.connect(
            lambda : self._updateVectorFit(enableTest=True))
        self.addAction(self.testAction)

        self.updateAction = QtWidgets.QAction("Update", self)
        self.updateAction.triggered.connect(self._updateVectorFit)
        self.addAction(self.updateAction)

        self.loadButton.setDefaultAction(self.loadAction)
        self.resetParamButton.setDefaultAction(self.resetParamAction)
        self.toggleParamButton.setDefaultAction(self.toggleParamAction)
        self.resetButton.setDefaultAction(self.resetAction)
        self.testButton.setDefaultAction(self.testAction)
        self.updateButton.setDefaultAction(self.updateAction)

    def _setupViews(self, *args, **kwargs):
        self.mainLayout = QtWidgets.QFormLayout()
        self.mainLayout.setContentsMargins(5, 10, 5, 10)  # (l, t, r, b)
        self.mainLayout.setSpacing(3)
        self.setLayout(self.mainLayout)

        # file load ..........................................................
        self.loadLayout = QtWidgets.QFormLayout()
        label = QtWidgets.QLabel("Base vectors", self)
        label.setStyleSheet(
            "border: 0px;"
            "font: bold;"
        )
        label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.mainLayout.addRow(label)

        self.fileLineEdit.setReadOnly(True)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.fileLineEdit)
        layout.addWidget(self.loadButton)
        self.mainLayout.addRow(layout)

        # Parameter settings and controls ....................................
        label = QtWidgets.QLabel("Bounds [%]", self)
        label.setIndent(5)
        label.setMinimumHeight(20)
        label.setStyleSheet("border: 0px;")

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.resetParamButton)
        layout.addWidget(self.toggleParamButton)
        self.mainLayout.addRow(layout)

        # fit settings and controls ..........................................
        label = QtWidgets.QLabel("Least square fit", self)
        label.setStyleSheet(
            "border: 0px;"
            "font: bold;"
        )
        label.setMinimumHeight(20)
        label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.mainLayout.addRow(label)

        for i, item in enumerate(self.wavRegionWidgets):
            if self.nroi == 1:
                item.setLabel("ROI")
            else:
                item.setLabel("ROI %d" % i)
            item.setScale(1e9)  # show in nm
            item.setDecimals(0)
            item.setSingleStep(1)
            item.setMaximumWidth(200)
            self.mainLayout.addRow(item)

        label = QtWidgets.QLabel("Method", self)
        label.setStyleSheet("border: 0px;")
        label.setIndent(6)
        label.setMinimumWidth(50)

        for i, (meth, info) in enumerate(_methods):
            self.methodComboBox.addItem(meth)
            self.methodComboBox.setItemData(i, info, QtCore.Qt.ToolTipRole)
        self.methodComboBox.insertSeparator(6)
        self.methodComboBox.setCurrentIndex(1)
        # self.methodComboBox.setCurrentText('gesv')
        self.methodComboBox.setMinimumWidth(67)
        self.methodComboBox.setMaximumWidth(67)
        self.normalCheckBox.setChecked(True)
        self.normalCheckBox.setText("Normal")

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.methodComboBox)
        layout.addStretch()
        layout.addWidget(self.normalCheckBox)
        self.mainLayout.addRow(label, layout)

        layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(self)
        label.setMinimumHeight(20)
        label.setStyleSheet("border: 0px;")
        layout.addWidget(label)

        layout.addWidget(self.resetButton)
        layout.addWidget(self.testButton)
        layout.addWidget(self.updateButton)
        self.mainLayout.addRow(layout)

    def _updateROI(self, name, bounds):
        """Update the region of interest for the wavelength in the analyzer.

        Parameters
        ----------
        name : str
            The parameter name defined in the corresponding instance of
            :class:`hsi.analysis.HSImageLSAnalysis`.
        bounds : list, tuple
            The absolute lower and upper bounds for the wavelength.
        """
        logger.debug("Update ROI bounds to {}.".format(bounds))
        self.hsVectorAnalysis.set_roi(bounds)
        self._updateVectorFit(enableTest=True)

    def _updateSettings(self):
        """Update variable settings for the analyzer."""
        meth = self.methodComboBox.currentText()
        if meth == 'gesv':
            self.normalCheckBox.blockSignals(True)
            self.normalCheckBox.setChecked(True)
            self.normalCheckBox.blockSignals(False)
            self.normalCheckBox.setEnabled(False)
        else:
            self.normalCheckBox.setEnabled(True)

        if meth in ('gesv', 'lstsq', 'nnls', 'cg', 'nelder-mead'):
            for widget in self.lsVarRegionWidgets:
                widget.setEnabled(False)
        else:
            for widget in self.lsVarRegionWidgets:
                widget.setEnabled(True)

        logger.debug(
            "Update method for the least square fit to `{}`.".format(meth))

        self._updateVectorFit(enableTest=True)

    def _updateVarBounds(self, name, bounds):
        """Update variable settings for the analyzer."""
        logger.debug(
            "Update bounds for the variable '{}' to {}.".format(name, bounds))
        self.hsVectorAnalysis.set_var_bounds(name, bounds)
        self._updateVectorFit(enableTest=True)

    def _updateVectorFit(self, enableTest=False):
        """Fit the spectral data using the base vectors."""
        if enableTest and self.testMask is None:
            logger.debug("_updateVectorFit: Test enabled but no mask defined")
            return
        elif enableTest:
            mask = self.testMask
        else:
            mask = self.mask

        meth = self.methodComboBox.currentText()
        enableNormal = self.normalCheckBox.isChecked()
        kwargs = {'method': meth, 'normal': enableNormal,
                  'mask': mask}
        logger.debug("_updateVectorFit: fit with arguments {}.".format(kwargs))

        start = timer()
        self.hsVectorAnalysis.fit(meth, normal=enableNormal, mask=mask)
        logger.debug("Elapsed time for fitting: %f sec" % (timer() - start))

        self.sigValueChanged.emit(self, enableTest)

    def finalize(self):
        """ Should be called manually before object deletion
        """
        logger.debug("Finalizing: {}".format(self))
        super(QHSCoFitConfigWidget, self).finalize()

    def getROI(self, index=-1):
        """Get the region(s) of interest for spectral fits."""
        if index == -1:
            if self.nroi == 1:
                return self.wavRegionWidgets[index].value()
            else:
                return [item.value() for item in self.wavRegionWidgets]
        elif index > -1 and index < self.nroi:
            return self.wavRegionWidgets[index].value()

    def getSpectra(self, hsformat=None):
        """Get the spectral fits."""
        return self.hsVectorAnalysis.model(hsformat)

    def getSolution(self):
        """Get the solution vectors of the least square fit for the entire
        spectral dataset.

        Return a dictionary of solution vectors according to the base vectors.
        """
        meth = self.methodComboBox.currentText()
        if meth in ("gesv", "lstsq", "cg", "nelder-mead"):  # unconstrained
            return self.hsVectorAnalysis.get_solution(unpack=True, clip=False)
        else:  # constrained
            return self.hsVectorAnalysis.get_solution(unpack=True, clip=True)

    def loadFile(self, filePath):
        """Load hyper spectral image file.

        Parameters
        ----------
        filePath : str
            The full path to the input file providing the base spectra.
        """
        logger.debug("Load base spectral file: {}".format(filePath))

        # remove all existing parameter config widgets
        while len(self.lsVarRegionWidgets):
            widget = self.lsVarRegionWidgets.pop(0)
            widget.sigValueChanged.disconnect(self._updateVarBounds)
            logger.debug("Delete widget {}".format(widget))
            widget.deleteLater()
            # self.mainLayout.removeRow(widget)

        # load dictionary of base vectors into the analyzer
        baseVectors = self.hsVectorAnalysis.loadtxt(filePath, mode='bvec')
        if not baseVectors:
            self.fileLineEdit.setText("File is not valid.")
            return  # return if no base vectors were found in file

        # prepare least square problem
        self.hsVectorAnalysis.prepare_ls_problem()

        # update line edit with valid file path
        self.fileLineEdit.setText(filePath)

        # add parameter config widgets for each base vector
        for i, vec in enumerate(baseVectors.values()):
            widget = QParamRegionWidget(vec.name, vec.bounds, 100., self)
            widget.setLabel(vec.label)
            widget.setDecimals(1)
            widget.setSingleStep(0.1)
            widget.setMaximumWidth(200)
            widget.sigValueChanged.connect(self._updateVarBounds)
            # widget.setMaximumWidth(150)
            self.lsVarRegionWidgets.append(widget)
            self.mainLayout.insertRow(i+3, widget)

        # reconfigure wavelength region widget
        lbnd, ubnd = self.hsVectorAnalysis.wavelen[[0, -1]]

        for item in self.wavRegionWidgets:
            item.setBounds([lbnd, ubnd])
            item.setValueDefault([lbnd, ubnd])
            item.setValue([lbnd, ubnd])
            item.setSingleStep(1)
        self._updateVectorFit()

    def onLoadFile(self):
        """Load hyper spectral image file using a dialog box. """
        filter = "Normal text file (*.txt)"

        dir = os.path.join(getPkgDir(), "data")
        filePath, filter = QtWidgets.QFileDialog.getOpenFileName(
            None, 'Select file:', dir, filter)

        if filePath != "":
            self.fileLineEdit.setText(filePath)
            self.loadFile(filePath)

    def reset(self):
        """Set the default fitting configuration."""

        self.methodComboBox.blockSignals(True)
        self.normalCheckBox.blockSignals(True)

        self.methodComboBox.setCurrentIndex(1)
        self.normalCheckBox.setChecked(True)

        self.methodComboBox.blockSignals(False)
        self.normalCheckBox.blockSignals(False)

        # triggers a test fit using either of the methods 'gesv' or 'bvls_f'
        for item in self.wavRegionWidgets:
            item.reset()

    def resetParam(self):
        """Set the default values for the parameter bounds."""
        for widget in self.lsVarRegionWidgets:
            widget.blockSignals(True)
            widget.reset()
            widget.blockSignals(False)
            bounds = widget.value()
            self.hsVectorAnalysis.set_var_bounds(widget.name, bounds)
            logger.debug(
                "Reset bounds for {} to {}".format(widget.name, bounds))
        self._updateVectorFit(enableTest=True)

    def setData(self, y, x=None, hsformat=None):
        """Set the spectral data to be fitted.

        The Data will be automatically fitted if either of the `fast` methods
        is selected. These are

            - 'bvls_f'   : Bounded value least square algorithm, fortran.
            - 'gesv'     : Linear matrix equation (unconstrained).
            - 'lstsq'    : Least square algorithm (unconstrained).
            - 'nnls'     : Non-negative least squares.

        Parameters
        ----------
        y :  numpy.ndarray
            The spectral data.
        x :  numpy.ndarray, optional
            The wavelengths at which the spectral data are sampled. If not
            set, any previously defined wavelength data will be used. If no
            data are available an error is raised.
        """
        logger.debug("Set spectral data.")
        self.hsVectorAnalysis.set_data(y, x, hsformat)
        self.hsVectorAnalysis.prepare_ls_problem()

        # update region of interest if new x axis
        if x is not None:
            bounds = x[[0, -1]]
            for item in self.wavRegionWidgets:
                item.blockSignals(True)
                item.setBounds(bounds)
                item.setValueDefault(bounds)
                item.setValue(bounds)
                item.blockSignals(False)

        meth = self.methodComboBox.currentText()
        if meth in ('bvls_f', 'gesv', 'lstsq', 'nnls'):
            self._updateVectorFit(enableTest=False)
        else:
            # change method to either of the fast 'gesv' or 'bvls_f'
            self.methodComboBox.setCurrentIndex(1)

        # initialize test mask
        shape = self.hsVectorAnalysis.shape
        if self.testMask is None and len(shape):
            self.testMask = [0]*len(shape)

    def setFormat(self, hsformat):
        self.hsVectorAnalysis.set_format(hsformat)

    def setMethod(self, method):
        """Set method for least square fit.

        Parameters
        ----------
        method :  str
            The least square method. Should be one of

                - 'bvls'     : Bounded value least square algorithm.
                - 'bvls_f'   : Bounded value least square algorithm, fortran.
                - 'gesv'     : Linear matrix equation (unconstrained).
                - 'lstsq'    : Least square algorithm (unconstrained).
                - 'nnls'     : Non-negative least squares.
                - 'trf'      : Trust region reflective algorithm.
                - 'cg'       : Conjugate gradient algorithm (unconstrained).
                - 'l-bfgs-b' : Constrained BFGS algorithm.
                - 'nelder-mead' : Nelder-Mead algorithm (unconstrained).
                - 'powell'   : Powell algorithm.
                - 'slsqp'    : Sequential least squares Programming.
                - 'tnc'    : Truncated Newton (TNC) algorithm.
        """
        if method in [meth for meth, info in _methods]:
            logger.debug(
                "Set the method for the ls problem to: {}.".format(method))
            self.methodComboBox.setCurrentText(method)

    def setNormalEnabled(self, enableNormal):
        """Filter applied to the output data.

        Parameters
        ----------
        enableNormal : bool
            Enables or disable the normal form: :math:`\mathbf{A}^{\mathsf{T}}
            \mathbf{A}\mathbf{x} = \mathbf{A}^{\mathsf{T}} \mathbf{b}`.
        """
        self.normalCheckBox.setChecked(enableNormal)

    def setROI(self, bounds, index=-1):
        """Set the region of interest for the wavelength in the widget.

        Parameters
        ----------
        bounds : list, tuple
            The absolute lower and upper bounds for the wavelength.
        """
        logger.debug("Set ROI for the wavelength to {}".format(bounds))

        if index == -1:
            for item in self.wavRegionWidgets:
                item.setValue(bounds)
        elif index > -1 and index < self.nroi:
            self.wavRegionWidgets[index].setValue(bounds)

    def setBounds(self, bounds, index=-1):
        """Set the region of interest for the wavelength in the widget.

        Parameters
        ----------
        bounds : list, tuple
            The absolute lower and upper bounds for the wavelength.
        """
        if index == -1:
            logger.debug(
                "Set ROI bounds for the wavelength to {}".format(bounds))
            for item in self.wavRegionWidgets:
                item.setBounds(bounds)
        elif index > -1 and index < self.nroi:
            logger.debug(
                "Set ROI bounds with index {} for the wavelength to {}".format(
                    index, bounds))
            self.wavRegionWidgets[index].setBounds(bounds)

    def setMask(self, mask):
        """Set the general mask for fitting procedures.

        Parameters
        ----------
        mask : (tuple, list, or numpy.ndarray), optional
            Evaluate the fit only for selected spectra using either a tuple,
            list, array of integer arrays, one for each dimension, or a boolean
            array serving as a mask.
        """
        # if self.hsVectorAnalysis.spectra is None:
        #     self.testMask = None

        self.mask = mask
        logger.debug("Set general mask for fitting procedures.")

    def setTestMask(self, mask):
        """Set the test mask for fitting procedures.

        Parameters
        ----------
        mask : (tuple, list, or numpy.ndarray), optional
            Evaluate the fit only for selected spectra using either a tuple,
            list, array of integer arrays, one for each dimension, or a boolean
            array serving as a mask.
        """
        # if self.hsVectorAnalysis.spectra is None:
        #     self.testMask = None

        self.testMask = mask
        logger.debug("Set test mask to {}".format(mask))

    def toggleParamView(self):
        """Toggel parameters view."""
        if self.toggleParamButton.isChecked():
            for widget in self.lsVarRegionWidgets:
                widget.show()
        else:
            for widget in self.lsVarRegionWidgets:
                widget.hide()
