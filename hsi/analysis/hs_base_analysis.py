# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:39:03 2021

@author: kpapke
"""
import numpy

from ..log import logmanager

from ..core.hs_formats import HSFormatFlag, HSAbsorption, convert

logger = logmanager.getLogger(__name__)

__all__ = ['HSBaseAnalysis']


class HSBaseAnalysis(object):
    """
    Base class for hs image analysis.

    Attributes
    ----------
    wavelen :  numpy.ndarray
        The wavelengths at which the spectral data are sampled.
    spectra :  numpy.ndarray
        The spectral data.
    hsformat :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
        The hsformat for the hyperspectral data. Should be one of:

            - :class:`HSIntensity<hsi.HSIntensity>`
            - :class:`HSAbsorption<hsi.HSAbsorption>`
            - :class:`HSExtinction<hsi.HSExtinction>`
            - :class:`HSRefraction<hsi.HSRefraction>`

    keys : list
        A list of solution parameter keys.

    """

    def __init__(self, spectra=None, wavelen=None, hsformat=HSAbsorption):
        """ Constructor

        Parameters
        ----------
        spectra :  numpy.ndarray, optional
            The spectral data.
        wavelen :  numpy.ndarray, optional
            The wavelengths at which the spectral data are sampled.
        hsformat :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The hsformat for the hyperspectral data. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`
        """
        self.wavelen = None  # wavelength axis
        self.spectra = None  # image data flatten to 2D ndarray

        self._anaTrgVector = None  # target vectors (reshaped spectral data)
        self._anaVarVector = None  # vector of unknowns
        self._anaVarScales = None  # scale factors for each unknown variable
        self._anaVarBounds = None  # bounds for each unknown variable
        self._anaResVector = None  # residual vector (squared Euclidean 2-norm)
        self._anaSysMatrix = None  # assembly of normalized based vector

        # list of solution parameter keys
        self.keys = []

        # check hsformat, if not previously defined also set the hsformat
        if not HSFormatFlag.has_flag(hsformat):
            raise Exception("Unknown hsformat '{}'.".format(hsformat))
        self.hsformat = hsformat

        # Forwards data arguments to self.set_data() if available
        if spectra is not None:
            self.set_data(spectra, wavelen)

    def _ravel_mask(self, mask=None):
        """Convert a mask to raveled indices for fitting selectively.

        Parameters
        ----------
        mask : (tuple, list, or numpy.ndarray), optional
            Evaluate the fit only for selected spectra using either a tuple,
            list, array of integer arrays, one for each dimension, or a boolean
            array serving as a mask.

        Returns
        -------
        raveled_indices: numpy.ndarray
            An array of indices into the flattened version of
            :attr:`~.HSImageLSAnalysis.yData` apart from the first dimension
            which corresponds to the wavelength.
        """
        shape = self.spectra.shape[1:]
        if isinstance(
                mask, (tuple, list, numpy.ndarray)) and len(mask) == len(shape):
            # tuple, list, or array of integer arrays, one for each dimension
            raveled_mask = numpy.ravel_multi_index(mask, shape, mode='raise')
            if not hasattr(raveled_mask, '__len__'):  # convert integer to list
                raveled_mask = [raveled_mask]
            logger.debug("Index Mask: {} to {} {}".format(
                mask, raveled_mask, type(raveled_mask)))
        elif isinstance(mask, numpy.ndarray) and mask.shape == shape:
            # array of boolean providing the same shape as the spectral data
            rmask = mask.reshape([-1])
            raveled_mask = numpy.where(rmask)[0]  # where returns a tuple of array
            logger.debug("Boolean Mask: {} to {}".format(mask, raveled_mask))
        else:
            # select all spectra if no mask defined
            raveled_mask = range(int(numpy.prod(shape)))
            logger.debug("No Mask")

        return raveled_mask

    def clear(self):
        """ Clear all spectral information including base vectors."""
        self.wavelen = None  # wavelength axis
        self.spectra = None  # image data flatten to 2D ndarray

        self._anaTrgVector = None
        self._anaVarVector = None
        self._anaResVector = None
        self._anaSysMatrix = None
        self._anaVarScales = None
        self._anaVarBounds = None

    def get_residual(self):

        if isinstance(self.spectra, numpy.ndarray) and isinstance(
                self._anaResVector, numpy.ndarray):
            shape = self.spectra.shape
            return self._anaResVector.reshape(shape[1:])
        else:
            return None

    def get_solution(self, unpack=False, clip=True):
        """Get the solution vector for each spectrum.

        Parameters
        ----------
        unpack :  bool
            If true split the solution vector in a dictionary according to the
            labes of the base vectors.
        clip : bool
            Clip the parameter according to the predefined bounds.
        """
        if self._anaVarVector is None:
            return

        # x = self._anaVarVector
        if clip:
            lbnd = self._anaVarBounds[:, 0]
            ubnd = self._anaVarBounds[:, 1]

            _lbnd = lbnd.copy()
            _ubnd = ubnd.copy()
            _lbnd[numpy.isnan(_lbnd)] = -numpy.inf
            _ubnd[numpy.isnan(_ubnd)] = numpy.inf

            # print(self._anaVarScales)
            # numpy.clip(x, lbnd[:, None], ubnd[:, None], x)

            x = self._anaVarScales * numpy.clip(
                self._anaVarVector, _lbnd[:, None], _ubnd[:, None])
        else:
            x = self._anaVarScales * self._anaVarVector

        if unpack:
            shape = self.spectra.shape[1:]
            return {key: x[i].reshape(shape) for i, key in
                    enumerate(self.keys)}
        else:
            k, n = x.shape  # number of variables, spectra
            shape = (k,) + self.spectra.shape[1:]
            return x.reshape(shape)

    def set_data(self, y, x=None, hsformat=None):
        """Set spectral data to be fitted.

        Parameters
        ----------
        y :  numpy.ndarray
            The spectral data.
        x :  numpy.ndarray, optional
            The wavelengths at which the spectral data are sampled. If not
            defined the internally stored wavelength data in
            :attr:`~.HSImageLSAnalysis.xData` are used. If no data are
            available an error is raised.
        hsformat :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The hsformat for the hyperspectral data. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`
        """
        if hsformat is None:
            hsformat = self.hsformat

        if not HSFormatFlag.has_flag(hsformat):
            raise Exception("Unknown hsformat '{}'.".format(hsformat))

        if isinstance(y, list):
            y = numpy.array(y)
        if not isinstance(y, numpy.ndarray) or y.ndim < 1:
            raise Exception("Spectral y data must be ndarray of at least one "
                            "dimension.")

        # ensure two- or higher-dimensional array
        if y.ndim < 2:
            y = y[:, numpy.newaxis]

        if x is not None:
            if isinstance(x, list):
                x = numpy.array(x)
            if not isinstance(x, numpy.ndarray) or x.ndim > 1:
                raise Exception("Spectral x data must be 1D ndarray.")
            if len(x) != len(y):
                raise Exception("Spectral x and y data must be of same length.")

            logger.debug("set_data: Set spectral data. Update wavelength.")
            self.spectra = convert(self.hsformat, hsformat, y, x)
            self.wavelen = x.view(numpy.ndarray)

        else:
            if self.wavelen is None:  # set yData without
                raise Exception("Wavelength information is not available. "
                                "Cannot update spectral y data")
            elif len(self.wavelen) != len(y):
                raise Exception(
                    "Spectral x and y data must be of same length.")
            else:
                logger.debug(
                    "set_data: Set spectral data. Preserve wavelength.")
                self.spectra = convert(self.hsformat, hsformat, y, self.wavelen)

    def set_format(self, hsformat):
        """Set the hsformat for the hyperspectral data.

        Parameters
        ----------
        hsformat :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The hsformat for the hyperspectral data. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`

        """
        if not HSFormatFlag.has_flag(hsformat):
            raise Exception("Unknown hsformat '{}'.".format(hsformat))

        if hsformat != self.hsformat:
            if self.spectra is not None:
                self.spectra[:] = convert(
                    hsformat, self.hsformat, self.spectra, self.wavelen)
            self.hsformat = hsformat

    @property
    def shape(self):
        if self.spectra is None:
            return tuple([])
        else:
            return self.spectra.shape[1:]
