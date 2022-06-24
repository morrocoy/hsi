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
        self._anaVarVector = None  # vector of variable solutions
        self._anaFixVector = None  # vector of fixed solutions
        self._anaVarScales = None  # scale factors for each unknown variable
        self._anaVarBounds = None  # bounds for each unknown variable
        self._anaResVector = None  # residual vector (squared Euclidean 2-norm)
        self._anaSysMatrix = None  # assembly of normalized based vector
        self._anaVarBuffer = None  # storage for variable solutions

        self._buffer_maxcount = 10  # max number of solution stores
        self._buffer_count = 0  # number of solution stores
        self._buffer_index = 0  # index on solution buffer


        # list of solution parameter keys
        self._keys = []
        self.prefix = ""  # key prefix to distinguish analyzers
        self.use_prefix = True  # enable or disable prefix

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
        self._anaVarBuffer = None

    def get_residual(self):

        if isinstance(self.spectra, numpy.ndarray) and isinstance(
                self._anaResVector, numpy.ndarray):
            shape = self.spectra.shape
            return self._anaResVector.reshape(shape[1:])
        else:
            return None

    def push_solution(self):
        self._buffer_index = (self._buffer_index + 1) % self._buffer_maxcount
        self._anaVarBuffer[self._buffer_index] = self._anaVarVector.copy()

        if self._buffer_count < self._buffer_maxcount:
            self._buffer_count += 1

    def pop_solution(self):
        if self._buffer_count > 0:
            self._buffer_index = \
                (self._buffer_index + self._buffer_maxcount - 1) \
                % self._buffer_maxcount
            self._buffer_count -= 1

    def get_solution(self, which="last", unpack=False, clip=True, norm=False, flatten=False):
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

        # index circular buffer
        _ibuf = [
            (self._buffer_index + self._buffer_maxcount - i) %
            self._buffer_maxcount for i in range(self._buffer_count)]

        if norm:
            scale = 1
        else:
            scale = self._anaVarScales

        if clip:
            lbnd = self._anaVarBounds[:, 0]
            ubnd = self._anaVarBounds[:, 1]

            _lbnd = lbnd.copy()
            _ubnd = ubnd.copy()
            _lbnd[numpy.isnan(_lbnd)] = -numpy.inf
            _ubnd[numpy.isnan(_ubnd)] = numpy.inf

            if which == "last":
                x = scale * numpy.clip(
                    self._anaVarVector, _lbnd[:, None], _ubnd[:, None])

            elif which == "all":
                x = scale * numpy.clip(
                    self._anaVarBuffer[_ibuf], _lbnd[:, None], _ubnd[:, None])

            elif isinstance(which, int) and 0 < which < self._buffer_maxcount:
                x = scale * numpy.clip(self._anaVarBuffer[_ibuf[which]],
                                       _lbnd[:, None], _ubnd[:, None])

            else:
                raise Exception(
                    "Selection '%s' for solution vector unknown." % which)

        else:
            if which == "last":
                x = scale * self._anaVarVector

            elif which == "all":
                x = scale * self._anaVarBuffer[_ibuf]

            elif isinstance(which, int) and 0 < which < self._buffer_maxcount:
                x = scale * self._anaVarBuffer[_ibuf[which]]

            else:
                raise Exception(
                    "Selection '%s' for solution vector unknown." % which)

        if unpack:
            shape = self.spectra.shape[1:]

            if len(x.shape) == 3:
                return {"%s_%d" % (key, j) : x[j, i, :].reshape(shape) for i, key in
                        enumerate(self.keys) for j in range(len(x))}
            elif len(x.shape) == 2:
                return {key: x[i].reshape(shape) for i, key in
                        enumerate(self.keys)}
            else:
                raise Exception("Wrong solution format '{}'.".format(x.shape))

        else:
            if flatten:
                return x
            elif len(x.shape) == 3:
                p, k, n = x.shape  # number of variables, spectra
                shape = (p, k,) + self.spectra.shape[1:]
                return x.reshape(shape)
            elif len(x.shape) == 2:
                k, n = x.shape  # number of variables, spectra
                shape = (k,) + self.spectra.shape[1:]
                return x.reshape(shape)
            else:
                raise Exception("Wrong solution format '{}'.".format(x.shape))

    @property
    def keys(self):
        if self.use_prefix:
            return [self.prefix + key for key in self._keys]
        else:
            return self._keys

    def set_prefix_enabled(self, b):
        self.use_prefix = b

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
