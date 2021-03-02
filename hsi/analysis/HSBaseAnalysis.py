# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:39:03 2021

@author: kpapke
"""
import numpy as np

from ..core.formats import HSFormatFlag, HSAbsorption, convert

import logging

LOGGING = True
# LOGGING = False
logger = logging.getLogger(__name__)
logger.propagate = LOGGING


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
    format :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
        The format for the hyperspectral data. Should be one of:

            - :class:`HSIntensity<hsi.HSIntensity>`
            - :class:`HSAbsorption<hsi.HSAbsorption>`
            - :class:`HSExtinction<hsi.HSExtinction>`
            - :class:`HSRefraction<hsi.HSRefraction>`

    keys : list
        A list of solution parameter keys.

    """

    def __init__(self, spectra=None, wavelen=None, format=HSAbsorption):
        """ Constructor

        Parameters
        ----------
        spectra :  numpy.ndarray, optional
            The spectral data.
        wavelen :  numpy.ndarray, optional
            The wavelengths at which the spectral data are sampled.
        format :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The format for the hyperspectral data. Should be one of:

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

        # check format, if not previously defined also set the format
        if not HSFormatFlag.hasFlag(format):
            raise Exception("Unknown format '{}'.".format(format))
        self.format = format

        # Forwards data arguments to self.setData() if available
        if spectra is not None:
            self.setData(spectra, wavelen)



    def _ravelMask(self, mask=None):
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
        if isinstance(mask, (tuple, list, np.ndarray)) and len(mask) == len(shape):
            # tuple, list, or array of integer arrays, one for each dimension
            raveledMask = np.ravel_multi_index(mask, shape, mode='raise')
            if not hasattr(raveledMask, '__len__'):  # convert integer to list
                raveledMask = [raveledMask]
            logger.debug("Index Mask: {} to {} {}".format(mask, raveledMask, type(raveledMask)))
        elif isinstance(mask, np.ndarray) and mask.shape == shape:
            # array of boolean providing the same shape as the spectral data
            rmask = mask.reshape([-1])
            raveledMask = np.where(rmask)[0]  # where returns a tuple of array
            logger.debug("Boolean Mask: {} to {}".format(mask, raveledMask))
        else:
            # select all spectra if no mask defined
            raveledMask = range(int(np.prod(shape)))
            logger.debug("No Mask")

        return raveledMask


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


    def getResiduals(self):

        if isinstance(self.spectra, np.ndarray) and isinstance(
                self._anaResVector, np.ndarray):
            shape = self.spectra.shape
            return self._anaResVector.reshape(shape[1:])
        else:
            return None


    def getSolution(self, unpack=False, clip=True):
        """Get the solution vector for each spectrum.

        Parameters
        ----------
        unpack :  bool
            If true split the solution vector in a dictionary according to the
            labes of the base vectors.
        """
        if self._anaVarVector is None:
            return

        x = self._anaVarVector
        if clip:
            lbnd = self._anaVarBounds[:, 0]
            ubnd = self._anaVarBounds[:, 1]
            # np.clip(x, lbnd[:, None], ubnd[:, None], x)
            x = self._anaVarScales * np.clip(
                self._anaVarVector, lbnd[:, None], ubnd[:, None])
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


    def setData(self, y, x=None, format=None):
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
        """
        if format is None:
            format = self.format

        if not HSFormatFlag.hasFlag(format):
            raise Exception("Unknown format '{}'.".format(format))

        if isinstance(y, list):
            y = np.array(y)
        if not isinstance(y, np.ndarray) or y.ndim < 1:
            raise Exception("Spectral y data must be ndarray of at least one "
                            "dimension.")

        # ensure two- or higher-dimensional array
        if y.ndim < 2:
            y = y[:, np.newaxis]

        if x is not None:
            if isinstance(x, list):
                x = np.array(x)
            if not isinstance(x, np.ndarray) or x.ndim > 1:
                raise Exception("Spectral x data must be 1D ndarray.")
            if len(x) != len(y):
                raise Exception("Spectral x and y data must be of same length.")

            logger.debug("setData: Set spectral data. Update wavelength.")
            self.spectra = convert(self.format, format, y, x)
            self.wavelen = x.view(np.ndarray)

        else:
            if self.wavelen is None: # set yData without
                raise Exception("Wavelength information is not available. "
                                "Cannot update spectral y data")
            elif len(self.wavelen) != len(y):
                raise Exception("Spectral x and y data must be of same length.")
            else:
                logger.debug("setData: Set spectral data. Preserve wavelength.")
                self.spectra = convert(self.format, format, y, self.wavelen)


    def setFormat(self, format):
        """Set the format for the hyperspectral data.

        Parameters
        ----------
        format :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The format for the hyperspectral data. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`

        """
        if not HSFormatFlag.hasFlag(format):
            raise Exception("Unknown format '{}'.".format(format))

        if format != self.format:
            if self.spectra is not None:
                self.spectra[:] = convert(
                    format, self.format, self.spectra, self.wavelen)
            self.format = format


    @property
    def shape(self):
        if self.spectra is None:
            return tuple([])
        else:
            return self.spectra.shape[1:]


