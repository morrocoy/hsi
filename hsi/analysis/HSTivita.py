# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:30:26 2020

@author: papkai
"""
import os.path
import numpy as np
from scipy import signal, ndimage

from .. import CONFIG_OPTIONS
from .. import __version__

from ..misc import getPkgDir
from ..core.HSFile import HSFile
from ..core.formats import HSFormatFlag, HSAbsorption, HSFormatDefault, convert

from .HSComponent import HSComponent
from .HSComponentFile import HSComponentFile

import logging

LOGGING = True
# LOGGING = False
logger = logging.getLogger(__name__)
logger.propagate = LOGGING


__all__ = ['HSTivita']


if CONFIG_OPTIONS['enableBVLS']:
    import bvls
    def bvls_f(*args, **kwargs):
        return bvls.bvls(*args, **kwargs)
else:
    def bvls_f(*args, **kwargs):
        return None


class HSTivita:
    """
    Class to approximate hyper spectral image data by a weighted sum of base
    spectra in order to analysize their individual contributions.

    Features:

    - load base spectra configuration
    - linear, nonlinear, constrained and unconstrained approximations

    Attributes
    ----------
    wavelen :  numpy.ndarray
        The wavelengths at which the spectral data are sampled.
    spectra :  numpy.ndarray
        The spectral data.
    baseVectors : dict of HSComponent
        A dictionary of base vector to represent the spectral data.
    format :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
        The format for the hyperspectral data. Should be one of:

            - :class:`HSIntensity<hsi.HSIntensity>`
            - :class:`HSAbsorption<hsi.HSAbsorption>`
            - :class:`HSExtinction<hsi.HSExtinction>`
            - :class:`HSRefraction<hsi.HSRefraction>`


    """

    def __init__(self, spectra=None, wavelen=None, bounds=None,
                 format=HSAbsorption):
        """ Constructor

        Parameters
        ----------
        spectra :  numpy.ndarray, optional
            The spectral data.
        wavelen :  numpy.ndarray, optional
            The wavelengths at which the spectral data are sampled.
        bounds :  list or tuple, optional
            The lower and upper bounds for the region of interest.
        format :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The format for the hyperspectral data. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`
        """
        self.wavelen = None  # wavelength axis
        self.spectra = None  # image data flatten to 2D ndarray

        self._tivitaSolVector = None  # vector of unknowns (tivita index values)

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


    def evaluate(self, mask=None):
        """Evaluate Tivita index values.

        Parameters
        ----------
        mask : (tuple, list, or numpy.ndarray), optional
            Evaluate the fit only for selected spectra using either a tuple,
            list, array of integer arrays, one for each dimension, or a boolean
            array serving as a mask.
        """
        # vector of unknowns for each spectrum

        if self.spectra is None:
            self._tivitaSolVector = None

        else:
            m = len(self.spectra)  # number of wavelengths

            # target vector: spectral data in a reshaped 2-dimensional format
            spectra = self.spectra.reshape(m, -1)
            wavelen = self.wavelen

            # retrieve the selected spectra
            index_mask = self._ravelMask(mask)

            m, n = spectra.shape  # number wavelengths, spectra
            self._tivitaSolVector = np.zeros((4, n))

            b = spectra[:, index_mask]

            self._tivitaSolVector[0, index_mask] = self.evalOxygenation(
                b, wavelen)
            self._tivitaSolVector[1, index_mask] = self.evalNIRPerfIndex(
                b, wavelen)
            self._tivitaSolVector[2, index_mask] = self.evalTHIndex(
                b, wavelen)
            self._tivitaSolVector[3, index_mask] = self.evalTWIndex(
                b, wavelen)


    @classmethod
    # def evalNIRPerfIndex(cls, spectra, wavelen, reg0=[655e-9, 735e-9],
    #                      reg1=[825e-9, 925e-9]):
    def evalNIRPerfIndex(cls, spectra, wavelen, reg0=[825e-9, 925e-9],
                         reg1=[655e-9, 735e-9]):
        ratio = cls.evalIndexValue(spectra, wavelen, reg0, reg1)

        p1 = 0.0
        p99 = 1.9+p1
        res = (ratio - p1) / (p99 - p1)
        return res


    @classmethod
    def evalTHIndex(cls, spectra, wavelen, reg0=[530e-9, 590e-9],
                    reg1=[785e-9, 825e-9]):
        ratio = cls.evalIndexValue(spectra, wavelen, reg0, reg1)
        # p1 = np.percentile(ratio, 1)
        # p99 = np.percentile(ratio, 99)
        p1 = 1.3
        p99 = 1.7+p1
        res = (ratio - p1) / (p99 - p1)
        return res


    # @classmethod
    # def evalTWIndex(cls, spectra, wavelen, reg0=[880e-9, 900e-9],
    #                 reg1=[955e-9, 980e-9]):
    #     ratio = cls.evalIndexValue(spectra, wavelen, reg0, reg1)
    #
    #     # p1 = np.percentile(ratio, 1)
    #     # p99 = np.percentile(ratio, 99)
    #     p1 = 0.4
    #     p99 = 0.7+p1
    #     res = (ratio - p1) / (p99 - p1)
    #
    #     return res

    @classmethod
    def evalTWIndex(cls, spectra, wavelen, reg0=[955e-9, 980e-9],
                    reg1=[880e-9, 900e-9]):
        ratio = cls.evalIndexValue(spectra, wavelen, reg0, reg1)

        # p1 = np.percentile(ratio, 1)
        # p99 = np.percentile(ratio, 99)
        p1 = 0.4
        p99 = 1.6+p1
        res = (ratio - p1) / (p99 - p1)

        return res


    @staticmethod
    def evalIndexValue(spectra, wavelen, reg0, reg1):
        """Evaluate Tivita index values.

        Parameters
        ----------
        reg0 : list or tuple
            The lower and upper limit of the first wavelength region.
        reg1 : list or tuple
            The lower and upper limit of the second wavelength region.
        """
        idx0 = np.where((wavelen >= reg0[0]) * (wavelen <= reg0[1]))[0]
        idx1 = np.where((wavelen >= reg1[0]) * (wavelen <= reg1[1]))[0]

        val0 = np.mean(spectra[idx0], axis=0)
        val1 = np.mean(spectra[idx1], axis=0)

        ratio = val0 / val1

        # p1 = np.percentile(ratio, 1)
        # p99 = np.percentile(ratio, 99)

        p1 = 0
        p99 = 1
        res = (ratio - p1) / (p99 - p1)

        return res


    @staticmethod
    def evalOxygenation(spectra, wavelen, reg0=[570e-9, 590e-9],
                    reg1=[740e-9, 780e-9]):
        """Evaluate Tivita index values.

        Parameters
        ----------
        reg0 : list or tuple
            The lower and upper limit of the first wavelength region.
        reg1 : list or tuple
            The lower and upper limit of the second wavelength region.
        """
        # second derivative of absorption
        ddspectra = signal.savgol_filter(
            spectra, window_length=5, polyorder=3, deriv=2, axis=0)

        idx0 = np.where((wavelen >= reg0[0]) * (wavelen <= reg0[1]))[0]
        idx1 = np.where((wavelen >= reg1[0]) * (wavelen <= reg1[1]))[0]

        val0 = np.min(ddspectra[idx0], axis=0)
        val1 = np.min(ddspectra[idx1], axis=0)

        r0 = 1
        r1 = .15
        ratios = val0 / val1
        res = val0 / r0 / (val0 / r0 + val1 / r1)

        return res


    def getVarVector(self, unpack=False, clip=True):
        """Get the solution vector for each spectrum.

        Parameters
        ----------
        unpack :  bool
            If true split the solution vector in a dictionary according to the
            labes of the base vectors.
        """
        x = self._tivitaSolVector

        if clip:
            lbnd = np.zeros(4)
            ubnd = np.ones(4)
            np.clip(x, lbnd[:, None], ubnd[:, None], x)

        if unpack:
            keys = ['oxy', 'nir', 'thi', 'twi']
            shape = self.spectra.shape[1:]
            return {key: x[i].reshape(shape) for i, key in
                    enumerate(keys)}
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


