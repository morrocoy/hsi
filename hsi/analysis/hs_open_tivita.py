# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:30:26 2020

@author: kpapke
"""
import numpy as np
from scipy import signal

from ..log import logmanager
from ..core.hs_formats import HSAbsorption

from .hs_base_analysis import HSBaseAnalysis



logger = logmanager.getLogger(__name__)

__all__ = ['HSOpenTivita']


class HSOpenTivita(HSBaseAnalysis):
    """
    Class to analyze hyper spectral image data by using the original algorithms
    of Tivita.

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
        super(HSOpenTivita, self).__init__(spectra, wavelen, hsformat)
        self.prefix = "tivita_"
        self.keys = ['tivita_oxy', 'tivita_nir', 'tivita_thi', 'tivita_twi']
        self.labels = [
            "Oxygenation (TIVITA)",
            "NIR-Perfusion (TIVITA)",
            "THI (TIVITA)",
            "TWI (TIVITA)"
        ]

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
            self._anaVarVector = None
            self._anaVarBounds = None

        else:
            m = len(self.spectra)  # number of wavelengths

            # target vector: spectral data in a reshaped 2-dimensional hsformat
            spectra = self.spectra.reshape(m, -1)
            wavelen = self.wavelen

            # retrieve the selected spectra
            index_mask = self._ravel_mask(mask)

            m, n = spectra.shape  # number wavelengths, spectra
            self._anaVarVector = np.zeros((4, n))
            self._anaVarScales = np.ones((4, n))
            self._anaVarBounds = np.array([[0., 1.] for i in range(4)])

            b = spectra[:, index_mask]

            self._anaVarVector[0, index_mask] = self.evalOxygenation(
                b, wavelen)
            self._anaVarVector[1, index_mask] = self.evalNIRPerfIndex(
                b, wavelen)
            self._anaVarVector[2, index_mask] = self.evalTHIndex(
                b, wavelen)
            self._anaVarVector[3, index_mask] = self.evalTWIndex(
                b, wavelen)




    @classmethod
    # def evaluate_nir(cls, spectra, wavelen, reg0=[655e-9, 735e-9],
    #                      reg1=[825e-9, 925e-9]):
    def evalNIRPerfIndex(cls, spectra, wavelen, reg0=[825e-9, 925e-9],
                         reg1=[655e-9, 735e-9]):
        ratio = cls.evalIndexValue(spectra, wavelen, reg0, reg1)

        p1 = 0.0
        p99 = 1.9 + p1
        res = (ratio - p1) / (p99 - p1)
        return res

    @classmethod
    def evalTHIndex(cls, spectra, wavelen, reg0=[530e-9, 590e-9],
                    reg1=[785e-9, 825e-9]):
        ratio = cls.evalIndexValue(spectra, wavelen, reg0, reg1)
        # p1 = np.percentile(ratio, 1)
        # p99 = np.percentile(ratio, 99)
        p1 = 1.3
        p99 = 1.7 + p1
        res = (ratio - p1) / (p99 - p1)
        return res

    @classmethod
    def evalTWIndex(cls, spectra, wavelen, reg0=[955e-9, 980e-9],
                    reg1=[880e-9, 900e-9]):
        ratio = cls.evalIndexValue(spectra, wavelen, reg0, reg1)

        # p1 = np.percentile(ratio, 1)
        # p99 = np.percentile(ratio, 99)
        p1 = 0.4
        p99 = 1.6 + p1
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