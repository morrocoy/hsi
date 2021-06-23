# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:05:42 2020

@author: kpapke
"""
import os
import numpy as np
from scipy import signal, ndimage

from ..log import logmanager

from .formats import HSFormatFlag, HSFormatDefault, convert
from .formats import HSIntensity, HSAbsorption

logger = logmanager.getLogger(__name__)

__all__ = ['HSImage']


class HSImage:
    """A class used to represent a hyperspectral image.

    Objects of this class may be used to load hyperspectral image data from a
    file.
    Various filters may be applied on both, the image and spectral directions.
    The in-build RGB filter is able to extract the different color channels.
    Besides the raw format, the hyperspectral data may be accessed as
    absorption, extinction, or refraction values.
    In addition, the data may be retrieved with a standard normal variate
    correction.

    Attributes
    ----------
    format :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
        The output format for the hyperspectral data. Should be one of:

            - :class:`HSIntensity<hsi.HSIntensity>`
            - :class:`HSAbsorption<hsi.HSAbsorption>`
            - :class:`HSExtinction<hsi.HSExtinction>`
            - :class:`HSRefraction<hsi.HSRefraction>`

    spectra :  numpy.ndarray
        The hyperspectral image in raw data format.
    fspectra :  numpy.ndarray
        The hyperspectral image in filtered data format (if any filter).
    rgbBounds :  dict('red': list, 'green': list, 'blue': list)
        A dictionary which defines lower and upper bounds for each color.

        ====== ============ ============
        color  lower bound  upper bound
        ====== ============ ============
        red    585 nm       725 nm
        green  540 nm       590 nm
        blue   530 nm       560 nm
        ====== ============ ============

    wavelen :  np.ndarray
        An Array of wavelengths at which the spectra apply to.


    Example
    -------

    .. code-block:: python
       :emphasize-lines: 4

        import matplotlib.pyplot as plt
        import hsi
        from hsi import HSImage

        hsImage = HSImage(file)

        # extract rgb image from hyperspectral data
        rgbImage = hsImage.getRGBValue()

        # use absorption as output format of the hyperspectral data
        hsImage.setFormat(hsi.HSAbsorption)

        # get hyperspectral raw data
        hsiRaw = hsImage.spectra

        # add gaussian image filter, enable filter, and get current data
        hsImage.addFilter(mode='image', type='gauss', sigma=1, truncate=4)
        hsiGauss = hsImage.fspectra

        # add polynomial filter to spectra and get current data
        hsImage.addFilter(mode='spectra', type='savgol', size=7, order=2, deriv=0)
        hsiSGF = hsImage.fspectra

        # plot rgb image
        plt.imshow(rgbImage)
        plt.show()

        # plot hyperspectral image at the first wavelength point
        plt.imshow(hsiRaw[0,:, :])
        plt.show()

    """

    def __init__(self, spectra=None, wavelen=None, format=HSFormatDefault):
        """Constructor

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Parameters
        ----------
        filePath : np.ndarray or filepath, optional
            Either the multidimensional array of the spectral data or the full
            path to the input file.
        wavelen : list of np.ndarray, optional
            The wavelengths at which the spectral data are sampled.
        format :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The format for the hyperspectral data. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`

        """
        self.rgbBounds = {
            'red': [585e-9, 725e-9],
            'green': [540e-9, 590e-9],
            'blue': [530e-9, 560e-9],
        }

        self.spectra = None # data cube representing hyper spectral image
        self.fspectra = None # gaussian filtered image for each wavelength
        self.wavelen = None

        # format for the hyperspectral data
        if not HSFormatFlag.hasFlag(format):
            raise Exception("Unknown format '{}'.".format(format))
        self.format = format

        # load data from file or set directly
        if isinstance(spectra, str):
            self.load(spectra)
        elif isinstance(spectra, (np.ndarray, list)):
            self.setData(spectra, wavelen, format)



    def _testFilter(self):
        self.addFilter(mode='image', type='gauss', sigma=1, truncate=4)
        self.addFilter(mode='spectra', type='savgol', size=7, order=2, deriv=0)


    def _where(self, bounds):
        if bounds[0] < self.wavelen[0]:
            min_index = 0
        else:
            min_index = np.where(self.wavelen > bounds[0])[0][0]

        if bounds[1] > self.wavelen[-1]:
            max_index = len(self.wavelen) - 1
        else:
            max_index = np.where(self.wavelen > bounds[1])[0][0]

        return range(min_index, max_index)


    def addFilter(self, mode='image', type='gauss', size=7,
                  sigma=1., truncate=4, order=0, deriv=0):
        """Add an image or spectral filter to the output data.

        Parameters
        ----------
        mode : {'image', 'spec'}
            Defines the filter orientation for the hyperspectral image.
        type : str
            The type of filter. Should be one of

                - 'gauss'  : Gaussian filter using :func:`scipy.ndimage.gaussian_filter`
                - 'mean'   : Mean filter using :func:`scipy.ndimage.uniform_filter`
                - 'median' : Median filter using :func:`scipy.ndimage.median_filter`
                - 'savgol' : Savitzky--Golay filter using :func:`scipy.signal.savgol_filter`

        size : int, optional
            The sizes of the filter. Not used if Gaussian filter is selected.
        sigma : float, optional
            The Sigma value of a Gaussian filter.
        truncate : float, optional
            Truncate the Gaussian filter at this many standard deviations.
        order : int, optional
            The polynomial order of a Savitzky--Golay filter.
        deriv : int, optional
            The order of the derivative to compute. Only if Savitzky--Golay
            filter is selected.

        """
        if self.spectra is None:
            return None

        fspectra = self.fspectra
        if mode == 'image':
            if type == 'gauss':
                fspectra = ndimage.gaussian_filter(
                    fspectra, sigma=(0, sigma, sigma), truncate=truncate)
            elif type == 'mean':
                fspectra = ndimage.uniform_filter(
                    fspectra, size=(0, size, size))
            elif type == 'median':
                fspectra = ndimage.median_filter(fspectra, size=(1, size, size))
            else:
                raise TypeError("Unknown filter type {}".format(type))

        elif mode == 'spectra':
            if type == 'gauss':
                fspectra = ndimage.gaussian_filter(
                    fspectra, sigma=(sigma, 0, 0), truncate=truncate)
            elif type == 'mean':
                fspectra = ndimage.uniform_filter(fspectra, size=(size, 0, 0))
            elif type == 'median':
                fspectra = ndimage.median_filter(fspectra, size=(size, 1, 1))
            elif type == 'savgol':
                if not size % 2: # window size must be an odd number
                    size += 1
                fspectra = signal.savgol_filter(
                    fspectra, window_length=size, polyorder=order,
                    deriv=deriv, axis=0)
            else:
                raise TypeError("Unknown filter type {}".format(type))
        else:
            raise TypeError("Unknown filter mode {}".format(mode))

        self.fspectra = fspectra


    def clearFilter(self):
        """Remove all filters previously added to the hyperspectral image."""
        self.fspectra = self.spectra


    def getTissueMask(self, thresholds=[0.2, 0.8], bounds=None):
        """Retrieve Mask to select human tissue.

        Parameters
        ----------
        bounds: dict('absorption': list, 'reflection': list), optional
            A dictionary which defines lower and upper bounds for the regions
            of higher absorption and reflection.

        """
        if self.spectra is None:
            return None

        if bounds is None:
            bounds = {
                'absorption': [545e-9, 555e-9],
                'reflection': [645e-9, 655e-9],
            }

        intensity = convert(HSIntensity, self.format,
                            self.fspectra, self.wavelen)

        aval = np.take(intensity, self._where(bounds['absorption']), axis=0)
        rval = np.take(intensity, self._where(bounds['reflection']), axis=0)
        aavg = np.mean(aval, axis=0)
        ravg = np.mean(rval, axis=0)

        tissueIndex = (ravg - aavg) / (ravg + aavg)
        mask = np.logical_and(
            tissueIndex > thresholds[0], tissueIndex < thresholds[1])
        ndimage.binary_fill_holes(mask, output=mask)
        ndimage.binary_opening(mask, structure=np.ones((3, 3)), output=mask)

        return mask.astype('<i1')



    def getRGBValue(self, gamma='auto', bounds=None):
        """Retrieve RGB image from the intensity of hyperspectral data.

        Parameters
        ----------
        gamma : ('auto', float, None), optional
            Gamma correction value.
        bounds: dict('red': list, 'green': list, 'blue': list), optional
            A dictionary which defines lower and upper bounds for each color.

        Returns
        -------
        r,g,b : tuple
            A tuple of rgb arrays.

        """
        if self.spectra is None:
            return None

        if bounds is None:
            bounds = self.rgbBounds

        logger.debug("Derive rgb image from hyperspectral data")
        intensity = convert(HSIntensity, self.format,
                            self.fspectra, self.wavelen)

        rval = np.take(intensity, self._where(bounds['red']), axis=0)
        gval = np.take(intensity, self._where(bounds['green']), axis=0)
        bval = np.take(intensity, self._where(bounds['blue']), axis=0)

        logger.debug("Red: bounds: {}, found {} samples.".format(
            bounds['red'], len(rval)))
        logger.debug("Green: bounds: {}, found {} samples.".format(
            bounds['green'], len(gval)))
        logger.debug("Blue: bounds: {}, found {} samples.".format(
            bounds['blue'], len(bval)))

        ravg = np.mean(rval, axis=0)
        gavg = np.mean(gval, axis=0)
        bavg = np.mean(bval, axis=0)
        image = np.stack([ravg, gavg, bavg], axis=2).clip(0., 1.)

        # automatic gamma correction
        if isinstance(gamma, str) and gamma == 'auto':
            scale = 1.3
            mid = 0.5
            img_gray = np.dot(image, [0.2989, 0.5870, 0.1140])  # rgb to gray
            mean = np.mean(img_gray)
            gamma = np.log(mid) / np.log(mean)
            image = (scale * np.power(image, scale*gamma)).clip(0., 1.)

        # manual gamma correction
        elif isinstance(gamma, float):
            scale = 1.
            image = (scale * np.power(image, scale * gamma)).clip(0., 1.)

        return image


    def load(self, filePath, rotation=True, ndim=3, dtype=">f4"):
        """Load data cube from binary file.

        Note: wavelength information not included in files. It is fixed
        to the range: 500, 505, ..., 955 nm.

        Parameters
        ----------
        filePath : str
            The full path to the input file.
        rotation: bool, optional
            A flag to additionally rotate the image by 90 deg.
        ndim: int, optional
            The number of dimensions for the data cube.
        dtype: :class:`numpy.dtype`, optional
            The data type in which the spectral values are stored.

        """
        if filePath is None or not os.path.isfile(filePath):
            print("File %s not found" % filePath)
            return

        size = np.dtype(dtype).itemsize

        with open(filePath, 'rb') as file:
            buffer = file.read(size * ndim)
            shape = np.frombuffer(buffer, dtype='>i4')

            # dtypeImg = np.dtype(dtype)
            # dtypeImg = dtypeImg.newbyteorder('>')
            buffer = file.read()
            spectra = np.frombuffer(buffer, dtype=dtype)

        new_dtype = np.dtype(dtype).newbyteorder('<')

        # reshape spectral data to three-dimensional array
        spectra = spectra.reshape(shape, order='C').astype(new_dtype)
        # correct orientation for axisOrder row-major
        if rotation:
            spectra = np.rot90(spectra)
        # put wavelength axis first
        spectra = np.transpose(spectra, axes=(2, 0, 1))

        self.wavelen = np.linspace(500e-9, 1000e-9, 100, endpoint=False)
        self.spectra = convert(self.format, HSIntensity, spectra, self.wavelen)
        self.fspectra = self.spectra.copy()


    def setData(self, spectra, wavelen=None, format=None):
        """Set spectral data to be fitted.

        Parameters
        ----------
        spectra :  numpy.ndarray
            The spectral data.
        wavelen :  numpy.ndarray, optional
            The wavelengths at which the spectral data are sampled. If not
            defined the internally stored wavelength data in
            :attr:`~.HSImageLSAnalysis.xData` are used. If no data are
            available an error is raised.
        """
        if format is None:
            format = self.format

        if not HSFormatFlag.hasFlag(format):
            raise Exception("Unknown format '{}'.".format(format))

        if isinstance(spectra, list):
            spectra = np.array(spectra)
        if not isinstance(spectra, np.ndarray) or spectra.ndim < 1:
            raise Exception("Spectral y data must be ndarray of at least one "
                            "dimension.")

        # ensure two- or higher-dimensional array for spectra
        if spectra.ndim < 2:
            spectra = spectra[:, np.newaxis]

        if wavelen is not None:
            if isinstance(wavelen, list):
                wavelen = np.array(wavelen)
            if not isinstance(wavelen, np.ndarray) or wavelen.ndim > 1:
                raise Exception("Spectral x data must be 1D ndarray.")
            if len(wavelen) != len(spectra):
                raise Exception("Spectral x and y data must be of same length.")

            logger.debug("set_data: Set spectral data. Update wavelength.")
            self.wavelen = wavelen.view(np.ndarray)
            self.spectra = convert(self.format, format, spectra, wavelen)
            self.fspectra = self.spectra.copy()

        else:
            if self.wavelen is None: # set spectra without new wavelengths
                raise Exception("Wavelength information is not available. "
                                "Cannot update spectral y data")
            elif len(self.wavelen) != len(spectra):
                raise Exception("Spectral x and y data must be of same length.")
            else:
                logger.debug("set_data: Set spectral data. Preserve wavelength.")
                self.spectra = convert(self.format, format, spectra, self.wavelen)
                self.fspectra = self.spectra.copy()


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
        # check format, if not previously defined also set the format
        if not HSFormatFlag.hasFlag(format):
            raise Exception("Unknown format '{}'.".format(format))

        old_format = self.format
        self.spectra = convert(format, old_format, self.spectra, self.wavelen)
        self.fspectra = convert(format, old_format, self.fspectra, self.wavelen)
        self.format = format

    def setRange(self, start, stop, endpoint=False):
        """Set the wavelength range.

        Parameters
        ----------
        start : float
            The lower limit of the range.
        stop : float
            The upper limit of the range.
        endpoint : bool, optional
            If True, stop is the last sample. Otherwise, it is not included.
            Default is True.
        """
        if isinstance(self.spectra, np.ndarray):
            self.wavelen = np.linspace(
                start, stop, len(self.spectra), endpoint=endpoint)


    @property
    def shape(self):
        """tuple: The shape of the hyperspectral image."""
        if isinstance(self.spectra, np.ndarray):
            return self.spectra.shape
        else:
            return None