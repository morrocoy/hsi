# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:05:42 2020

@author: kpapke
"""
import os
import numpy
from scipy import signal, ndimage

from ..log import logmanager

from .hs_formats import HSFormatFlag, HSFormatDefault, convert
from .hs_formats import HSIntensity

logger = logmanager.getLogger(__name__)

__all__ = ['HSImage']


class HSImage:
    """A class used to represent a hyperspectral image.

    Objects of this class may be used to load hyperspectral image data from a
    file.
    Various filters may be applied on both, the image and spectral directions.
    The in-build RGB filter is able to extract the different color channels.
    Besides the raw hsformat, the hyperspectral data may be accessed as
    absorption, extinction, or refraction values.
    In addition, the data may be retrieved with a standard normal variate
    correction.

    Attributes
    ----------
    hsformat :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
        The output hsformat for the hyperspectral data. Should be one of:

            - :class:`HSIntensity<hsi.HSIntensity>`
            - :class:`HSAbsorption<hsi.HSAbsorption>`
            - :class:`HSExtinction<hsi.HSExtinction>`
            - :class:`HSRefraction<hsi.HSRefraction>`

    spectra :  numpy.ndarray
        The hyperspectral image in raw data hsformat.
    fspectra :  numpy.ndarray
        The hyperspectral image in filtered data hsformat (if any filter).
    rgbBounds :  dict('red': list, 'green': list, 'blue': list)
        A dictionary which defines lower and upper bounds for each color.

        ====== ============ ============
        color  lower bound  upper bound
        ====== ============ ============
        red    585 nm       725 nm
        green  540 nm       590 nm
        blue   530 nm       560 nm
        ====== ============ ============

    wavelen :  numpy.ndarray
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
        rgbImage = hsImage.as_rgb()

        # use absorption as output hsformat of the hyperspectral data
        hsImage.set_format(hsi.HSAbsorption)

        # get hyperspectral raw data
        hsiRaw = hsImage.spectra

        # add gaussian image filter, enable filter, and get current data
        hsImage.add_filter(mode='image', filter_type='gauss', sigma=1,
                           truncate=4)
        hsiGauss = hsImage.fspectra

        # add polynomial filter to spectra and get current data
        hsImage.add_filter(mode='spectra', filter_type='savgol', size=7,
                           order=2, deriv=0)
        hsiSGF = hsImage.fspectra

        # plot rgb image
        plt.imshow(rgbImage)
        plt.show()

        # plot hyperspectral image at the first wavelength point
        plt.imshow(hsiRaw[0,:, :])
        plt.show()

    """

    def __init__(self, spectra=None, wavelen=None, hsformat=HSFormatDefault):
        """Constructor

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Parameters
        ----------
        spectra : numpy.ndarray or str, optional
            Either the multidimensional array of the spectral data or the
            path to the input file.
        wavelen : list of numpy.ndarray, optional
            The wavelengths at which the spectral data are sampled.
        hsformat :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The hsformat for the hyperspectral data. Should be one of:

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

        self.spectra = None  # data cube representing hyper spectral image
        self.fspectra = None  # gaussian filtered image for each wavelength
        self.wavelen = None

        # hsformat for the hyperspectral data
        if not HSFormatFlag.has_flag(hsformat):
            raise Exception("Unknown hsformat '{}'.".format(hsformat))
        self.hsformat = hsformat

        # load data from file or set directly
        if isinstance(spectra, str):
            self.load(spectra)
        elif isinstance(spectra, (numpy.ndarray, list)):
            self.set_data(spectra, wavelen, hsformat)

    def _test_filter(self):
        self.add_filter(mode='image', filter_type='gauss', sigma=1, truncate=4)
        self.add_filter(mode='spectra', filter_type='savgol', size=7, order=2,
                        deriv=0)

    def _where(self, bounds):
        if bounds[0] < self.wavelen[0]:
            min_index = 0
        else:
            min_index = numpy.where(self.wavelen > bounds[0])[0][0]

        if bounds[1] > self.wavelen[-1]:
            max_index = len(self.wavelen) - 1
        else:
            max_index = numpy.where(self.wavelen > bounds[1])[0][0]

        return range(min_index, max_index)

    def add_filter(self, mode='image', filter_type='gauss', size=7,
                   sigma=1., truncate=4, order=0, deriv=0):
        """Add an image or spectral filter to the output data.

        Parameters
        ----------
        mode : {'image', 'spec'}
            Defines the filter orientation for the hyperspectral image.
        filter_type : str
            The type of filter. Should be one of

                - 'gauss'  : Gaussian filter using
                             :func:`scipy.ndimage.gaussian_filter`
                - 'mean'   : Mean filter using
                             :func:`scipy.ndimage.uniform_filter`
                - 'median' : Median filter using
                             :func:`scipy.ndimage.median_filter`
                - 'savgol' : Savitzky--Golay filter using
                             :func:`scipy.signal.savgol_filter`

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
            if filter_type == 'gauss':
                fspectra = ndimage.gaussian_filter(
                    fspectra, sigma=(0, sigma, sigma), truncate=truncate)
            elif filter_type == 'mean':
                fspectra = ndimage.uniform_filter(
                    fspectra, size=(0, size, size))
            elif filter_type == 'median':
                fspectra = ndimage.median_filter(fspectra, size=(1, size, size))
            else:
                raise TypeError("Unknown filter type {}".format(filter_type))

        elif mode == 'spectra':
            if filter_type == 'gauss':
                fspectra = ndimage.gaussian_filter(
                    fspectra, sigma=(sigma, 0, 0), truncate=truncate)
            elif filter_type == 'mean':
                fspectra = ndimage.uniform_filter(fspectra, size=(size, 0, 0))
            elif filter_type == 'median':
                fspectra = ndimage.median_filter(fspectra, size=(size, 1, 1))
            elif filter_type == 'savgol':
                if not size % 2:  # window size must be an odd number
                    size += 1
                fspectra = signal.savgol_filter(
                    fspectra, window_length=size, polyorder=order,
                    deriv=deriv, axis=0)
            else:
                raise TypeError("Unknown filter type {}".format(filter_type))
        else:
            raise TypeError("Unknown filter mode {}".format(mode))

        self.fspectra = fspectra

    def as_rgb(self, gamma='auto', bounds=None):
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
        intensity = convert(HSIntensity, self.hsformat,
                            self.fspectra, self.wavelen)

        rval = numpy.take(intensity, self._where(bounds['red']), axis=0)
        gval = numpy.take(intensity, self._where(bounds['green']), axis=0)
        bval = numpy.take(intensity, self._where(bounds['blue']), axis=0)

        logger.debug("Red: bounds: {}, found {} samples.".format(
            bounds['red'], len(rval)))
        logger.debug("Green: bounds: {}, found {} samples.".format(
            bounds['green'], len(gval)))
        logger.debug("Blue: bounds: {}, found {} samples.".format(
            bounds['blue'], len(bval)))

        ravg = numpy.mean(rval, axis=0)
        gavg = numpy.mean(gval, axis=0)
        bavg = numpy.mean(bval, axis=0)
        image = numpy.stack([ravg, gavg, bavg], axis=2).clip(0., 1.)

        # automatic gamma correction
        if isinstance(gamma, str) and gamma == 'auto':
            scale = 1.3
            mid = 0.5
            img_gray = numpy.dot(image, [0.2989, 0.5870, 0.1140])  # rgb to gray
            mean = numpy.mean(img_gray)
            gamma = numpy.log(mid) / numpy.log(mean)
            image = (scale * numpy.power(image, scale*gamma)).clip(0., 1.)

        # manual gamma correction
        elif isinstance(gamma, float):
            scale = 1.
            image = (scale * numpy.power(image, scale * gamma)).clip(0., 1.)

        return image

    def clear_filter(self):
        """Remove all filters previously added to the hyperspectral image."""
        self.fspectra = self.spectra

    def get_tissue_mask(self, thresholds=None, bounds=None):
        """Retrieve Mask to select human tissue.

        Parameters
        ----------
        thresholds : list of float, optional
            Lower and upper threshold applied to the intensity ratio.
        bounds : dict('absorption': list, 'reflection': list), optional
            A dictionary which defines lower and upper bounds for the regions
            of higher absorption and reflection.

        """
        if self.spectra is None:
            return None

        if thresholds is None:
            thresholds = [0.2, 0.8]

        if bounds is None:
            bounds = {
                'absorption': [545e-9, 555e-9],
                'reflection': [645e-9, 655e-9],
            }

        intensity = convert(HSIntensity, self.hsformat,
                            self.fspectra, self.wavelen)

        aval = numpy.take(intensity, self._where(bounds['absorption']), axis=0)
        rval = numpy.take(intensity, self._where(bounds['reflection']), axis=0)
        aavg = numpy.mean(aval, axis=0)
        ravg = numpy.mean(rval, axis=0)

        tissue_index = (ravg - aavg) / (ravg + aavg)
        mask = numpy.logical_and(
            tissue_index > thresholds[0], tissue_index < thresholds[1])
        ndimage.binary_fill_holes(mask, output=mask)
        ndimage.binary_opening(mask, structure=numpy.ones((3, 3)), output=mask)

        return mask.astype('<i1')

    def load(self, file_path, rotation=True, ndim=3, dtype=">f4"):
        """Load data cube from binary file.

        Note: wavelength information not included in files. It is fixed
        to the range: 500, 505, ..., 955 nm.

        Parameters
        ----------
        file_path : str
            The full path to the input file.
        rotation: bool, optional
            A flag to additionally rotate the image by 90 deg.
        ndim: int, optional
            The number of dimensions for the data cube.
        dtype: :class:`numpy.dtype`, optional
            The data type in which the spectral values are stored.

        """
        if file_path is None or not os.path.isfile(file_path):
            print("File %s not found" % file_path)
            return

        size = numpy.dtype(dtype).itemsize

        with open(file_path, 'rb') as file:
            buffer = file.read(size * ndim)
            shape = numpy.frombuffer(buffer, dtype='>i4')

            # dtypeImg = numpy.dtype(dtype)
            # dtypeImg = dtypeImg.newbyteorder('>')
            buffer = file.read()
            spectra = numpy.frombuffer(buffer, dtype=dtype)

        new_dtype = numpy.dtype(dtype).newbyteorder('<')

        # reshape spectral data to three-dimensional array
        spectra = spectra.reshape(shape, order='C').astype(new_dtype)
        # correct orientation for axisOrder row-major
        if rotation:
            spectra = numpy.rot90(spectra)
        # put wavelength axis first
        spectra = numpy.transpose(spectra, axes=(2, 0, 1))

        self.wavelen = numpy.linspace(500e-9, 1000e-9, 100, endpoint=False)
        self.spectra = convert(self.hsformat, HSIntensity, spectra, self.wavelen)
        self.fspectra = numpy.copy(self.spectra)

    def set_data(self, spectra, wavelen=None, hsformat=None):
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
        hsformat :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The output hsformat for the hyperspectral data.
        """
        if hsformat is None:
            hsformat = self.hsformat

        if not HSFormatFlag.has_flag(hsformat):
            raise Exception("Unknown hsformat '{}'.".format(hsformat))

        if isinstance(spectra, list):
            spectra = numpy.array(spectra)
        if not isinstance(spectra, numpy.ndarray) or spectra.ndim < 1:
            raise Exception("Spectral y data must be ndarray of at least one "
                            "dimension.")

        # ensure two- or higher-dimensional array for spectra
        if spectra.ndim < 2:
            spectra = spectra[:, numpy.newaxis]

        if wavelen is not None:
            if isinstance(wavelen, list):
                wavelen = numpy.array(wavelen)
            if not isinstance(wavelen, numpy.ndarray) or wavelen.ndim > 1:
                raise Exception("Spectral x data must be 1D ndarray.")
            if len(wavelen) != len(spectra):
                raise Exception("Spectral x and y data must be of same length.")

            logger.debug("set_data: Set spectral data. Update wavelength.")
            self.wavelen = wavelen.view(numpy.ndarray)
            self.spectra = convert(self.hsformat, hsformat, spectra, wavelen)
            self.fspectra = numpy.copy(self.spectra)

        else:
            if self.wavelen is None:  # set spectra without new wavelengths
                raise Exception("Wavelength information is not available. "
                                "Cannot update spectral y data")
            elif len(self.wavelen) != len(spectra):
                raise Exception("Spectral x and y data must be of same length.")
            else:
                logger.debug(
                    "set_data: Set spectral data. Preserve wavelength.")
                self.spectra = convert(
                    self.hsformat, hsformat, spectra, self.wavelen)
                self.fspectra = numpy.copy(self.spectra)

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
        # check hsformat, if not previously defined also set the hsformat
        if not HSFormatFlag.has_flag(hsformat):
            raise Exception("Unknown hsformat '{}'.".format(hsformat))

        old_format = self.hsformat
        self.spectra = convert(
            hsformat, old_format, self.spectra, self.wavelen)
        self.fspectra = convert(
            hsformat, old_format, self.fspectra, self.wavelen)
        self.hsformat = hsformat

    def set_range(self, start, stop, endpoint=False):
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
        if isinstance(self.spectra, numpy.ndarray):
            self.wavelen = numpy.linspace(
                start, stop, len(self.spectra), endpoint=endpoint)

    @property
    def shape(self):
        """tuple: The shape of the hyperspectral image."""
        if isinstance(self.spectra, numpy.ndarray):
            return self.spectra.shape
        else:
            return None
