# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 19:07:19 2021

@author: kpapke
"""
import os.path
import datetime
import numpy as np

from .. import __version__

from ..log import logmanager
from ..misc import getPkgDir

from .hs_formats import HSFormatDefault
from .hs_file import HSFile

from .hs_component import HSComponent

logger = logmanager.getLogger(__name__)


__all__ = ['HSComponentFile']

LABEL_VECTOR = 'bvec'


class HSComponentFile(HSFile):
    """A class used to import and export spectral data using plain ASCII files.

    Objects of this class may be used to load and save a collection of spectral
    datasets which share the same wavelength information. The datasets may be
    or arbitrary dimension, such as a single spectral density or
    three-dimensional hyperspectral images.

    Attributes
    ----------
    file_path : str
        The full path to the intput file.
    _buf_info :  dict
        A buffer for the following items:

            - title (str), A description for the data collection.
            - version (str), The hsi package version.
            - date (datetime), The date of creation
            - hsformat (:obj:`hsi.HSFormatFlag`), The spectral hsformat.

    _buf_spectra :  dict of numpy.ndarray
        A buffer for spectral data, one dictionary item for each dataset.
    _buf_components :  dict of tuples
        A buffer for metadata of spectral vectors, one dictionary item for each
        vector where the key correspond to the vector's name. The tuples
        contain the following items in the given order:

            - label (str), The label of the base spectrum.
            - weight (str), The weight for the base spectrum.
            - lower bound (float), The lower for the scaling factor.
            - upper bound (float), The upper for the scaling factor.

    _buf_wavelen :  numpy.ndarray
        A buffer for the wavelengths at which the spectral data are sampled.
    """

    def __init__(self, file_path=None, hsformat=HSFormatDefault, title=None):
        """ Constructor

        Parameters
        ----------
        file_path :  str
            The absolute path to the input file.
        hsformat :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The hsformat for the hyperspectral data. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`

        title :  str, optional
            A brief description of the data collection to be set.

        """
        super(HSComponentFile, self).__init__(file_path, hsformat, title)

        # spectral base vectors
        self._buf_components = {}  # dictionary using vector's name as key

    def clear(self):
        self._buf_spectra.clear()
        self._buf_components.clear()
        self._buf_wavelen = None
        logger.debug("Clear hsvectorfile ")

    def buffer(self, *args, **kwargs):
        """Add spectral data to the internal buffers.

        Parameters
        ----------
        spectra : list, numpy.ndarray, :class:`HSVector<hsi.analysis.HSVector>`
            The spectral data.
        wavelen : numpy.ndarray, optional
            The wavelengths at which the spectral data are sampled. Not used if
            spectra is of type :class:`HSVector<hsi.analysis.HSVector>`
        label : str, optional
            The label of the dataset. Not used if spectra is of type
            :class:`HSVector<hsi.analysis.HSVector>`
        hsformat :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`
            The hsformat for the hyperspectral data. Not used if spectra is of
            type :class:`HSVector<hsi.analysis.HSVector>`


        If non-keyword arguments are used, they will be interpreted as
        buffer(spectra) for a single argument, buffer(spectra, wavelen) for two
        arguments, buffer(spectra, wavelen, label) for three arguments, and
        buffer(spectra, wavelen, label, hsformat) for four arguments.

        Forward all arguments to either of the following methods depending on
        the type of **spectra**

        ============= ==========================================================
        list          :func:`buffer_data()<hsi.analysis.HSVectorFile.buffer_data>`
        numpy.ndarray :func:`buffer_data()<hsi.analysis.HSVectorFile.buffer_data>`
        HSVector      :func:`buffer_component()<hsi.analysis.HSVectorFile.buffer_component>`
        ============= ==========================================================


        """
        if len(args):
            spectra = args[0]
        else:
            spectra = kwargs.get('spectra', None)

        if isinstance(spectra, (list, np.ndarray)):
            self.buffer_data(*args, **kwargs)
        elif isinstance(spectra, HSComponent):
            self.buffer_component(*args, **kwargs)

    def buffer_component(self, spectra):
        """Add the data of a spectral base vector to the buffer.

        Parameters
        ----------
        spectra :  :obj:`hsi.analysis.HSVector`
            The spectral data.
        """
        if not isinstance(spectra, HSComponent) or not spectra.shape:
            logger.debug(
                "Empty vector '{}'. Skip writing to buffer".format(spectra))
            return -1

        # store spectral information of the vector in buffer 'LABEL_VECTOR'
        ret = self.buffer_data(spectra.yIntpData, spectra.xIntpData,
                               hsformat=spectra.hsformat, label=LABEL_VECTOR)
        if ret != 0:
            logger.debug(
                "Could not append vector {} to buffer. ".format(spectra.name))
            return

        # store metadata of vector in separate buffer
        self._buf_components[spectra.name] = (
            spectra.label, spectra.weight, spectra.bounds[0], spectra.bounds[1])

    def load(self):
        """Load spectral information from a text file in the internal buffers.
        """
        self._bufInfo["title"] = None
        self._bufInfo["version"] = "hsi " + __version__
        self._bufInfo["date"] = datetime.datetime.now(),
        self._bufInfo["hsformat"] = HSFormatDefault
        self.clear()  # clear any previously defined spectral datasets

        if self._filePath is None:
            return

        if os.path.isfile(self._filePath):
            fpath = self._filePath
        else:
            fpath = os.path.join(getPkgDir(), "data", self._filePath)
            if not os.path.isfile(fpath):
                logger.debug("File '%s' not found." % self._filePath)
                return

        logger.debug("Open file {}.".format(fpath))
        with open(fpath, 'r') as file:
            info = self.read_header_info(file)
            if info['version'] is None:
                logger.debug("No valid hsi input file.")
            elif info['hsformat'] is None:
                logger.debug("Unknown spectral hsformat '%s'." % self._filePath)
            else:
                metadata = self.read_metadata(file, skiprows=1)
                if LABEL_VECTOR in metadata:
                    vectors = self.read_component_table(
                        file, metadata, skiprows=3)
                else:
                    vectors = {}
                spectra, wavelen = self.read_data_table(
                    file, metadata, skiprows=3)
                self._bufInfo.update(info)
                self._buf_components.update(vectors)
                self._buf_spectra.update(spectra)
                self._buf_wavelen = wavelen

    def read(self):
        """Read the spectral information from a text file."""
        self.load()

        if not self._buf_spectra:
            logger.debug("No spectral loaded.")
            return {}, {}, None

        wavelen = self._buf_wavelen.view()
        components = dict()

        if LABEL_VECTOR in self._buf_spectra:
            spectra = self._buf_spectra[LABEL_VECTOR].view()
            for i, (key, value) in enumerate(self._buf_components.items()):
                components[key] = HSComponent(
                    spectra[:, i], wavelen, wavelen, name=key, label=value[0],
                    hsformat=self.hsformat, weight=value[1], bounds=value[2:4])

        spectra = {key: value for key, value in self._buf_spectra.items()
                   if key != LABEL_VECTOR}

        return components, spectra, wavelen

    @staticmethod
    def read_component_table(file, metadata, skiprows=0):
        """ Read the vector table from a file

        Parameters
        ----------
        file : file object
            An object exposing a file-oriented API (with methods such as read()
            or write()) to an underlying resource.
        metadata : dict
            Provides the label and shape tuple for each dataset.
        skiprows : int, optional
            Skip the first skiprows lines; default: 0.

        Returns
        -------
        vectors :  dict of tuple
            A dictionary of parameters for each spectral base vector, where the
            key correspond to the vector's name. The tuples contain the
            following items in the given order:

                - label (str), The label of the base spectrum.
                - weight (float), The weight for the base spectrum.
                - lower bound (float), The lower for the scaling factor.
                - upper bound (float), The upper for the scaling factor.


        """
        while skiprows:
            file.readline()
            skiprows = skiprows - 1

        if LABEL_VECTOR not in metadata:
            return {}

        components = dict()
        nwavelen, nspectra = metadata[LABEL_VECTOR]

        for i in range(nspectra):
            sval = file.readline().split()
            if len(sval) == 5:
                name = sval[1]
                label = sval[1]
                weight = float(sval[2])
                lower_bound = float(sval[3])
                upper_bound = float(sval[4])
                components[name] = (label, weight, lower_bound, upper_bound)

            elif len(sval) == 6:
                name = sval[1]
                label = sval[2]
                weight = float(sval[3])
                lower_bound = float(sval[4])
                upper_bound = float(sval[5])
                components[name] = (label, weight, lower_bound, upper_bound)

        return components

    def write(self, *args, **kwargs):
        """Write the spectral information from the internal buffers into a
        text file.

        Forward all arguments to :func:`buffer()<hsi.HSFile.buffer>` before
        writing.
        """
        self.buffer(*args, **kwargs)
        if not self._buf_spectra:
            logger.debug("No spectral available to export.")
            return

        with open(self._filePath, 'w') as file:
            self.write_header_info(file, self._bufInfo)
            file.write("\n#\n")
            self.write_metadata(file, self._buf_spectra, self._buf_wavelen)
            if len(self._buf_components):
                file.write("\n#\n")
                self.write_component_table(file, self._buf_components)
            file.write("\n#\n")
            self.write_data_table(file, self._buf_spectra, self._buf_wavelen)

    @staticmethod
    def write_component_table(file, vectors):
        """ Write the parameter table of all vectors in a file.

        Parameters
        ----------
        file : file object
            An object exposing a file-oriented API (with methods such as read()
            or write()) to an underlying resource.
        vectors :  dict of tuple
            A dictionary of parameters for each spectral base vector, where the
            key correspond to the vector's name. The tuples contain the
            following items in the given order:

                - label (str), The label of the base spectrum.
                - weight (float), The weight for the base spectrum.
                - lower bound (float), The lower for the scaling factor.
                - upper bound (float), The upper for the scaling factor.

        """
        if not isinstance(vectors, dict):
            logger.debug("Argument 'vectors' is invalide.")
            return -1
        for key, value in vectors.items():
            if not isinstance(value, tuple) or len(value) != 4:
                logger.debug("Argument 'vectors' is invalide.")
                return -1

        logger.debug("Write parameter table for vectors.")

        file.write("%-21s %s" % ("# Vector Table:", LABEL_VECTOR))
        file.write("\n%-24s  %-24s  %-24s  %-24s  %-24s  %s" % (
            "# No.", "Name", "Label", "Weight", "Lower bound", "Upper bound"))
        fmt = "\n{:<24}  {:<24}  {:<24}  {:<24.16g}  {:<24.16g}  {:.16g}"
        for i, (key, value) in enumerate(vectors.items()):
            label = value[0]
            weight = value[1]
            lbnd = value[2] if value[2] is not None else -np.inf
            ubnd = value[3] if value[3] is not None else np.inf
            file.write(fmt.format(i, key, label, weight, lbnd, ubnd))
