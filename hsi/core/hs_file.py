# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 07:55:39 2021

@author: kpapke
"""
import os.path
import re
from ast import literal_eval
import datetime
import numpy

from .. import __version__

from ..log import logmanager
from ..misc import getPkgDir

from .hs_formats import HSFormatFlag, HSFormatDefault, convert

logger = logmanager.getLogger(__name__)

__all__ = ['HSFile']

LABEL_DEFAULT = 'spec'


class HSFile(object):
    """A class used to import and export spectral data using plain ASCII files.

    Objects of this class may be used to load and save a collection of spectral
    datasets which share the same wavelength information. The datasets may be
    or arbitrary dimension, such as a single spectral density or
    three-dimensional hyperspectral images.

    Attributes
    ----------
    file_path : str
        The full path to the intput file.
    _bufInfo :  dict
        A buffer for the following items:

            - title (str), A description for the data collection.
            - version (str), The hsi package version.
            - date (datetime), The date of creation
            - hsformat (:obj:`hsi.HSFormatFlag`), The spectral hsformat.

    _buf_spectra :  dict of numpy.ndarray
        A buffer for spectral data, one dictionary item for each dataset.
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
        self._filePath = file_path

        # file header information
        self._bufInfo = {
            'title': title,  # description of collection
            'version': "hsi " + __version__,  # current hsi package version
            'date': datetime.datetime.now(),  # date of creation
            'hsformat': hsformat,  # spectral hsformat
        }
        # buffer for spectral datasets and wavelength
        self._buf_spectra = {}  # dictionary using dataset's label as key
        self._buf_wavelen = None

    def __enter__(self):
        logger.debug("HSFile object __enter__().")
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        logger.debug("HSFile object __exit__().")
        self.close()

    def buffer(self, *args, **kwargs):
        """Add spectral data to the internal buffers.

        Parameters
        ----------
        spectra :  list, numpy.ndarray
            The spectral data.
        wavelen :  numpy.ndarray, optional
            The wavelengths at which the spectral data are sampled.
        label : str, optional
            The label of the dataset
        hsformat :  :obj:`hsi.HSFormatFlag`
            The hsformat for the hyperspectral data.

        If non-keyword arguments are used, they will be interpreted as
        buffer(spectra) for a single argument, buffer(spectra, wavelen) for two
        arguments, buffer(spectra, wavelen, label) for three arguments, and
        buffer(spectra, wavelen, label, hsformat) for four arguments.
        """
        if len(args):
            spectra = args[0]
        else:
            spectra = kwargs.get('spectra', None)

        if isinstance(spectra, (list, numpy.ndarray)):
            self.buffer_data(*args, **kwargs)

    def buffer_data(self, spectra, wavelen=None, label=LABEL_DEFAULT,
                    hsformat=HSFormatDefault):
        """Add a spectral dataset to the buffer or extend an existing one.

        Parameters
        ----------
        spectra :  list, numpy.ndarray
            The spectral data.
        wavelen :  numpy.ndarray, optional
            The wavelengths at which the spectral data are sampled.
        label : str, optional
            The label of the dataset
        hsformat :  :obj:`hsi.HSFormatFlag`
            The hsformat for the hyperspectral data.
        """

        # check hsformat, if not previously defined also set the hsformat
        if not HSFormatFlag.has_flag(hsformat):
            raise Exception("Unknown hsformat '{}'.".format(hsformat))

        # prepare spectral data (two- or higher-dimensional array)
        if isinstance(spectra, list):
            spectra = numpy.array(spectra)
        if not isinstance(spectra, numpy.ndarray) or spectra.ndim < 1:
            raise Exception("Argument 'spectra' must be ndarray of at "
                            "least one dimension.")
        if spectra.ndim == 1:
            spectra = spectra[:, numpy.newaxis]

        # raise error if no wavelength information are available
        if wavelen is None and self._buf_wavelen is None:
            logger.debug("Wavelength information is not available. "
                         "Skip writing to buffer")
            return -1

        # add first dataset which defines the wavelength samples
        elif wavelen is not None and self._buf_wavelen is None:
            if isinstance(wavelen, list):
                wavelen = numpy.array(wavelen)
            if not isinstance(wavelen, numpy.ndarray) or wavelen.ndim > 1:
                raise Exception("Argument 'wavelen' must be 1D ndarray.")
            if len(spectra) != len(wavelen):
                raise Exception("Arguments 'spectra' and 'wavelen' must "
                                "be of same length.")

            logger.debug("Add spectral data set '{}' of shape {}.".format(
                label, spectra.shape))
            self._buf_wavelen = wavelen.view(numpy.ndarray)
            self._buf_spectra[label] = convert(
                self.hsformat, hsformat, spectra, self._buf_wavelen)
            return 0

        # add a new dataset or extend an existing one
        else:
            if len(spectra) != len(self._buf_wavelen):
                raise Exception("Arguments 'spectra' and 'wavelen' must "
                                "be of same length.")
            logger.debug("Add spectral data set '{}' of shape {}.".format(
                label, spectra.shape))

            if label in self._buf_spectra.keys():
                self._buf_spectra[label] = numpy.append(
                    self._buf_spectra[label],
                    convert(self.hsformat, hsformat, spectra, self._buf_wavelen),
                    axis=-1)
                return 0
            else:
                self._buf_spectra[label] = convert(
                    self.hsformat, hsformat, spectra, self._buf_wavelen)
                return 0

    def clear(self):
        self._buf_spectra.clear()
        self._buf_wavelen = None
        logger.debug("Clear hsfile.")

    def close(self):
        self.clear()
        self._bufInfo["title"] = None
        self._bufInfo["version"] = None
        self._bufInfo["date"] = None
        self._bufInfo["hsformat"] = None

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
                spectra, wavelen = self.read_data_table(
                    file, metadata, skiprows=3)
                self._bufInfo.update(info)
                self._buf_spectra.update(spectra)
                self._buf_wavelen = wavelen

    @staticmethod
    def parse_string(tag, string):
        """Get a tagged substring from a string

        Parameters
        ----------
        tag : str
            The identifier for the value.
        string : str
            The input string.

        Returns
        -------
        str
            The substring after the tag. If no substring was found, the
            function returns None.
        """
        match = re.search(r'%s\s+(.*)' % tag, string)
        if not match:
            return None
        else:
            return match.group(1).strip(",.:\'")

    @staticmethod
    def parse_value(tag, string):
        """Evaluate a tagged value from a string

        Parameters
        ----------
        tag : str
            The identifier for the value.
        string : str
            The input string.

        Returns
        -------
        int, float, list
            The evaluated value after the tag. If no value was found, the
            function returns None.
        """
        match = re.search(r'%s\s+(.*)' % tag, string)
        if not match:
            return None

        sval = match.group(1)
        regex = r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?'
        match = re.findall(regex, sval)

        n = len(match)
        if n > 1:
            return [literal_eval(s) for s in match]
        elif n == 1:
            return literal_eval(match[0])
        else:
            return None

    @property
    def date(self):
        """datetime.datetime: The date of creation."""
        return self._bufInfo['date']

    @property
    def hsformat(self):
        """:obj:`hsi.HSFormatFlag`: The hsformat of the spectral data. """
        return self._bufInfo['hsformat']

    @property
    def spectra(self):
        """dict: A dictionary of spectral data, one entry for each dataset."""
        return self._buf_spectra

    @property
    def title(self):
        """str: Brief description of the data collection."""
        return self._bufInfo['title']

    @property
    def version(self):
        """str: The hsi file version."""
        return self._bufInfo['version']

    @property
    def wavelen(self):
        """numpy.ndarray: The wavelengths at which spectral data are sampled."""
        return self._buf_wavelen

    def read(self):
        """Read the spectral information from a text file."""
        self.load()

        if not self._buf_spectra:
            logger.debug("No spectral loaded.")
            return {}, None

        wavelen = self._buf_wavelen.view()
        spectra = {key: value for key, value in self._buf_spectra.items()}

        return spectra, wavelen

    @classmethod
    def read_header_info(cls, file, skiprows=0):
        """ Read the header information from file

        Parameters
        ----------
        file : file object
            An object exposing a file-oriented API (with methods such as read()
            or write()) to an underlying resource.
        skiprows : int, optional
            Skip the first skiprows lines; default: 0.

        Returns
        -------
        dict
            A dictionary containing the following items:

                - title (str), A description for the data collection.
                - version (str), The hsi package version.
                - date (datetime), The date of creation
                - hsformat (:obj:`hsi.HSFormatFlag`), The spectral hsformat.

        """
        while skiprows:  # skip header lines
            file.readline()
            skiprows = skiprows - 1

        info = dict()

        # brief description of data collection
        info['title'] = cls.parse_string("Description:", file.readline())
        logger.debug("Read info description: {}.".format(info['title']))

        # hsi package version
        version = cls.parse_string("Version:", file.readline())
        regex = r'^hsi [0-9].[0-9].[0-9]+$'
        match = re.findall(regex, version)
        if match:
            info["version"] = version
            logger.debug("Read info version: {}.".format(info['version']))
        else:
            info["version"] = None
            logger.debug("Read info version: invalid.")

        # date of creation
        sdate = cls.parse_string("Date:", file.readline())
        info['date'] = datetime.datetime.strptime(sdate, "%b %d %Y, %H:%M")
        logger.debug("Read info date: {}.".format(info['date'].strftime(
            "%b %d %Y, %H:%M")))

        # spectral hsformat
        sformat = cls.parse_string("Format:", file.readline())
        info['hsformat'] = HSFormatFlag.from_str(sformat)
        if info['hsformat'] is None:
            logger.debug("Read info hsformat: None.")
        else:
            logger.debug("Read info hsformat: {}.".format(info['hsformat'].key))

        return info

    @classmethod
    def read_metadata(cls, file, skiprows=0):
        """ Read the shape information for ech dataset contained in a file

        Parameters
        ----------
        file : file object
            An object exposing a file-oriented API (with methods such as read()
            or write()) to an underlying resource.
        skiprows : int, optional
            Skip the first skiprows lines; default: 0.

        Returns
        -------
        dict
            A dictionary whose keys and values represent the label or shape
            tuple, respectively, for each dataset.
        """
        while skiprows:  # skip header lines
            file.readline()
            skiprows = skiprows - 1

        # read data shape information
        nwavelen = cls.parse_value("Wavelengths:", file.readline())
        ndataset = cls.parse_value("Datasets:", file.readline())

        # read label and shape tuple of each dataset
        metadata = {}
        for i in range(ndataset):
            line = file.readline()
            line_items = line.split(':')
            label = cls.parse_string("Set", line_items[0])
            value = cls.parse_value("Set '%s':" % label, line)

            # interprete integer number n as tuple (nwavelen, n)
            if isinstance(value, int):
                shape = (nwavelen, value)  # ensure at least two dims
            # interprete list of integer as tuple (nwavelen, *list)
            elif isinstance(value, (list, numpy.ndarray)):
                shape = (nwavelen, ) + tuple(value)
            # interprete tuple of integer as tuple (nwavelen, *tuple)
            elif isinstance(value, tuple):
                shape = (nwavelen, ) + value
            else:
                shape = (nwavelen, 0)  # Unknown shape information

            metadata[label] = shape
            logger.debug("Read metadata for dataset '{}' of shape {}.".format(
                label, shape))

        return metadata

    @staticmethod
    def read_data_table(file, metadata=None, skiprows=0, maxrows=None):
        """ Read the data table from a file

        Parameters
        ----------
        file : file object
            An object exposing a file-oriented API (with methods such as read()
            or write()) to an underlying resource.
        metadata : list, tuple, dict
            Provides the label and shape tuple for each dataset.
        skiprows : int, optional
            Skip the first skiprows lines; default: 0.
        maxrows : int, optional
            Read maxrows lines of content after skiprows lines. The default
            is to read all the lines.

        Returns
        -------
        spectra :  dict
            The spectral datasets. A dictionary whose keys and values represent
            the label or data, respectively, for each dataset.
        wavelen :  numpy.ndarray
            The wavelengths at which the spectral data are sampled.
        """
        # skip header lines
        while skiprows:
            file.readline()
            skiprows = skiprows - 1

        # determine maximum number of rows to read
        if metadata is None and isinstance(maxrows, int):
            rows = maxrows
        elif isinstance(metadata, (list, tuple)):
            rows = metadata[0]
        elif isinstance(metadata, dict):
            rows = min([shape[0] for shape in metadata.values()])
        else:
            rows = 0

        if maxrows is None or rows < maxrows:
            maxrows = rows

        # read data table
        if maxrows > 0:
            data = numpy.array(
                [line.strip().split() for i, line in enumerate(file) if
                 (i < maxrows)], dtype=float)
        else:
            data = numpy.array(
                [line.strip().split() for line in file], dtype=float)
        wavelen = data[:, 0]
        spectra = data[:, 1:]

        # retrieve datasets from table
        datasets = {}
        rows, cols = spectra.shape
        newshape = (rows, cols)
        # reshape table according to a shape information provided by a tuple
        if isinstance(metadata, (list, tuple)) and len(metadata) > 1:
            if cols == numpy.prod(metadata[1:]):
                newshape = (rows,) + metadata[1:]
            logger.debug("Read dataset '{}' of shape {}.".format(
                LABEL_DEFAULT, newshape))
            datasets[LABEL_DEFAULT] = spectra.reshape(newshape)

        # split and reshape table according to shape tuples provided by a dict
        elif isinstance(metadata, dict):
            mcols = numpy.sum(
                numpy.prod(shape[1:]) for shape in metadata.values()
                if isinstance(shape, (list, tuple)) and len(shape) > 1)
            if cols == mcols:
                i = 0
                for label, shape in metadata.items():
                    n = numpy.prod(shape[1:])
                    newshape = (rows,) + shape[1:]
                    logger.debug("Read dataset '{}' of shape {}.".format(
                        label, newshape))
                    if n > 0:
                        datasets[label] = spectra[:, i:i+n].reshape(newshape)
                    else:
                        datasets[label] = numpy.empty(newshape)
                    i = i + n

            else:
                logger.debug("Read dataset '{}' of shape {}.".format(
                    LABEL_DEFAULT, newshape))
                datasets[LABEL_DEFAULT] = spectra.reshape(newshape)

        # keep flat table hsformat if metadata are not defined or inconsistent
        else:
            logger.debug("Read dataset '{}' of shape {}.".format(
                LABEL_DEFAULT, newshape))
            datasets[LABEL_DEFAULT] = spectra.reshape(newshape)

        return datasets, wavelen

    def set_format(self, hsformat):
        """Set the hsformat of the the spectral data.

        Parameters
        ----------
        hsformat : :obj:`HSFormatFlag<hsi.HSFormatFlag>`
            The spectral hsformat to be set. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`

        """
        if not HSFormatFlag.has_flag(hsformat):
            raise Exception("Unknown hsformat '{}'.".format(hsformat))

        old_format = self.hsformat
        for key in self._buf_spectra.keys():
            self._buf_spectra[key] = convert(
                hsformat, old_format, self._buf_spectra[key], self._buf_wavelen)

        self._bufInfo['hsformat'] = hsformat

    def set_title(self, title):
        """Set the hsformat of the the spectral data.

        Parameters
        ----------
        title : str
            A brief description of the data collection to be set.
        """
        self._bufInfo['title'] = title

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
            file.write("\n#\n")
            self.write_data_table(file, self._buf_spectra, self._buf_wavelen)

    @staticmethod
    def write_header_info(file, info):
        """Write the header information to a file.

        Parameters
        ----------
        file : file object
            An object exposing a file-oriented API (with methods such as read()
            or write()) to an underlying resource.
        info : dict
            A dictionary containing the following items:

                - title (str), A description for the data collection.
                - version (str), The hsi package version.
                - date (datetime), The date of creation
                - hsformat (:obj:`hsi.HSFormatFlag`), The spectral hsformat.

        """
        title = info.get('title', None)
        version = info.get('version', "hsi " + __version__)
        date = info.get('date', datetime.datetime.now())
        hsformat = info.get('hsformat', None)

        if title is None:
            title = ""
        if isinstance(date, datetime.datetime):
            sdate = date.strftime("%b %d %Y, %H:%M")
        else:
            sdate = date
        if isinstance(hsformat, HSFormatFlag):
            sformat = hsformat.key
        else:
            sformat = "None"

        logger.debug("Write info description: {}.".format(title))
        logger.debug("Write info version: {}.".format(version))
        logger.debug("Write info date: {}.".format(sdate))
        logger.debug("Write info hsformat: {}.".format(sformat))

        file.write("{:<21} {:}".format("# Title:", title))
        file.write("\n{:<21} {:}".format("# Version:", version))
        file.write("\n{:<21} {:}".format("# Date:", sdate))
        file.write("\n{:<21} {:}".format("# Format:", sformat))

    @staticmethod
    def write_metadata(file, spectra, wavelen):
        """Write the shape information for each dataset in a file

        Parameters
        ----------
        file : file object
            An object exposing a file-oriented API (with methods such as read()
            or write()) to an underlying resource.
        spectra :  numpy.ndarray or dict of numpy.ndarray
                A dictionary of spectral data, one entry for each dataset.
        wavelen :  numpy.ndarray
            The wavelengths at which the spectral data are sampled.
        """
        nwavelen = len(wavelen)

        # conform spectral datasets
        if isinstance(spectra, dict):
            datasets = spectra
        elif isinstance(spectra, numpy.ndarray):
            datasets = {LABEL_DEFAULT: spectra}
        elif isinstance(spectra, list):
            datasets = {LABEL_DEFAULT: numpy.array(spectra)}
        else:
            logger.debug("Undefined datasets. Skip writing")
            return

        buffer = ""
        buffer += "{:<21} {:}".format("# Wavelengths:", nwavelen)
        buffer += "\n{:<21} {:}".format("# Datasets:", len(datasets))

        # extract shape information
        for key, value in datasets.items():
            if isinstance(value, list):
                value = numpy.array(value)
            elif not isinstance(value, numpy.ndarray) or len(value) != nwavelen:
                logger.debug("Inconsistent shape information for dataset '{}'. "
                             "Skip writing".format(key))
                return

            if value.ndim > 1:
                shape = value.shape[1:]
            else:
                shape = (1, )
            logger.debug("Write metadata for dataset '{}' of shape {}.".format(
                key, shape))
            buffer += "\n{:<21} {}".format("# Set '%s':" % key, shape)

        file.write(buffer)

    @staticmethod
    def write_data_table(file, spectra, wavelen):
        """ Write the data table as a collection of all datasets in a file.

        Parameters
        ----------
        file : file object
            An object exposing a file-oriented API (with methods such as read()
            or write()) to an underlying resource.
        spectra :  numpy.ndarray or dict of numpy.ndarray
                A dictionary of spectral data, one entry for each dataset.
        wavelen :  numpy.ndarray
            The wavelengths at which the spectral data are sampled.
        """
        nwavelen = len(wavelen)

        # conform spectral datasets
        if isinstance(spectra, dict):
            datasets = spectra
        elif isinstance(spectra, numpy.ndarray):
            datasets = {LABEL_DEFAULT: spectra}
        elif isinstance(spectra, list):
            datasets = {LABEL_DEFAULT: numpy.array(spectra)}
        else:
            logger.debug("Undefined datasets. Skip writing")
            return

        # ravel spectral data sets to flat two-dimensional arrays
        rdatasets = {}
        for key, value in datasets.items():
            if isinstance(value, list):
                value = numpy.array(value)
            elif not isinstance(value, numpy.ndarray) or len(value) != nwavelen:
                logger.debug("Inconsistent shape information for dataset '{}'. "
                             "Skip writing".format(key))
                return

            if value.ndim > 1:
                value = value.reshape(nwavelen, -1)
            else:  # must be one-dimensional
                value = value[:, numpy.newaxis]  # ensure two dimensions
            rdatasets[key] = value

        # write title
        file.write("%-21s %s\n" % ("# Table:", "Spectral data"))

        # write table headline
        header_items = ["# Wavelength [m]"]
        for key, value in rdatasets.items():
            m, n = value.shape
            header_items.extend(["%s %d" % (key, i) for i in range(n)])
        header_line = ""
        for item in header_items[:-1]:
            header_line += "%-24s  " % item
        header_line += "%s" % header_items[-1]
        file.write(header_line)

        # write table data
        data = numpy.hstack([value for value in rdatasets.values()])
        m, n = data.shape
        logger.debug("Write data collection of shape '{}'.".format((m, n)))
        fmt = "\n" + "  ".join(["{:<24.15g}"] * n + ["{:.15g}"])
        for i in range(m):
            file.write(fmt.format(wavelen[i], *data[i]))
