# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:53:04 2021

@author: kpapke
"""
import os.path
import re
import numpy as np
import pandas as pd
import h5py

from .formats import HSFormatFlag, HSFormatDefault, convert

from ..misc import getPkgDir

import logging

LOGGING = True
# LOGGING = False
logger = logging.getLogger(__name__)
logger.propagate = LOGGING


__all__ = ['HSDataset']

class HSDataset(object):

    def __init__(self, filePath, format=None):
        """Constructor.

        Parameters
        ----------
        filePath :  str
            The absolute path to the input file.
        format : :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The spectral format to be set. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`

        """
        # source file
        self._file = None

        # information of dataset
        self.filePath = None  # path to the input file
        self.descr = None  # descrption of the dataset
        self.groups = []  # groups in the h5 file referring to hs data
        self.metadata = None  # dataframe of metadata for all records

        # self.pids = []  # patient id
        # self.dates = []  # date
        # self.hsimages = []  # hyperspectral images
        # self.masks = []  # selection masks applied on the image
        # self.notes = []  # notes

        # hyperspectral image of all records
        self.format = None
        self.featureNames = None
        self.targets = []
        self.targetNames = None

        # load metadata dataset if file path is defined
        self.load(filePath)


    def __enter__(self):
        logger.debug("HSDataset object __enter__().")
        return self


    def __exit__(self, exception_type, exception_value, traceback):
        logger.debug("HSDataset object __exit__().")
        self.close()


    def __getitem__(self, index):

        # return column of metadata if index is an appropriate key
        if isinstance(index, str) and index in self.metadata.columns.values:
            return self.metadata[index]

        # return entry if index is integer
        elif isinstance(index, int):
            return self.select(index)
        else:
            return None


    def clear(self):
        """Clear any loaded data. """
        logger.debug("Clear head.")
        self.filePath = None
        self.descr = None
        self.metadata = None
        self.groups.clear()


    def close(self):
        """Close the internally loaded file and clean up any related data."""
        if self._file is not None:
            logger.debug("Close file {}.".format(self.filePath))
            self._file.close()
        self.clear()


    def items(self):
        for i in range(len(self.metadata.index)):
            yield tuple(self.select(i))


    def load(self, filePath):
        """Open the source file and load metadata for the dataset and its
        entries.

        Parameters
        ----------
        filePath :  str
            The absolute path to the input file.
        """
        self.clear()
        if os.path.isfile(filePath):
            fpath = filePath
        else:
            fpath = os.path.join(getPkgDir(), "data", filePath)
            if not os.path.isfile(fpath):
                logger.debug("File '%s' not found." % (filePath))
                return

        # retrieve pd.dataframe of metadata from file
        with pd.HDFStore(fpath, 'r') as store:
            if '/metadata' in store.keys():
                logger.debug("Load metadata from {}".format(fpath))
                self.metadata = store['metadata']
            else:
                logger.debug("File '%s' does not contain metadata." % (filePath))
                return

        # open file for continues use
        file = h5py.File(fpath, 'r')

        # dataset description
        keys = file.keys()
        if 'descr' in keys:
            self.descr = file['descr'][()]
        else:
            self.descr = None

        # groups containing hyperspectral data (format: yyyy-mm-dd-HH-MM-SS)
        pattern = re.compile("^\d{4}(-\d{2}){5}$")
        self.groups = [key for key in keys if pattern.match(key)]

        # keep reference to file
        self.filePath = fpath
        self._file = file


    def select(self, index):
        """Retrieve the hyperspectral data of the selected entry.

        """
        if index >= len(self.metadata.index):
            raise Exception("Index Error: {}.".format(index))

        # get series of metadata
        df = self.metadata.iloc[index]

        key = df['group']
        if key not in self.groups:
            logger.debug(
                "No data available for entry {}: {}.".format(index, key))
            return None, None, None

        logger.debug("Load data of entry {}: {}.".format(index, key))
        group = self._file[key]
        keys = group.keys()
        akeys = group.attrs.keys()

        spectra = group['spectra'][()] if 'spectra' in keys else None
        wavelen = group['wavelen'][()] if 'wavelen' in keys else None
        masks = group['masks'][()] if 'masks' in keys else None

        sformat = group.attrs['format'] if 'format' in akeys else None
        format = HSFormatFlag.fromStr((sformat))

        # series of complementary metadata
        df2 = pd.Series(format, index=['format'], dtype=object)

        return df.append(df2), spectra, wavelen, masks


    def setFormat(self, format):
        """Set the format for the hyperspectral data.

        Parameters
        ----------
        format : :obj:`HSFormatFlag<hsi.HSFormatFlag>`
            The spectral format to be set. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`

        """
        if not HSFormatFlag.hasFlag(format):
            raise Exception("Unknown format '{}'.".format(format))
        self.format = format



    def setTargetNames(self, names):
        self.targetNames = names

