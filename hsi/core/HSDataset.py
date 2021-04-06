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
import tables


from ..log import logmanager
from ..misc import getPkgDir

logger = logmanager.getLogger(__name__)

__all__ = ['HSDataset']


class PatientInfo(tables.IsDescription):
    pn = tables.Int64Col()  # Signed 64-bit integer
    pid = tables.Int64Col()  # Signed 64-bit integer
    name = tables.StringCol(32)  # 32-byte character string (utf-8)
    descr = tables.StringCol(64)  # 64-byte character String (utf-8)
    timestamp = tables.StringCol(32)  # 32-byte character String (utf-8)
    hsformat = tables.StringCol(32)  # 32-byte character String (utf-8)
    target = tables.Int32Col()  # Signed 32-bit integer


class HSImageData(tables.IsDescription):
    wavelen = tables.Float64Col(shape=(100, ))  # array of 64-bit floats
    spectra = tables.Float32Col(shape=(100, 480, 640))  # array of 32-bit floats
    masks = tables.Int8Col(shape=(5, 480, 640))  # array of bytes


class HSDataset:
    """Class to to iterate through the patient and hsidata tables"""

    def __init__(self, fname, node="/records", mode="r"):
        """Constructor.

        Parameters
        ----------
        fname : file, str, or pathlib.Path
            The File, filepath, or generator to read.
        mode : str, optional
            The mode in which the file is opened. Default is "rb".
        """
        self.file = None  # file handle
        self.mode = mode  # opening mode
        self.owner = False  # ownership of the underlying file
        self.node = node  # node in hdf5 file

        self.patientInfo = None
        self.hsImageData = None

        self._index = 0

        # hyperspectral image of all records
        # self.featureNames = None
        # self.targets = []
        self.targetNames = ["not healed", "healed"]

        # underlying dataset file already open
        if hasattr(fname, "read") and hasattr(fname, "mode"):
            self.file = fname
            self.mode = fname.mode
            self.owner = False  # no ownership of the underlying file

        # open underlying dataset file
        else:
            logger.debug("Open file object {} in mode {}.".format(fname, mode))
            self.file = tables.open_file(fname, mode)
            self.mode = mode
            self.owner = True  # has ownership of the underlying file
            if self.mode in ("r", "rb"):
                self.load()
            elif self.mode in ("w", "wb"):
                self.file.create_group(self.node)


    def __enter__(self):
        logger.debug("HSDataset object __enter__().")
        return self


    def __exit__(self, exception_type, exception_value, traceback):
        logger.debug("HSDataset object __exit__().")
        self.close()


    def __getitem__(self, index):

        # return column of tables if index equals a column name
        if isinstance(index, str):
            if index in self.patientInfo.keys:
                return self.patientInfo[index]
            elif index in self.hsImageData.keys:
                return self.hsImageData[index]
            else:
                raise KeyError("Key {} not found in dataset.".format(index))

        # return entry if index is integer
        elif isinstance(index, int):
            return self.select(index)
        else:
            return None


    def __iter__(self):
        self._index = 0
        return self


    def __len__(self):
        if isinstance(self.patientInfo, tables.Table):
            return self.patientInfo.nrows
        else:
            return 0


    def __next__(self):
        if self._index < self.__len__():
            result = self.select(self._index)
            self._index += 1
            return result
        raise StopIteration  # end of Iteration


    def append(self, info, spectra, wavelen, masks, results=None):
        if self.mode in ("w", "wb"):
            patientInfo = self.patientInfo.row  # get a pointer to the Row
            hsImageData = self.hsImageData.row

            patientInfo["pn"] = info["pn"]  # i f'Particle: {i:6d}'
            patientInfo["pid"] = info["pid"]
            patientInfo["name"] = str.encode("")
            patientInfo["descr"] = str.encode(info["descr"])
            patientInfo["timestamp"] = str.encode(info["timestamp"])
            patientInfo["hsformat"] = str.encode(info["hsformat"])
            patientInfo["target"] = info["target"]
            patientInfo.append()  # writes record to the table I/O buffer

            hsImageData["spectra"] = spectra.astype(np.float32)  # float32
            hsImageData["wavelen"] = wavelen.astype(np.float64)  # float64
            hsImageData["masks"] = masks.astype(np.int8)  # signed byte,
            hsImageData.append()  # writes record to the table I/O buffer

            # flush the table’s I/O buffer to write all this data to disk
            self.patientInfo.flush()
            self.hsImageData.flush()


    def clear(self):
        """Clear any loaded data. """
        logger.debug("Clear head.")
        self.filePath = None
        self.descr = None
        # self.metadata = None
        # self.groups.clear()


    def close(self):
        """Close the internally loaded file and clean up any related data."""
        if self.file is not None and self.owner:
            logger.debug("Close file {}.".format(self.file.filename))
            self.file.close()
        self.clear()


    def initTables(self, node="/records", descr=None, expectedrows=None):
        if self.file is None and self.mode in ("w", "wb"):
            h5file = tables.open_file(self.filePath, mode="w")

            # Create a new group
            group = h5file.create_group("/", "records")
            group._v_attrs.descr = descr  # description of dataset

            # Creating a new table
            self.patientInfo = h5file.create_table(
                group,
                name="patient",
                description=PatientInfo,
                title="Patient information",
                # chunkshape=None,
            )

            self.hsImageData = h5file.create_table(
                group,
                name="hsidata",
                description=HSImageData,
                title="Hyperspectral image data",
                expectedrows=expectedrows,
                # chunkshape=None,
            )

            self.file = h5file


    def load(self):
        self.patientInfo = self.file.get_node(self.node + "/patient")
        self.hsImageData = self.file.get_node(self.node + "/hsidata")


    def select(self, index):
        """Retrieve information and hyperspectral image data of a patient.
        """
        if index < self.__len__():
            return (
                self.patientInfo[index], self.hsImageData[index])
        else:
            raise Exception("Index Error: {}.".format(index))


    @staticmethod
    def open(filePath, node="/records", mode='r'):
        """ Creates a new mch File object and reads metadata, leaving the file
        open to allow reading data chunks

        Parameters
        ----------
        filePath : str
            The path to the file as a string or pathlib.Path.
        mode : str, optional
            The mode in which the file is opened. Default is 'r'.
        """
        return HSDataset(filePath, node=node, mode=mode)



# class HSDataset2(object):
#
#     def __init__(self, filePath):
#         """Constructor.
#
#         Parameters
#         ----------
#         filePath :  str
#             The absolute path to the input file.
#
#         """
#         # source file
#         self._file = None
#
#         # information of dataset
#         self.filePath = None  # path to the input file
#         self.descr = None  # descrption of the dataset
#         self.groups = []  # groups in the h5 file referring to hs data
#         self.metadata = None  # dataframe of metadata for all records
#
#         # self.pids = []  # patient id
#         # self.dates = []  # date
#         # self.hsimages = []  # hyperspectral images
#         # self.masks = []  # selection masks applied on the image
#         # self.notes = []  # notes
#
#         # hyperspectral image of all records
#         self.featureNames = None
#         self.targets = []
#         self.targetNames = None
#
#         # load metadata dataset if file path is defined
#         self.load(filePath)
#
#
#     def __enter__(self):
#         logger.debug("HSDataset object __enter__().")
#         return self
#
#
#     def __exit__(self, exception_type, exception_value, traceback):
#         logger.debug("HSDataset object __exit__().")
#         self.close()
#
#
#     def __getitem__(self, index):
#
#         # return column of metadata if index is an appropriate key
#         if isinstance(index, str) and index in self.metadata.columns.values:
#             return self.metadata[index]
#
#         # return entry if index is integer
#         elif isinstance(index, int):
#             return self.select(index)
#         else:
#             return None
#
#
#     def __len__(self):
#         if isinstance(self.metadata, pd.DataFrame):
#             return len(self.metadata.index)
#         else:
#             return 0
#
#
#     def clear(self):
#         """Clear any loaded data. """
#         logger.debug("Clear head.")
#         self.filePath = None
#         self.descr = None
#         self.metadata = None
#         self.groups.clear()
#
#
#     def close(self):
#         """Close the internally loaded file and clean up any related data."""
#         if self._file is not None:
#             logger.debug("Close file {}.".format(self.filePath))
#             self._file.close()
#         self.clear()
#
#
#     def items(self):
#         for i in range(len(self.metadata.index)):
#             yield tuple(self.select(i))
#
#
#     def load(self, filePath):
#         """Open the source file and load metadata for the dataset and its
#         entries.
#
#         Parameters
#         ----------
#         filePath :  str
#             The absolute path to the input file.
#         """
#         self.clear()
#         if os.path.isfile(filePath):
#             fpath = filePath
#         else:
#             fpath = os.path.join(getPkgDir(), "data", filePath)
#             if not os.path.isfile(fpath):
#                 logger.debug("File '%s' not found." % (filePath))
#                 return
#
#         # retrieve pd.dataframe of metadata from file
#         with pd.HDFStore(fpath, 'r') as store:
#             if '/metadata' in store.keys():
#                 logger.debug("Load metadata from {}".format(fpath))
#                 self.metadata = store['metadata']
#             else:
#                 logger.debug("File '%s' does not contain metadata." % (filePath))
#                 return
#
#         # open file for continues use
#         file = h5py.File(fpath, 'r')
#
#         # dataset description
#         keys = file.keys()
#         if 'descr' in keys:
#             self.descr = file['descr'][()]
#         else:
#             self.descr = None
#
#         # groups containing hyperspectral data (format: yyyy-mm-dd-HH-MM-SS)
#         pattern = re.compile("^\d{4}(-\d{2}){5}$")
#         self.groups = [key for key in keys if pattern.match(key)]
#
#         # keep reference to file
#         self.filePath = fpath
#         self._file = file
#
#
#     def select(self, index):
#         """Retrieve the hyperspectral data of the selected entry.
#
#         """
#         if index >= len(self.metadata.index):
#             raise Exception("Index Error: {}.".format(index))
#
#         # get series of metadata
#         df = self.metadata.iloc[index]
#
#
#         key = df['group']
#         if key not in self.groups:
#             logger.debug(
#                 "No data available for entry {}: {}.".format(index, key))
#             return None, None, None
#
#         logger.debug("Load data of entry {}: {}.".format(index, key))
#         group = self._file[key]
#         keys = group.keys()
#         # akeys = group.attrs.keys()
#
#         spectra = group['spectra'][()] if 'spectra' in keys else None
#         wavelen = group['wavelen'][()] if 'wavelen' in keys else None
#         maskarr = group['masks'][()] if 'masks' in keys else None
#
#         masks = {label: maskarr[i] for i, label in enumerate([
#             "tissue", "critical wound region", "wound region",
#             "wound and proximity"])}
#
#
#
#         # sformat = group.attrs['format'] if 'format' in akeys else None
#         # series of complementary metadata
#         # df2 = pd.Series(format, index=['format'], dtype=object)
#         return spectra, wavelen, masks, df
#
#
#     def setTargetNames(self, names):
#         self.targetNames = names

