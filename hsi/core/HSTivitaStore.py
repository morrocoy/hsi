# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:53:04 2021

@author: kpapke
"""
import sys
import os.path
import pathlib

import re

import numpy as np
import numpy.lib.recfunctions as rfn
import pandas as pd
import tables

from hsi import HSIntensity, HSAbsorption
from hsi import HSImage


from ..log import logmanager


logger = logmanager.getLogger(__name__)

__all__ = ["HSTivitaStore"]


class HSTivitaStore:
    """ A dictionary-like IO interface providing read access Tivita datasets.

    Attributes
    ----------
    file : :obj:`tables.file.File`
        The underlying file.
    owner :  bool
        True if ownership of the underlying file is provided.
    path :  str
        The relative path from the root directory.
    index : int
        The current row of the selected tables providing the dataset.

    """

    def __init__(self, fname, path="/"):
        """Constructor.

        Parameters
        ----------
        fname : file, str, or pathlib.Path
            The descriptor File, filepath, or generator to read.
        path : str, optional
            The path within the underlying hdf5 file. Default is the root path.
        """
        self.file = None  # file handle
        self.owner = False  # ownership of the underlying file
        self.root = None  # absolute root path
        self.path = path  # path relative to root

        self.tables = {}  # dictionary of tables
        self.index = 0

        # open underlying dataset descriptor file
        if isinstance(fname, (str, pathlib.Path)):
            logger.debug(f"Open descriptor file object {fname}.")
            self.file = pd.ExcelFile(fname)  # tables.open_file(fname, mode)
            self.owner = True  # has ownership of the underlying file

        # underlying dataset file already open
        elif isinstance(fname, pd.ExcelFile):
            logger.debug("Retrieve file object {}.".format(
                fname.io))
            self.file = fname
            self.owner = False  # no ownership of the underlying file

        else:
            raise ValueError("Argument fname must be a pandas.ExcelFile, "
                             "filepath, or a generator to read or write")

        self.root = os.path.dirname(os.path.abspath(self.file.io))
        self.path = os.path.join(self.root, *path.split("/"))

        # check whether internal path to dataset exists
        if not os.path.isdir(self.path):
            raise(f"Path to dataset not found ({path} -> {self.path}).")



    def __enter__(self):
        logger.debug("HSTivitaStore object __enter__().")
        return self


    def __exit__(self, exception_type, exception_value, traceback):
        logger.debug("HSTivitaStore object __exit__().")
        self.close()


    def __getitem__(self, index):
        """ Returns the entry of the dataset specified by the index.

        Parameters
        ----------
        index : int
            The tables row.

        Returns
        -------
        record : dict
            A dictionary of the selected data items.
        """
        # return column of tables if index equals a column name
        if isinstance(index, str):
            for table in self.tables.values():
                if index in table.keys:
                    return table[index]
            else:
                raise KeyError(f"Key {index} not found in dataset.")

        # return entry if index is integer
        elif isinstance(index, int):
            return self.select(index)
        else:
            return None


    def __iter__(self):
        """ Returns an iterator on the object to iterate through the rows of
        attached tables. """
        self.index = 0
        return self


    def __len__(self):
        """ Returns the row count of the attached tables comprising the dataset.
        """
        if len(self.tables):
            logger.debug("Get row count of the attached tables.")
            table = next(iter(self.tables.values()))
            return len(table)
        else:
            return 0


    def __next__(self):
        """ Returns the next entry of the attached tables in an iteration. """
        if self.index < self.__len__():
            logger.debug(f"Get next entry of attached tables ({self.index}).")
            result = self.select(self.index)
            self.index += 1
            return result
        raise StopIteration  # end of Iteration


    def attacheTable(self, name, dtype, sheet_name=0, header=0, usecols=None,
                     skiprows=None, nrows=None):
        """ Attach an existing table from the excel file.

        Parameters
        ----------
        name : str
            The table name.
        dtype : numpy.dtype
            A structured datatypes to describe the table columns.
        sheet_name : int or str, optional
            The index or name of the worksheet within the excel file.
        header : int, list of int, default 0
            Row (0-indexed) to use for the column labels of the parsed
            DataFrame.
        usecols : int, str, list-like, or callable default None
            The columns to use in the excel sheet.
        skiprows : int, optional
            The number of rows to skip.
        nrows : int
            The number of rows to read.
        """
        # converters = {name: dtype[name] for name in dtype.names}
        df = self.file.parse(
            sheet_name=sheet_name,
            header=header,
            usecols=usecols, skiprows=skiprows, nrows=nrows,
            # dtype=dtype,
            # converters=converters,
            # encoding=sys.getfilesystemencoding()
            encoding='utf-8',
        )
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # overwrite header
        column_names = [dtype.names[i] for i in np.argsort(usecols)]
        df.rename(
            columns={old: new for old, new in zip(df.columns, column_names)},
            inplace=True
        )
        # df = df.astype(converters)
        # table = df#.to_records(index=False)#column_dtypes=dtype) #.to_numpy()
        # retrieve timestamp from group name
        # df['timestamp'] = pd.to_datetime(
        #     df['group'], format="%Y-%m-%d-%H-%M-%S")

        # convert dataframe to numpy record array using specified datatype
        table = np.empty(len(df), dtype=dtype)
        for index, row in df.iterrows():
            for column_name in dtype.names:
                if dtype[column_name].kind == 'S':
                    table[index][column_name] = str.encode(row[column_name])
                else:
                    table[index][column_name] = row[column_name]

        self.tables[name] = table


    def clear(self):
        """ Clear any loaded data and detach all tables"""
        logger.debug("Clear head and detach all tables.")
        self.tables.clear()


    def close(self):
        """ Close the underlying hdf5 file and clean up any related data.

        Note: the file is only closed if the object provides ownership.
        """
        if self.file is not None and self.owner:
            logger.debug(f"Close file {self.file.io}.")
            self.file.close()
        self.clear()


    def detachTable(self, name):
        """ Remove a table from the selected list.

        Parameters
        ----------
        name : str
            The table name.
        """
        if name in self.tables.keys():
            logger.debug(f"Detach table {name}.")
            self.tables.pop(name)
        else:
            logger.debug(f"Table {name} not found.")


    def getTable(self, name):
        """ Retrieve a table object from the selected list if available.

        Parameters
        ----------
        name : str
            The table name.
        """
        return self.tables.get(name, None)


    def getTableNames(self):
        """ Return the list of selected table names.
        """
        return self.tables.keys()


    def readHSImage(self, path):
        parent, node_name = path.rsplit(os.sep, 1)
        filePath = os.path.join(parent, node_name, node_name + "_SpecCube.dat")

        hsImage = HSImage(filePath)
        nwavelen, rows, cols = hsImage.shape
        hsformat = HSIntensity

        hsImage.setFormat(hsformat)

        # add gaussian image filter for a cleaner tissue selection mask
        hsImage.addFilter(mode='image', type='gauss', sigma=1, truncate=4)

        # create the record array
        # record = np.array(
        #     [(str.encode(hsformat.key), hsImage.wavelen, hsImage.spectra)],
        #     dtype = np.dtype([
        #         ("hsformat", "<S32"),
        #         ("wavelen", "<f8", (nwavelen,)),
        #         ("spectra", "<f4", (nwavelen, rows, cols))
        #     ])
        # )

        record = {
            "hsformat": str.encode(hsImage.format.key),
            "wavelen": hsImage.wavelen.astype("<f8"),
            "spectra": hsImage.spectra.astype("<f4")
        }
        return record


    def readMasks(self, path):
        parent, node_name = path.rsplit(os.sep, 1)
        filePath = os.path.join(parent, node_name, node_name + "_Masks.npz")
        masks = np.load(filePath)

        return {name: masks[name] for name in masks.files}


    def dropMasksAtIndex(self):
        pass


    def select(self, index):
        """ Internal function to access a specific row of all attached tables.

        Parameters
        ----------
        index : int
            The tables row.
        """
        if index < self.__len__():
            record = []
            for table in self.tables.values():
                patient = table[index]
                node_name = patient["timestamp"].decode().replace('-', '_')
                path = os.path.join(self.path, node_name)
                hsimage = self.readHSImage(path)
                masks = self.readMasks(path)
                record.append((patient, hsimage, masks))

            if len(record) == 1:
                return record[0]
            else:
                return record

        else:
            raise Exception("Index Error: {}.".format(index))


    @staticmethod
    def open(filePath, path="/"):
        """ Create a new HSTivitaStore object and leave the file open for
        further processing.

        Parameters
        ----------
        fname : str or pathlib.Path
            The File, filepath, or generator to read.
        path : str, optional
            The path within the underlying hdf5 file. Default is the root path.
        """
        return HSTivitaStore(filePath, path=path)
