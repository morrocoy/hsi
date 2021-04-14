# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:53:04 2021

@author: kpapke
"""
import os.path
import pathlib

import re

import numpy as np
import numpy.lib.recfunctions as rfn
import pandas as pd

import h5py
import tables


from ..log import logmanager


logger = logmanager.getLogger(__name__)

__all__ = ["HSStore", "HSPatientInfo"]


# class HSPatientInfo(tables.IsDescription):
#     pn = tables.Int64Col()  # Signed 64-bit integer
#     pid = tables.Int64Col()  # Signed 64-bit integer
#     name = tables.StringCol(32)  # 32-byte character string (utf-8)
#     descr = tables.StringCol(64)  # 64-byte character String (utf-8)
#     timestamp = tables.StringCol(32)  # 32-byte character String (utf-8)
#     target = tables.Int32Col()  # Signed 32-bit integer

# class HSImageData(tables.IsDescription):
#     wavelen = tables.Float64Col(shape=(100,))  # array of 64-bit floats
#     spectra = tables.Float32Col(
#         shape=(100, 480, 640))  # array of 32-bit floats
#     masks = tables.Int8Col(shape=(5, 480, 640))  # array of bytes

HSPatientInfo = np.dtype([
    ("pn", '<i8'),
    ("pid", '<i8'),
    ("name", 'S32'),
    ("descr", 'S64'),
    ("timestamp", 'S32'),
    ("target", '<i4'),
])



class HSStore:
    """ A dictionary-like IO interface for storing datasets in HDF5 files.

    It makes extensive use of pytables (https://www.pytables.org/).

    Attributes
    ----------
    file : :obj:`tables.file.File`
        The underlying file.
    mode : {'a', 'w', 'r', 'r+'}
        The mode in which the file is opened. Default is 'r'.

        ``'r'``
            Read-only; no data can be modified.
        ``'w'``
            Write; a new file is created (an existing file with the same
            name would be deleted).
        ``'a'``
            Append; an existing file is opened for reading and writing,
            and if the file does not exist it is created.
        ``'r+'``
            It is similar to ``'a'``, but the file must already exist.
    owner :  bool
        True if ownership of the underlying file is provided.
    path :  str
        The path within the hdf5 file.
    tables : dict
        A dictionary of tables selected in the hdf5 file.
    index : int
        The current row of the selected tables providing the dataset.


    Examples
    --------

    Create a dataset using a structured numpy array

    .. code-block:: python
        :emphasize-lines: 19

        import numpy as np
        import tables
        from hsi import HSStore

        # define columns for the table
        PatientInfo = np.dtype([
            ("pn", '<i8'),
            ("name", 'S32'),
            ("age", '<i4'),
        ])

        # example data
        data = np.array([
            (0, b"Smith", 43),
            (1, b"Jones", 19),
            (2, b"Williams", 51)], dtype=PatientInfo)

        # open a file in "w"rite mode
        with HSStore.open("test.h5", mode="w", path="/records") as store:

            # create table in hdf5 file in group /records
            tablePatient = store.createTable(
                name="patient",
                dtype=PatientInfo,
                title="Patient information",
                expectedrows=3,
            )

            # get the record object associated with the table
            row = tablePatient.row

            # fill the table
            for entry in data:
                row["pn"] = entry["pn"]
                row["name"] = entry["name"]
                row["age"] = entry["age"]

                # inject the record values
                row.append()

            # flush the table buffers
            tablePatient.flush()

        # file is automatically closed when using the with statement
        # (this also will flush all the remaining buffers)


    Process a dataset by reading an existing and writing a new table.

    .. code-block:: python
        :emphasize-lines: 24,25,26

        import numpy as np
        import tables
        from hsi import HSStore
        import multiprocessing

        # analysis function takes the selected entries of the attached tables
        # in the reader and returns some pseudo data to be written in new table.
        def fun(args):
            # get input table entry
            patient = args[0]

            print("%8d | %-20s | %3d |" % (
                patient["pn"],
                patient["name"].decode(),
                patient["age"],
            ))

            # return analysis results
            return np.random.random((10, 10))


        if __name__ == '__main__':
            # open a file in "r+" mode
            with tables.open_file("test.h5", "r+") as file:
                reader = HSStore(file, path="/records")
                writer = HSStore(file, path="/records")

                # attach table for reading
                reader.attacheTable("patient")

                # create table in hdf5 file in group /records providing 10x10pts images
                table = writer.createTable(
                    name="analysis",
                    dtype=np.dtype([
                        ("oxy", "<f8", (10, 10)),
                    ]),
                    title="Analysis of oxygen saturation",
                    expectedrows=len(reader),
                )
                row = table.row

                print(f"Tables to read: {reader.getTableNames()}")
                print(f"Tables to write: {writer.getTableNames()}")
                print(f"Number of entries: {len(reader)}")

                # serial evaluation
                for args in iter(reader):
                    res = fun(args)
                    row["oxy"] = res
                    row.append()

                # parallel evaluation (requires to safely imported the main
                # module using if __name__ == '__main__')
                pool = multiprocessing.Pool(processes=3)
                for res in pool.imap(fun, iter(reader)):
                    row["oxy"] = res
                    row.append()
                pool.close()

                # flush the table buffers
                table.flush()
    """

    def __init__(self, fname, mode="r", path="/", descr=None):
        """Constructor.

        Parameters
        ----------
        fname : file, str, or pathlib.Path
            The File, filepath, or generator to read.
        mode : {'a', 'w', 'r', 'r+'}
            The mode in which the file is opened. Default is 'r'. Only used if
            fname corresponds to a filepath.
        path : str, optional
            The path within the underlying hdf5 file. Default is the root path.
        descr : str, optional
            A description for the dataset. Only used in writing mode.

        """
        self.file = None  # file handle
        self.mode = mode  # opening mode
        self.owner = False  # ownership of the underlying file
        self.path = path  # path within the hdf5 file

        self.tables = {}  # dictionary of tables
        self.index = 0

        # open underlying dataset file
        if isinstance(fname, (str, pathlib.Path)):
            logger.debug(f"Open file object {fname} in mode {mode}.")
            self.file = tables.open_file(fname, mode)
            self.mode = mode
            self.owner = True  # has ownership of the underlying file

        # underlying dataset file already open
        elif isinstance(fname, tables.file.File):
            logger.debug("Retrieve file object {} with mode {}.".format(
                fname.filename, fname.mode))
            self.file = fname
            self.mode = fname.mode
            self.owner = False  # no ownership of the underlying file

        else:
            raise ValueError("Argument fname must be a file, filepath, or a"
                             "generator to read or write")

        # check whether internal path to dataset exists in read only mode
        if self.mode in ("r", "rb") and not self.file.__contains__(self.path):
            raise("Path {} to dataset does not exist.".format(self.path))

        # create internal path for dataset if not available in any write mode
        elif self.mode in ("w", "wb", "a", "r+"):
            node = self.mkdir(self.path)
            # description of dataset
            if descr is not None:
                logger.debug(
                    f"Set description for dataset in directory {self.path}.")
                node._v_attrs.descr = descr


    def __enter__(self):
        logger.debug("HSStore object __enter__().")
        return self


    def __exit__(self, exception_type, exception_value, traceback):
        logger.debug("HSStore object __exit__().")
        self.close()


    def __getitem__(self, index):
        """ Returns the entry of the attached tables specified by the index.

        Parameters
        ----------
        index : int
            The tables row.

        Returns
        -------
        record : tuple
            A tuple of partial records comprised by the selected tables in the
            order of attachment or creation at the specified row.
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
            return table.nrows
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


    def attacheTable(self, name):
        """ Attach an existing table in the hdf5 file to the store object.

        Parameters
        ----------
        name : str
            The table name.

        Returns
        -------
        table or None
            The table if exists otherwise None.
        """
        if self.file is None or self.mode in ("w", "wb"):
            return None

        keys = [node.name for node in self.file.iter_nodes(
            self.path, classname='Table')]
        if name in keys:
            logger.debug(f"Attach table {name}.")
            table = self.file.get_node(self.path + "/" + name)
            self.tables[name] = table
            return table
        else:
            logger.debug(f"Table {name} not found.")
            return None


    def removeTable(self, name):
        """ Remove a table from the underlying hdf5 file if existing.

        Note: the file size is not reduced by this operation. The reference to
        the table node is removed and the space in the file becomes available
        for future use. The new table should preferably have the same
        description as the removed one.

        Parameters
        ----------
        name : str
            The table name.
        """
        if self.file is None or self.mode in ("r", "rb"):
            logger.debug(f"Cannot remove table {name} due to read only mode.")
            return

        keys = [node.name for node in self.file.iter_nodes(
            self.path, classname='Table')]
        if name in keys:
            logger.debug(f"Remove table {name}.")
            table = self.file.get_node(self.path + "/" + name)
            table.remove()
        else:
            logger.debug(f"Table {name} not found.")


    def clear(self):
        """ Clear any loaded data and detach all tables"""
        logger.debug("Clear head and detach all tables.")
        self.tables.clear()


    def close(self):
        """ Close the underlying hdf5 file and clean up any related data.

        Note: the file is only closed if the object provides ownership.
        """
        if self.file is not None and self.owner:
            logger.debug(f"Close file {self.file.filename}.")
            self.file.close()
        self.clear()


    def createTable(self, name, dtype, title=None, expectedrows=None,
                    chunkshape=None):
        """ Create a new table in the hdf5 file.

        Parameters
        ----------
        name : str
            The table name.
        dtype : numpy.dtype
            A structured datatypes to describe the table columns.
        title : str, optional
            A description for the table. It sets the TITLE HDF5 attribute.
        expectedrows : int, optional
            A user estimate of the number of records that will be in the table.
            If not provided, the default value of pytables is use.
        chunkshape : tuple, optional
            The shape of the data chunk to be read or written in a single HDF5
            I/O operation. Filters are applied to those chunks of data. The
            rank of the chunkshape for tables must be 1. If None, a sensible
            value is calculated based on the expectedrows parameter (which is
            recommended).
        Returns
        -------
        table, None
            The table if exists otherwise None.
        """
        if self.file is None or self.mode in ("r", "rb"):
            return None

        self.removeTable(name)  # remove table if already defined
        logger.debug(f"Create table {name} with columns {dtype}.")
        table = self.file.create_table(
            self.path, name=name, description=dtype, title=title,
            expectedrows=expectedrows, chunkshape=chunkshape,
        )
        self.tables[name] = table
        return table


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


    def mkdir(self, path, createparents=True):
        """ Create a directory within the hdf5 file.

        Parameters
        ----------
        path : str
            The absolute directory path within the file.
        createparents : bool
            A flag to automatically create the parent directories.
        """
        if not self.file.__contains__(path):
            parent, nodename = path.rsplit("/", 1)
            if parent == "":
                parent = "/"
            logger.debug(f"Create node {nodename} in {parent}.")
            node = self.file.create_group(
                parent, nodename, createparents=createparents)

            return node

        else:
            return self.file.get_node(path)


    def select(self, index):
        """ Internal function to access a specific row of all attached tables.

        Parameters
        ----------
        index : int
            The tables row.
        """
        if index < self.__len__():
            return [table[index] for table in self.tables.values()]

            # return (
            #     self.patientInfo[index], self.hsImageData[index])
            # return rfn.merge_arrays(
            #     [self.patientInfo[index], self.hsImageData[index]],
            #     flatten = True, usemask = False)[0]
        else:
            raise Exception("Index Error: {}.".format(index))


    @staticmethod
    def open(filePath, mode='r', path="/", descr=None):
        """ Create a new HSStore object and leave the file open for further
        processing.

        Parameters
        ----------
        fname : str or pathlib.Path
            The File, filepath, or generator to read.
        mode : {'a', 'w', 'r', 'r+'}, optional
            The mode in which the file is opened. Default is 'r'.
        path : str, optional
            The path within the underlying hdf5 file. Default is the root path.
        descr : str, optional
            A description for the dataset. Only used in writing mode.
        """
        return HSStore(filePath, mode=mode, path=path, descr=descr)



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

