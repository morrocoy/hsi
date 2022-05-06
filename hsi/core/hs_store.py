# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:53:04 2021

@author: kpapke
"""
import os.path
# import pathlib

# import re

import numpy
# import numpy.lib.recfunctions as rfn
# import pandas as pd

# import h5py
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

HSPatientInfo = numpy.dtype([
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
        PatientInfo = numpy.dtype([
            ("pn", '<i8'),
            ("name", 'S32'),
            ("age", '<i4'),
        ])

        # example data
        data = numpy.array([
            (0, b"Smith", 43),
            (1, b"Jones", 19),
            (2, b"Williams", 51)], dtype=PatientInfo)

        # open a file in "w"rite mode
        with HSStore.open("test.h5", mode="w", path="/records") as store:

            # create table in hdf5 file in group /records
            tablePatient = store.create_table(
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
            return numpy.random.random((10, 10))


        if __name__ == '__main__':
            # open an hdf5 file in "r+" mode
            with tables.open_file("test.h5", "r+") as file:
                reader = HSStore(file, path="/records")
                writer = HSStore(file, path="/records")

                # attach table for reading
                reader.attache_table("patient")

                # create table in group /records with 10x10pts images
                table = writer.create_table(
                    name="analysis",
                    dtype=numpy.dtype([
                        ("oxy", "<f8", (10, 10)),
                    ]),
                    title="Analysis of oxygen saturation",
                    expectedrows=len(reader),
                )
                row = table.row

                print(f"Tables to read: {reader.get_table_names()}")
                print(f"Tables to write: {writer.get_table_names()}")
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
        # if isinstance(fname, (str, pathlib.Path)):
        if isinstance(fname, (str)):
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

    def attache_table(self, name):
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

        # for node in self.file.iter_nodes(self.path):
            # node = self.file.get_node(self.path + "/" + name)

        path = self.path + "/" + name
        if self.file.__contains__(path):
            node = self.file.get_node(path)

            # direct table
            if isinstance(node, tables.table.Table) and node.name == name:
                logger.debug(f"Attach table {name}.")
                self.tables[name] = node
                return node

            # link to a table in an external file
            elif isinstance(node, tables.link.ExternalLink):
                link = node()
                if isinstance(link, tables.table.Table) and link.name == name:
                    logger.debug(f"Attach externally linked table '{name}'.")
                    self.tables[name] = link
                    return link

            else:
                logger.debug(f"Node {name} is not referring to a table.")
                return None

        else:
            logger.debug(f"Table {name} not found.")
            return None

    def remove_table(self, name):
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

    def create_table(self, name, dtype, title="", expectedrows=10000,
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

        # remove table if already defined
        if self.has_table(name):
            self.remove_table(name)

        logger.debug(f"Create table {name} with columns {dtype}.")
        table = self.file.create_table(
            self.path, name=name, description=dtype, title=title,
            expectedrows=expectedrows, chunkshape=chunkshape,
        )
        self.tables[name] = table
        return table

    def detach_table(self, name):
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

    def get_table(self, name):
        """ Retrieve a table object from the selected list if available.

        Parameters
        ----------
        name : str
            The table name.
        """
        return self.tables.get(name, None)

    def get_table_names(self):
        """ Return the list of selected table names.
        """
        return self.tables.keys()

    def has_table(self, name):
        """ Returns True if

        """
        path = self.path + "/" + name
        if self.file.__contains__(path):
            node = self.file.get_node(path)
            if isinstance(node, tables.table.Table):
                return True
            elif isinstance(
                    node, (tables.link.ExternalLink, tables.link.SoftLink)):
                link = node()
                if isinstance(link, tables.table.Table):
                    return True

        return False

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
            parent, node_name = path.rsplit("/", 1)
            if parent == "":
                parent = "/"
            logger.debug(f"Create node {node_name} in {parent}.")
            node = self.file.create_group(
                parent, node_name, createparents=createparents)

            return node

        else:
            return self.file.get_node(path)

    def link(self, file_path, path="/", table_names=None):
        """provide the dataset with a link to an external file.

        Parameters
        ----------
        file_path : str, or pathlib.Path
            The path to the file to be linked with.
        path : str, optional
            The path within the underlying hdf5 file. Default is the root path.
        table_names : list or tuple, optional
            A list of table names to link. By default all existing tables
            within 'path' will be linked.
        """
        with HSStore(os.path.abspath(file_path), mode="r", path=path) as reader:
            for node in reader.file.iter_nodes(self.path, classname='Table'):
                if table_names is None or node.name in table_names:
                    self.file.create_external_link(
                        path, node.name, target=node, createparents=True)

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
    def open(file_path, mode='r', path="/", descr=None):
        """ Create a new HSStore object and leave the file open for further
        processing.

        Parameters
        ----------
        file_path : str or pathlib.Path
            The File, filepath, or generator to read.
        mode : {'a', 'w', 'r', 'r+'}, optional
            The mode in which the file is opened. Default is 'r'.
        path : str, optional
            The path within the underlying hdf5 file. Default is the root path.
        descr : str, optional
            A description for the dataset. Only used in writing mode.
        """
        return HSStore(file_path, mode=mode, path=path, descr=descr)
