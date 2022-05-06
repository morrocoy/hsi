# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:53:04 2021

@author: kpapke
"""
# import sys
import os.path
# import pathlib

# import re

import numpy as np
# import numpy.lib.recfunctions as rfn
import pandas as pd
import cv2
# import tables

from .hs_formats import HSIntensity
from .hs_image import HSImage
from .hs_store import HSStore

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

        self.skipNameColumn = True  # flag to enable skipping column of names
        self.overwriteMasks = False  # flag to enable overwriting the mask files
        self.markerColor = [100, 255, 0]  # color of the selection contour

        self.maskconfig = {
            "critical": "_RGB_ROI_kritisch.png",
            "wound": "_RGB_ROI_Wunde.png",
            "proximity": "_RGB_ROI_Wundumgebung.png",
        }

        # open underlying dataset descriptor file
        # if isinstance(fname, (str, pathlib.Path)):
        if isinstance(fname, (str)):
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

    def attache_table(self, name, dtype, sheet_name=0, header=0, usecols=None,
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
        logger.debug(f"Attach table {name} from excel sheet.")
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
        #     df['group'], hsformat="%Y-%m-%d-%H-%M-%S")

        # convert dataframe to numpy record array using specified datatype
        table = np.empty(len(df), dtype=dtype)
        for index, row in df.iterrows():
            for column_name in dtype.names:
                if column_name == "name" and self.skipNameColumn:
                    table[index][column_name] = b''
                elif dtype[column_name].kind == 'S':
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

    def create_masks(self, path):
        parent, node_name = path.rsplit(os.sep, 1)
        logger.debug(f"Create masks for record {node_name}.")

        file_path = os.path.join(parent, node_name, node_name + "_SpecCube.dat")
        hsimage = HSImage(file_path)
        hsimage.set_format(HSIntensity)

        # add gaussian image filter for a cleaner tissue selection mask
        hsimage.add_filter(mode='image', filter_type='gauss', sigma=1, truncate=4)
        masks = {"tissue": hsimage.get_tissue_mask([0.1, 0.9])}  # tissue mask)

        # extract remaining masks from userdefined contours in images
        for name, node_suffix in self.maskconfig.items():
            mask, image = self.find_mask(os.path.join(
                parent, node_name, node_name + node_suffix),
                self.markerColor,
            )
            masks[name] = mask * masks["tissue"]

        # write masks as numpy files
        file_path = os.path.join(parent, node_name, node_name + "_Masks.npz")
        logger.debug(f"Write masks for record {node_name} in {file_path}.")
        np.savez(file_path, **masks)

        return masks

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

    @staticmethod
    def find_mask(file_path, marker_color=[100, 255, 0]):
        """ Extract a contour from an RGB image and derive a mask from it.

        Parameters
        ----------
        file_path : str, or pathlib.Path
            The path to the RGB image.
        marker_color: tuple, list
            A 3-element tuple or list describing the RGB color used to select
            the polygon. Default is the Tivita marker color [100, 255, 0].
        """
        if not os.path.isfile(file_path):
            logger.debug(f"WARNING: Image file {file_path} not found.")
            print(f"WARNING: Image file {file_path} not found.")
            return [], []

        logger.debug(f"Read image file {file_path}.")
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)

        rows, cols, channels = img.shape
        logger.debug(f"Image shape: {img.shape}.")
        if rows == 507 and cols == 645:
            # img = img[27:27+480, 3:3+640]
            img = img[27:27 + 480, 3:3 + 640]

        logger.debug(f"Select outermost polygon of color {marker_color}.")
        img_marked = img.copy()
        rows, cols, channels = img.shape

        # convert marker color from rgb to bgr
        marker_color = np.array(marker_color[::-1])

        # external (outermost) contours from color
        mask_color = cv2.inRange(img_marked, marker_color, marker_color)
        contours, hierarchy = cv2.findContours(mask_color, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

        # create a single channel black image
        logger.debug(f"Derive mask from polygon.")
        mask = np.zeros((rows, cols), dtype='<i1')
        cv2.fillPoly(mask, pts=contours,
                     color=1)  # set 1 within the contour

        # transform image from bgr to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return mask, img

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

    @staticmethod
    def read_hsimage(path):
        """ Internal function to read the hyperspectral data for the selected
        record.

        Parameters
        ----------
        path : str
            The path to the Tivita record data stored.
        """
        parent, node_name = path.rsplit(os.sep, 1)
        logger.debug(f"Read hyperspectral data for record {node_name}.")

        file_path = os.path.join(parent, node_name, node_name + "_SpecCube.dat")
        hsimage = HSImage(file_path)
        # nwavelen, rows, cols = hsimage.shape
        hsformat = HSIntensity

        hsimage.set_format(hsformat)

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
            "hsformat": str.encode(hsimage.hsformat.key),
            "wavelen": hsimage.wavelen.astype("<f8"),
            "spectra": hsimage.spectra.astype("<f4")
        }
        return record

    def read_masks(self, path):
        """ Internal function to read the mask file for the selected record.

        Parameters
        ----------
        path : str
            The path to the Tivita record data stored.
        """
        parent, node_name = path.rsplit(os.sep, 1)

        file_path = os.path.join(parent, node_name, node_name + "_Masks.npz")

        # for suffix in ["Mask.npy", "Masks.npz", "Masks2.npz"]:
        #     fpath = os.path.join(parent, node_name, node_name + "_" + suffix)
        #     if os.path.isfile(fpath):
        #         os.unlink(fpath)

        if os.path.isfile(file_path) and not self.overwriteMasks:
            logger.debug(f"Read selection masks for record {node_name}.")
            masks = np.load(file_path)
            return {name: masks[name] for name in masks.files}

        else:
            return self.create_masks(path)

    def select(self, index):
        """ Internal function to access a specific row of all attached tables.

        Parameters
        ----------
        index : int
            The tables row.
        """
        if index < self.__len__():
            table = next(iter(self.tables.values()))
            patient = table[index]

            # tivita defines the subfolder and file prefix by the timestamp
            node_name = patient["timestamp"].decode().replace('-', '_')
            path = os.path.join(self.path, node_name)
            logger.debug(f"Select record {node_name}.")

            # read hyperspectral data
            hsimage = self.read_hsimage(path)

            # read masks
            masks = self.read_masks(path)
            return (patient, hsimage, masks)

        else:
            raise Exception("Index Error: {}.".format(index))

    def to_hdf(self, fname, path="/", descr=None, nrows=None):
        """ Export attached table and associated data to an hdf5 file.

        Parameters
        ----------
        fname : file, str, or pathlib.Path
            The File, filepath, or generator to read.
        path : str, optional
            The path within the underlying hdf5 file. Default is the root path.
        descr : str, optional
            A description for the dataset. Only used in writing mode.
        nrows : int, optional
            The number of rows to be written.
        """
        expectedrows = self.__len__()
        if nrows is not None and nrows <= expectedrows:
            expectedrows = nrows

        if expectedrows <= 0:
            logger.debug(f"Empty table. Nothing to export.")
            return

        # reference to patient info table
        table_name, table = next(iter(self.tables.items()))

        # read first record to determine image size
        patient, hsimage, masks = self.select(0)
        nwavelen, rows, cols = hsimage["spectra"].shape

        with HSStore(fname, mode="w", path=path, descr=descr) as writer:

            # table of patient information
            table_patient = writer.create_table(
                name=table_name,
                dtype=table.dtype,
                title="Patient information",
                expectedrows=expectedrows,
            )

            # table of hyperspectral image data
            table_hsimage = writer.create_table(
                name="hsimage",
                dtype=np.dtype([
                    ("hsformat", "<S32"),
                    ("wavelen", "<f8", (nwavelen,)),
                    ("spectra", "<f4", (nwavelen, rows, cols))
                ]),
                title="Hyperspectral image data",
                expectedrows=expectedrows,
            )

            # table of masks to be applied on the hyperspectral image
            table_masks = writer.create_table(
                name="masks",
                dtype=np.dtype([
                    (name, "<i1", (rows, cols)) for name in masks.keys()
                ]),
                title="Masks applied on image data",
                expectedrows=expectedrows,
            )

            entry_patient = table_patient.row
            entry_hsimage = table_hsimage.row
            entry_masks = table_masks.row

            print(f"Number of entries to export: {expectedrows}")
            for i in range(expectedrows):
                print("Export record %s (%d/%d) ..." % (
                    patient["timestamp"], i+1, expectedrows))

                if i > 0:  # first element already loaded before
                    patient, hsimage, masks = self.select(i)

                # append patient information
                for column_name in table.dtype.names:
                    entry_patient[column_name] = patient[column_name]
                entry_patient.append()

                # append hyperspectral image data
                for column_name in hsimage.keys():
                    entry_hsimage[column_name] = hsimage[column_name]
                entry_hsimage.append()

                # append masks
                for column_name in masks.keys():
                    entry_masks[column_name] = masks[column_name]
                entry_masks.append()

            table_patient.flush()
            table_hsimage.flush()
            table_masks.flush()

    @staticmethod
    def open(file_path, path="/"):
        """ Create a new HSTivitaStore object and leave the file open for
        further processing.

        Parameters
        ----------
        file_path : str or pathlib.Path
            The File, filepath, or generator to read.
        path : str, optional
            The path within the underlying hdf5 file. Default is the root path.
        """
        return HSTivitaStore(file_path, path=path)
