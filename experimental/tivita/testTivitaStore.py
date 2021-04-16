# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:03:42 2021

@author: kai
"""
import sys
import os.path
import logging
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from tables_utils import getDirPaths, loadPatientData, loadHSData
import hsi
from hsi import HSFormatFlag, HSIntensity, HSImage, HSTivitaStore
from hsi.log import logmanager


# windows system
if sys.platform == "win32":
    _rootPath = os.path.join(
        "d:", os.sep, "projects", "hyperlimit", "amputation",
        "studies", "rostock_suedstadt_2018-2020")

# linux system
else:
    _rootPath = os.path.join(
        os.path.expanduser("~"), "projects", "hyperlimit", "amputation",
        "studies", "rostock_suedstadt_2018-2020")


def getDirPaths():
    return {
        'root': _rootPath,
        'data': os.path.join(_rootPath, "data"),
        'pictures': os.path.join(_rootPath, "pictures"),
        'results': os.path.join(_rootPath, "results"),
    }



def plotMasks(fileName, masks, image=None):
    """Create masked plots of the original rgb image.

    Parameters
    ----------
    filepath : str,
        The full path to the output file.
    image : np.ndarray
        The rgb image.
    mask :  dict
        A dictionarry of the mask arrays.
    """
    fig = plt.figure()
    fig.set_size_inches(10, 8)

    d, m =  divmod(len(masks), 2)
    cols = d
    rows = d + m

    if isinstance(masks, dict):
        keys = masks.keys()
    elif isinstance(masks, np.void):
        keys = masks.dtype.names
    else:
        raise ValueError("Mask must be either dict or numpy.void")

    if image is None:
        for i, key in enumerate(keys):
            ax = fig.add_subplot(cols, rows, i + 1)
            plt.imshow(masks[key], cmap="gray", vmin=0, vmax=1)
            ax.set_title(key)

    else:
        for i, key in enumerate(keys):
            mimage = image.copy()
            red = mimage[:, :, 0]
            green = mimage[:, :, 1]
            blue = mimage[:, :, 2]

            idx = np.nonzero(masks[key] == 0)  # gray out region out of mask
            gray = 0.2989 * red[idx] + 0.5870 * green[idx] + 0.1140 * blue[idx]
            red[idx] = gray
            green[idx] = gray
            blue[idx] = gray

            ax = fig.add_subplot(cols, rows, i+1)
            plt.imshow(mimage)
            ax.set_title(key)

    ext = fileName[fileName.rfind('.')+1:]
    filepath = os.path.join(_rootPath, "pictures", fileName)

    fig.savefig(filepath, format=ext)
    # plt.show()
    plt.close()



logger = logmanager.getLogger(__name__)

dirPaths = getDirPaths()


def main():
    infile = "rostock_suedstadt_2018-2020.xlsx"
    outfile = "rostock_suedstadt_2018-2020.h5"

    # define columns for the patient info table
    HSPatientInfo = np.dtype([
        ("pn", '<i8'),
        ("pid", '<i8'),
        ("name", 'S32'),
        ("descr", 'S64'),
        ("timestamp", 'S32'),
        ("target", '<i4'),
        # ("path", 'S32'),
    ])

    filePath = os.path.join(dirPaths['data'], infile)
    start = timer()
    with HSTivitaStore.open(filePath, path="/") as store:

        # configure store
        store.skipNameColumn = True
        store.markerColor = [100, 255, 0]
        store.overwriteMasks = True

        # attach table with patien info from underlying excel file
        store.attacheTable(
            name="patient",
            dtype=HSPatientInfo,
            sheet_name=0,
            usecols=[3, 4, 5, 6, 8, 7],
            skiprows=1,
        )

        filePath = os.path.join(dirPaths['data'], outfile)


        store.to_hdf(filePath, "/records", "Hallo Welt")

    print("\nElapsed time for processing dataset: %f sec" % (timer() - start))

        # print(f"Number of entries: {len(store)}")
        # for patient, hsimage, masks in iter(store):
        #     # patient, hsimage, masks = reader[0]
        #
        #     hsformat = HSFormatFlag.fromStr(hsimage["hsformat"].decode())
        #
        #     print("%8d | %8d | %-20s | %-20s | %-10s | %3d |" % (
        #         patient["pn"],
        #         patient["pid"],
        #         patient["descr"].decode(),
        #         patient["timestamp"].decode(),
        #         hsformat.key,
        #         patient["target"]
        #     ))
        #
        #     hsImage = HSImage(spectra=hsimage["spectra"],
        #                       wavelen=hsimage["wavelen"], format=hsformat)
        #     image = hsImage.getRGBValue()
        #
        #     fileName = "PN_%03d_PID_%07d_Date_%s_Masks.jpg" % (
        #         patient["pn"], patient["pid"], patient["timestamp"].decode())
        #     # plotMasks(fileName, masks, image)




if __name__ == '__main__':
    if __name__ == '__main__':
        # logmanager.setLevel(logging.DEBUG)
        # logger.info("Python executable: {}".format(sys.executable))
        # logger.info("Python hsi version: {}".format(hsi.__version__))
        main()
