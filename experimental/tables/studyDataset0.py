# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 12:38:38 2021

@author: kpapke
"""
import sys
import os.path
import logging

from timeit import default_timer as timer
import multiprocessing

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

import hsi
from hsi import HSStore, genHash, HSImage

from tables_utils import getDirPaths, plotMasks, plotParam

from hsi import HSImage, HSIntensity, HSAbsorption, HSFormatFlag
from hsi import genHash
from hsi.analysis import HSTivita
from hsi.log import logmanager

logger = logmanager.getLogger(__name__)


def task(patient, hsidata, wavelen, mask):
    """Example task applied on each entry.

    Parameters
    ----------
    patient : pd.Series
        Metadata of the record.
    hsidata :  numpy.ndarray
        The spectral data.
    wavelen :  numpy.ndarray
        The wavelengths at which the spectral data are sampled.
    mask :  numpy.ndarray
        Masks to be applied on the hyperspectral image.

    Returns
    -------
    numpy.ndarray : Array of values for validation.

    """
    hsformat = HSFormatFlag.from_str(patient["hsformat"])

    print("%8d | %8d | %-20s | %-20s | %-10s | %3d |" % (
        patient['pn'],
        patient['pid'],
        patient['descr'],
        patient['timestamp'],
        hsformat.key,
        patient['target']
    ))

    # image visualization .....................................................
    hsImage = HSImage(hsidata, wavelen, hsformat)
    image = hsImage.as_rgb()

    keys = [
        "tissue",
        "critical wound region",
        "wound region",
        "wound and proximity",
        "wound proximity"
    ]
    mask = {key: val for (key, val) in zip(keys, mask)}

    fileName = "PN_%03d_PID_%07d_Date_%s_Masks.jpg" % (
        patient["pn"], patient["pid"], patient["timestamp"])
    plotMasks(fileName, mask, image)

    # analysis ................................................................
    analysis = HSTivita(hsformat=HSIntensity)
    analysis.set_data(hsImage.spectra, hsImage.wavelen, hsformat=hsformat)
    analysis.evaluate(mask=mask["tissue"])
    param = analysis.get_solution(unpack=True, clip=True)
    # param = None
    fileName = "PN_%03d_PID_%07d_Date_%s_Tivita.jpg" % (
        patient["pn"], patient["pid"], patient["timestamp"])
    plotParam(fileName, param)

    return param



def main():
    dirPaths = getDirPaths()

    start = timer()

    # open file in (r)ead mode
    fileName = "rostock_suedstadt_2018-2020_0.h5"
    filePath = os.path.join(dirPaths['data'], fileName)


    # retrieve pd.dataframe of metadata from file
    with pd.HDFStore(filePath, 'r') as store:
        if '/metadata' in store.keys():
            dfMetadata = store['metadata']

    with h5py.File(filePath, 'r') as h5file:


        # print("\nSerial evaluation")
        # print("---------------------")
        #
        # for index, patient in dfMetadata.iterrows():
        #     group = h5file[patient["timestamp"]]
        #     keys = group.keys()
        #
        #     hsidata = group["hsidata"][()] if "hsidata" in keys else None
        #     wavelen = group["wavelen"][()] if "wavelen" in keys else None
        #     mask = group["mask"][()] if "mask" in keys else None
        #
        #     res = task(patient, hsidata, wavelen, mask)


        print("\nParallel evaluation")
        print("---------------------")

        nproc = 7
        nitems = len(dfMetadata.index)
        with multiprocessing.Pool(processes=nproc) as pool:

            i = 0
            j = 0
            buffer = []
            while i < nitems:
                while i < nitems and j < nproc:
                    patient = dfMetadata.iloc[i]
                    group = h5file[patient["timestamp"]]
                    keys = group.keys()

                    hsidata = group["hsidata"][()] if "hsidata" in keys else None
                    wavelen = group["wavelen"][()] if "wavelen" in keys else None
                    mask = group["mask"][()] if "mask" in keys else None

                    buffer.append((patient, hsidata, wavelen, mask))
                    i += 1
                    j += 1

                rst = pool.starmap(task, buffer)
                buffer.clear()
                j = 0
                # buffer.clear()

        print("\nElapsed time: %f sec" % (timer() - start))




if __name__ == '__main__':
    logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()





