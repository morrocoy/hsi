# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:24:11 2021

@author: kpapke
"""
import sys
import os.path
import logging

from timeit import default_timer as timer
import time
import multiprocessing
import pathos.multiprocessing as mp

from multiprocessing import Process, Pipe


import pandas as pd
import numpy as np
import tables

from tables_utils import getDirPaths, plotMasks, plotParam

import hsi
from hsi import HSStore
from hsi import HSImage, HSIntensity, HSAbsorption, HSFormatFlag
from hsi.analysis import HSTivita
from hsi.log import logmanager

logger = logmanager.getLogger(__name__)


# def task(patient, hsidata):
#     """Example task applied on each entry.
#
#     Parameters
#     ----------
#     patient : pd.Series
#         Metadata of the record.
#     spectra :  numpy.ndarray
#         The spectral data.
#     wavelen :  numpy.ndarray
#         The wavelengths at which the spectral data are sampled.
#     masks :  numpy.ndarray
#         Masks to be applied on the hyperspectral image.
#
#     Returns
#     -------
#     numpy.ndarray : Array of values for validation.
#
#     """
#     hsformat = HSFormatFlag.from_str(patient["hsformat"].decode())
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
#     # hsImage = HSImage(spectra=spectra, wavelen=wavelen, hsformat=hsformat)
#     hsImage = HSImage(spectra=hsidata["hsidata"], wavelen=hsidata["wavelen"], hsformat=hsformat)
#     image = hsImage.as_rgb()
#
#     keys = [
#         "tissue",
#         "critical wound region",
#         "wound region",
#         "wound and proximity",
#         "wound proximity"
#     ]
#     # masks = {key: val for (key, val) in zip(keys, masks)}
#     masks = {key: val for (key, val) in zip(keys, hsidata["masks"])}
#
#     fileName = "PN_%03d_PID_%07d_Date_%s_Masks.jpg" % (
#         patient["pn"], patient["pid"], patient["timestamp"].decode())
#     plotMasks(fileName, masks, image)
#
#     analysis = HSTivita(hsformat=HSIntensity)
#     analysis.set_data(hsImage.spectra, hsImage.wavelen, hsformat=hsformat)
#     analysis.evaluate(mask=masks["tissue"])
#     param = analysis.get_solution(unpack=True, clip=True)
#     # param = None
#     fileName = "PN_%03d_PID_%07d_Date_%s_Tivita.jpg" % (
#         patient["pn"], patient["pid"], patient["timestamp"].decode())
#     plotParam(fileName, param)
#
#     return param


def task(args):
    """Example task applied on each entry.

    Parameters
    ----------
    patient : pd.Series
        Metadata of the record.
    spectra :  numpy.ndarray
        The spectral data.
    wavelen :  numpy.ndarray
        The wavelengths at which the spectral data are sampled.
    masks :  numpy.ndarray
        Masks to be applied on the hyperspectral image.

    Returns
    -------
    numpy.ndarray : Array of values for validation.

    """
    patient = args[0]
    hsidata = args[1]

    hsformat = HSFormatFlag.from_str(patient["hsformat"].decode())

    print("%8d | %8d | %-20s | %-20s | %-10s | %3d |" % (
        patient["pn"],
        patient["pid"],
        patient["descr"].decode(),
        patient["timestamp"].decode(),
        hsformat.key,
        patient["target"]
    ))

    # hsImage = HSImage(spectra=spectra, wavelen=wavelen, hsformat=hsformat)
    hsImage = HSImage(spectra=hsidata["hsidata"], wavelen=hsidata["wavelen"], hsformat=hsformat)
    image = hsImage.as_rgb()

    keys = [
        "tissue",
        "critical wound region",
        "wound region",
        "wound and proximity",
        "wound proximity"
    ]
    # masks = {key: val for (key, val) in zip(keys, masks)}
    masks = {key: val for (key, val) in zip(keys, hsidata["masks"])}

    fileName = "PN_%03d_PID_%07d_Date_%s_Masks.jpg" % (
        patient["pn"], patient["pid"], patient["timestamp"].decode())
    # plotMasks(fileName, image, masks)

    analysis = HSTivita(hsformat=HSIntensity)
    analysis.set_data(hsImage.spectra, hsImage.wavelen, hsformat=hsformat)
    analysis.evaluate(mask=masks["tissue"])
    param = analysis.get_solution(unpack=True, clip=True)
    # param = None
    fileName = "PN_%03d_PID_%07d_Date_%s_Tivita.jpg" % (
        patient["pn"], patient["pid"], patient["timestamp"].decode())
    # plotParam(fileName, param)

    return param



def main():

    dirPaths = getDirPaths()

    start = timer()

    fileName = "rostock_suedstadt_2018-2020_3.h5"
    filePath = os.path.join(dirPaths['data'], fileName)
    with HSStore.open(filePath, mode="r") as dataset:

        # serial evaluation
        # for entry in iter(dataset):
        #     task(entry)

        # parallel evaluation
        pool = multiprocessing.Pool(processes=7)
        for rst in pool.imap(task, iter(dataset)):#, chunksize=1):
            pass

        # for rst in pool.starmap(task, list(dataset)):
        #     pass

        pool.close()

    print("\nElapsed time: %f sec" % (timer() - start))



if __name__ == '__main__':
    # logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()
