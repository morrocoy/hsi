# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:16:59 2021

@author: kpapke
"""
import sys
import os.path
import logging

from timeit import default_timer as timer
import multiprocessing

import pandas as pd
import numpy as np
import tables

from tables_utils import getDirPaths, plotMasks, plotParam

import hsi
from hsi import HSImage, HSIntensity, HSAbsorption, HSFormatFlag
from hsi import genHash
from hsi.analysis import HSTivita
from hsi.log import logmanager

logger = logmanager.getLogger(__name__)


def task(patient):
    hsformat = HSFormatFlag.fromStr(patient["hsformat"].decode())

    print("%8d | %8d | %-20s | %-20s | %-10s | %3d |" % (
        patient["pn"],
        patient["pid"],
        patient["descr"].decode(),
        patient["timestamp"].decode(),
        hsformat.key,
        patient["target"]
    ))

    hsImage = HSImage(
        spectra=patient["hsidata"], wavelen=patient["wavelen"], format=hsformat)
    image = hsImage.getRGBValue()

    keys = [
        "tissue",
        "critical wound region",
        "wound region",
        "wound and proximity",
        "wound proximity"
    ]
    mask = {key: val for (key, val) in zip(keys, patient["mask"])}

    fileName = "PN_%03d_PID_%07d_Date_%s_Masks.jpg" % (
        patient["pn"], patient["pid"], patient["timestamp"])
    # plotMasks(fileName, image, mask)

    analysis = HSTivita(format=HSIntensity)
    analysis.setData(hsImage.spectra, hsImage.wavelen, format=hsformat)
    analysis.evaluate(mask=mask["tissue"])
    param = analysis.getSolution(unpack=True, clip=True)
    # param = None
    fileName = "PN_%03d_PID_%07d_Date_%s_Tivita.jpg" % (
        patient["pn"], patient["pid"], patient["timestamp"])
    # plotParam(fileName, param)

    return param


def main():

    dirPaths = getDirPaths()

    start = timer()

    # open file in (r)ead mode
    fileName = "rostock_suedstadt_2018-2020_1.h5"
    h5file = tables.open_file(os.path.join(dirPaths['data'], fileName), mode="r")

    group = h5file.root.records
    table = h5file.root.records.patient

    # print(group._v_attrs.descr)
    # print(repr(table))

    # print("\nSerial evaluation")
    # print("---------------------")
    # for patient in table.iterrows():
    #     task(patient)

    print("\nParallel evaluation")
    print("---------------------")

    nproc = 7
    nitems = table.nrows
    print("Items: ", nitems)
    with multiprocessing.Pool(processes=nproc) as pool:

        i = 0
        j = 0
        buffer = []
        while i < nitems:
            while i < nitems and j < nproc:
                patient = table[i]

                buffer.append(patient)
                i += 1
                j += 1

            rst = pool.starmap(task, buffer)
            buffer.clear()
            j = 0
            # buffer.clear()


    # Finally, close the file (this also will flush all the remaining buffers!)
    h5file.close()

    print("\nElapsed time: %f sec" % (timer() - start))



if __name__ == '__main__':
    # logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()
