# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 16:05:22 2021

@author: kpapke
"""
import sys
import os.path
import logging

from timeit import default_timer as timer
import multiprocessing

from multiprocessing import Process, Pipe


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


# def task(patient, spectra, wavelen, masks):
def task(patient, hsidata):
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
    hsformat = HSFormatFlag.fromStr(patient["hsformat"].decode())

    print("%8d | %8d | %-20s | %-20s | %-10s | %3d |" % (
        patient["pn"],
        patient["pid"],
        patient["descr"].decode(),
        patient["timestamp"].decode(),
        hsformat.key,
        patient["target"]
    ))

    # hsImage = HSImage(spectra=spectra, wavelen=wavelen, format=hsformat)
    hsImage = HSImage(spectra=hsidata["hsidata"], wavelen=hsidata["wavelen"], format=hsformat)
    image = hsImage.getRGBValue()

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

    analysis = HSTivita(format=HSIntensity)
    analysis.setData(hsImage.spectra, hsImage.wavelen, format=hsformat)
    analysis.evaluate(mask=masks["tissue"])
    param = analysis.getSolution(unpack=True, clip=True)
    # param = None
    fileName = "PN_%03d_PID_%07d_Date_%s_Tivita.jpg" % (
        patient["pn"], patient["pid"], patient["timestamp"].decode())
    # plotParam(fileName, param)

    return param



def main():

    dirPaths = getDirPaths()

    start = timer()

    # open file in (r)ead mode
    fileName = "rostock_suedstadt_2018-2020_2.h5"
    h5file = tables.open_file(os.path.join(dirPaths['data'], fileName), mode="r")

    group = h5file.root.records
    table = h5file.get_node('/records/patient')
    table1 = h5file.get_node('/records/hsidata')
    # spectra = h5file.get_node('/records/hsidata/spectra')
    # wavelen = h5file.get_node('/records/hsidata/wavelen')
    # masks = h5file.get_node('/records/hsidata/masks')


    # print("\nSerial evaluation")
    # print("---------------------")
    # for i, (patient, hsidata) in enumerate(zip(table.iterrows(), table1.iterrows())):
    #     task(patient, hsidata)


    print("\nParallel evaluation")
    print("---------------------")

    nproc = 7
    nitems = table.nrows
    print("Items: ", nitems)
    with multiprocessing.Pool(processes=nproc) as pool:

        # apply_async with unlimited pool size
        # mrst = [
        #     pool.apply_async(task, (buf.fetch_all_fields(),))
        #     for buf in table.iterrows()]
        # [rst.get(timeout = 3) for rst in mrst]

        # apply_async with limited pool size
        for i in range(0, nitems, nproc):
            mrst = [
                pool.apply_async(task, (table[i + j], table1[i + j]))
                for j in range(nproc if i + nproc < nitems else nitems - i)
            ]
            # [rst.get(timeout=10) for rst in mrst]
            [rst.get() for rst in mrst]

    #     # starmap with unlimited pool size
    #     # rst = pool.starmap(task, [(buf.fetch_all_fields(),)
    #     #     for buf in table.iterrows()])
    #
    #     # for i in range(0, nitems, nproc):
    #     #     rst = pool.starmap(task, [
    #     #         (table[i + j], spectra[i + j], wavelen[i + j], masks[i + j])
    #     #         for j in range(nproc if i + nproc < nitems else nitems - i)
    #     #     ])
    #
    #     for i in range(0, nitems, nproc):
    #         rst = pool.starmap(task, [
    #             (table[i + j], table1[i + j])
    #             for j in range(nproc if i + nproc < nitems else nitems - i)
    #         ])


    # Finally, close the file (this also will flush all the remaining buffers!)
    h5file.close()

    print("\nElapsed time: %f sec" % (timer() - start))



if __name__ == '__main__':
    # logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()
