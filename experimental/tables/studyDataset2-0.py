# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 16:05:22 2021

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
    # plotMasks(fileName, masks, image)

    analysis = HSTivita(hsformat=HSIntensity)
    analysis.set_data(hsImage.spectra, hsImage.wavelen, hsformat=hsformat)
    analysis.evaluate(mask=masks["tissue"])
    param = analysis.get_solution(unpack=True, clip=True)
    # param = None
    fileName = "PN_%03d_PID_%07d_Date_%s_Tivita.jpg" % (
        patient["pn"], patient["pid"], patient["timestamp"].decode())
    # plotParam(fileName, param)

    return param

def task1(args):
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





class tableIterator:
    """Class to to iterate through the patient and hsidata tables"""

    def __init__(self, patient, hsidata):
        self.patient = patient
        self.hsidata = hsidata
        self.max = self.patient.nrows

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < self.max:
            result = (self.patient[self._index], self.hsidata[self._index])
            self._index  += 1
            return result
        raise StopIteration  # end of Iteration




def main():

    dirPaths = getDirPaths()

    start = timer()

    # open file in (r)ead mode
    fileName = "rostock_suedstadt_2018-2020_2_test.h5"
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

    # p = multiprocessing.Pool()
    # start = time.time()
    # for x in p.imap(func, range(3)):
    #     print("{} (Time elapsed: {}s)".hsformat(x, int(time.time() - start)))
    # for x in p.imap(func2, [(1, 4), (2, 5), (3, 6)]):
    #     print("{} (Time elapsed: {}s)".hsformat(x, int(time.time() - start)))

    # for x in p.imap(func2, zip(range(3), range(3, 6))):
    #     print("{} (Time elapsed: {}s)".hsformat(x, int(time.time() - start)))

    # pool = mp.ProcessingPool(nodes=7)
    # for i in pool.imap(func3, [table[i] for i in range(nitems)]):
    #     print(i)
    # results = pool.imap(func3, [table[i] for i in range(nitems)])
    # results = pool.imap(func3, table.iterrows())
    # results = pool.imap(func4, [(table[i], table1[i]) for i in range(nitems)])

    # for rst in pool.imap(task1, [(table[i], table1[i]) for i in range(nitems)]):
    #     pass

    # print("...")
    # results = list(results)

    # pool.map(task, [(table[i], table1[i]) for i in range(nitems)])

    table_iter = tableIterator(table, table1)
    with multiprocessing.Pool(processes=nproc) as pool:


        # apply_async with unlimited pool size
        # mrst = [
        #     pool.apply_async(task, (buf.fetch_all_fields(),))
        #     for buf in table.iterrows()]
        # [rst.get(timeout = 3) for rst in mrst]

        # apply_async with limited pool size
        # for i in range(0, nitems, nproc):
        #     mrst = [
        #         pool.apply_async(task, (table[i + j], table1[i + j]))
        #         for j in range(nproc if i + nproc < nitems else nitems - i)
        #     ]
        #     # [rst.get(timeout=10) for rst in mrst]
        #     [rst.get() for rst in mrst]

        # rst = pool.starmap(task, zip(table.iterrows(), table1.iterrows()))
        # rst = pool.imap(task, zip(table.iterrows(), table1.iterrows()))
        # rst = pool.imap(task, [1,2])

        # starmap with unlimited pool size
        # rst = pool.starmap(task, [(buf.fetch_all_fields(),)
        #     for buf in table.iterrows()])

        # for i in range(0, nitems, nproc):
        #     rst = pool.starmap(task, [
        #         (table[i + j], spectra[i + j], wavelen[i + j], masks[i + j])
        #         for j in range(nproc if i + nproc < nitems else nitems - i)
        #     ])

        # rst = pool.starmap(task, [(table[i], table1[i]) for i in range(nitems)])
        # for rst in pool.imap_unordered(task1, [(table[i], table1[i]) for i in range(nitems)], chunksize=1):
        #     pass

        for rst in pool.imap(task1, table_iter, chunksize=1):
        # for rst in pool.imap_unordered(task1, tableIter2(table, table1), chunksize=1):
            pass

        # # print("...")
        # results = list(rst)

        # for i in range(0, nitems, nproc):
        #     rst = pool.starmap(task, [
        #         (table[i + j], table1[i + j])
        #         for j in range(nproc if i + nproc < nitems else nitems - i)
        #     ])


    # Finally, close the file (this also will flush all the remaining buffers!)
    h5file.close()

    print("\nElapsed time: %f sec" % (timer() - start))



if __name__ == '__main__':
    # logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()
