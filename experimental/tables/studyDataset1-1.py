# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 07:40:56 2021

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


def task_split(patient, hsidata, wavelen, mask):
    hsformat = HSFormatFlag.from_str(patient["hsformat"].decode())

    print("%8d | %8d | %-20s | %-20s | %-10s | %3d |" % (
        patient["pn"],
        patient["pid"],
        patient["descr"].decode(),
        patient["timestamp"].decode(),
        hsformat.key,
        patient["target"]
    ))

    hsImage = HSImage(
        spectra=hsidata, wavelen=wavelen, hsformat=hsformat)
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
    # plotMasks(fileName, mask, image)

    analysis = HSTivita(hsformat=HSIntensity)
    analysis.set_data(hsImage.spectra, hsImage.wavelen, hsformat=hsformat)
    analysis.evaluate(mask=mask["tissue"])
    param = analysis.get_solution(unpack=True, clip=True)
    # param = None
    fileName = "PN_%03d_PID_%07d_Date_%s_Tivita.jpg" % (
        patient["pn"], patient["pid"], patient["timestamp"])
    # plotParam(fileName, param)

    return param



def task(patient):
    hsformat = HSFormatFlag.from_str(patient["hsformat"].decode())

    print("%8d | %8d | %-20s | %-20s | %-10s | %3d |" % (
        patient["pn"],
        patient["pid"],
        patient["descr"].decode(),
        patient["timestamp"].decode(),
        hsformat.key,
        patient["target"]
    ))

    hsImage = HSImage(
        spectra=patient["hsidata"], wavelen=patient["wavelen"], hsformat=hsformat)
    image = hsImage.as_rgb()

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

    analysis = HSTivita(hsformat=HSIntensity)
    analysis.set_data(hsImage.spectra, hsImage.wavelen, hsformat=hsformat)
    analysis.evaluate(mask=mask["tissue"])
    param = analysis.get_solution(unpack=True, clip=True)
    # param = None
    fileName = "PN_%03d_PID_%07d_Date_%s_Tivita.jpg" % (
        patient["pn"], patient["pid"], patient["timestamp"])
    # plotParam(fileName, param)

    return param


def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


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

    parmap(task, table.iterrows())

    # with multiprocessing.Pool(processes=nproc) as pool:

    #     # apply_async with unlimited pool size
    #     # mrst = [
    #     #     pool.apply_async(task, (buf.fetch_all_fields(),))
    #     #     for buf in table.iterrows()]
    #     # [rst.get(timeout = 3) for rst in mrst]
    #
    #     # apply_async with limited pool size
    #     for i in range(0, nitems, nproc):
    #         mrst = [
    #             pool.apply_async(task, (table[i + j],))
    #             for j in range(nproc if i + nproc < nitems else nitems - i)
    #         ]
    #         # [rst.get(timeout=10) for rst in mrst]
    #         [rst.get() for rst in mrst]

        # starmap with unlimited pool size
        # rst = pool.starmap(task, [(buf.fetch_all_fields(),)
        #     for buf in table.iterrows()])

        # for i in range(0, nitems, nproc):
        #     rst = pool.starmap(task, [
        #         (table[i + j],)
        #         for j in range(nproc if i + nproc < nitems else nitems - i)
        #     ])

        # i = 0
        # j = 0
        # buffer = []
        # while i < nitems:
        #     while i < nitems and j < nproc:
        #         patient = table[i]
        #
        #         metadata = {
        #             key: patient[key] for key in
        #             ["pn", "pid", "descr", "timestamp", "hsformat", "target"]
        #         }
        #         hsidata =  patient["hsidata"]
        #         wavelen = patient["wavelen"]
        #         mask = patient["mask"]
        #         buffer.append((patient, hsidata, wavelen, mask))
        #
        #         # buffer.append((patient,))
        #         i += 1
        #         j += 1
        #
        #     rst = pool.starmap(task_split, buffer)
        #     buffer.clear()
        #     j = 0



    # Finally, close the file (this also will flush all the remaining buffers!)
    h5file.close()

    print("\nElapsed time: %f sec" % (timer() - start))



if __name__ == '__main__':
    # logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()
