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

import hsi
from hsi import HSStore, genHash
from hsi import HSIntensity, HSAbsorption, HSRefraction
from hsi.log import logmanager

logger = logmanager.getLogger(__name__)


def task(metadata, spectra, wavelen, masks):
    """Example task applied on each entry.

    Parameters
    ----------
    metadata : pd.Series
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
    # print("pid {:10} | spectra {:<10.6f} | masks {:<10.6f}".format(
    #     metadata['pid'], np.mean(spectra), np.mean(masks)))

    # checksum
    hash = genHash(spectra)
    state = "valid" if hash == metadata['hash'] else "invalid"

    print("pid {:10} | spectra {:<10.6f} | mask {:<10.6f} | hash {} {}".format(
        metadata['pn'], np.mean(spectra), np.mean(masks["tissue"]), hash, state))

    msg = "pid {:10} | spectra {:<10.6f} | mask {:<10.6f} | hash {} {}".format(
        metadata['pn'], np.mean(spectra), np.mean(masks["tissue"]), hash, state)

    # return values for validation
    # res = metadata['pn'], np.mean(spectra), np.mean(masks)
    return msg



def main():
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    lock = multiprocessing.Lock()

    # data_path = os.path.join(os.getcwd(), "..", "data")
    data_path = os.path.join(os.getcwd(), "..", "..", "..", "amputation", "data")
    # data_path = os.path.join("c:", os.sep, "temp")

    # fileName = "test_dataset.h5"
    fileName = "rostock_suedstadt_2018-2020"
    # fileName = "rostock_suedstadt_2018-2020_zip"
    filePath = os.path.join(data_path, fileName+".h5")

    with HSStore(filePath) as dataset:
        print(dataset.descr)
        print("\nFound %d groups in File" % len(dataset.groups))
        print("---------------------------------")
        for i, group in enumerate(dataset.groups):
            print(group)

        print("\nLoad spectral data (parallel)")
        print("---------------------------------")
        start = timer()

        # with multiprocessing.Pool(processes=4) as pool:
        #     rst = pool.starmap(task, dataset.items())  # very slow using yield
        #     # pool.starmap(task, list(dataset.items()))  # list preferred solution
        #
        #     for msg in rst:
        #         print(msg)

        nproc = 4
        nitems = len(dataset)
        with multiprocessing.Pool(processes=nproc) as pool:
            # res = pool.apply_async(dataset.select, range(10))

            i = 0
            j = 0
            buffer = []
            while i < nitems:
                while i < nitems and j < nproc:
                    buffer.append(dataset[i])
                    i += 1
                    j += 1

                # rst = pool.starmap(task, buffer)
                # for msg in rst:
                #     print(msg)

                mrst = [pool.apply_async(task, buf) for buf in buffer]
                [rst.get(timeout=10) for rst in mrst]
                buffer.clear()
                j = 0


        # nproc = 4
        # nitems = len(dataset)
        #
        # i = 0
        # j = 0
        # plist = []
        # while i < nitems:
        #     while i < nitems and j < nproc:
        #         p = multiprocessing.Process(target=task, args=dataset[i])
        #         p.start()
        #         plist.append(p)
        #         i += 1
        #         j += 1
        #
        #     for p in plist:
        #         p.join()
        #
        #     plist.clear()
        #     j = 0



        print("\nElapsed time: %f sec" % (timer() - start))

        # # refdata = np.zeros((n, 3))
    # refchecksum = np.load(os.path.join(data_path, fileName + "_cs.npy"))
    # print("Residual: {}".format(np.max(np.abs(checksum-refchecksum), axis=0)))
    # # print(data[:, 1]-refdata[:, 1])



if __name__ == '__main__':
    logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()
