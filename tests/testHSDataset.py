# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:46:45 2021

@author: kpapke
"""
import sys
import os.path
import logging
from timeit import default_timer as timer
import hashlib

import numpy as np

import hsi
from hsi import HSStore, genHash
from hsi import HSIntensity, HSAbsorption, HSRefraction
from hsi.log import logmanager

logger = logmanager.getLogger(__name__)


def task( spectra, wavelen, masks, metadata):
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
    # print("pid {:10} | spectra {:<10.6f} | mask {:<10.6f}".format(
    #     metadata['pid'], np.mean(spectra), np.mean(masks["tissue"])))

    # checksum
    hash = genHash(spectra)
    state = "valid" if hash == metadata['hash'] else "invalid"

    print("pid {:10} | spectra {:<10.6f} | mask {:<10.6f} | hash {} {}".format(
        metadata['pid'], np.mean(spectra), np.mean(masks["tissue"]), hash, state))

    # return values for validation
    res = np.array(
        [metadata['pid'], np.mean(spectra), np.mean(masks["tissue"])], dtype='>f4')
    return res


def main():
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

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

        print("\nLoad spectral data")
        print("---------------------------------")
        start = timer()

        n = len(dataset.groups)
        checksum = np.zeros((n, 3), dtype='>f4')

        for i in range(n):
            spectra, wavelen, masks, metadata = dataset[i]
            checksum[i, :] = task(spectra, wavelen, masks, metadata)

        # for i, (metadata, spectra, wavelen, masks) in dataset.items():
        #     checksum[i, :] = task(metadata, spectra, wavelen, masks)

        print("\nElapsed time: %f sec" % (timer() - start))


    refchecksum = np.load(os.path.join(data_path, fileName + "_cs.npy"))
    print("Residual: {}".format(np.max(np.abs(checksum-refchecksum), axis=0)))
    # print(data[:, 1]-refdata[:, 1])



if __name__ == '__main__':
    logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()
