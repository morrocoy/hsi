# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:46:45 2021

@author: kpapke
"""
import sys
import os.path
from timeit import default_timer as timer
import multiprocessing

import numpy as np

import hsi
from hsi import HSDataset
from hsi import HSIntensity, HSAbsorption, HSRefraction
import logging

# LOGGING = True
LOGGING = False
logger = logging.getLogger(__name__)
logger.propagate = LOGGING



def calc(*args):
    metadata = args[0][0]
    spectra = args[0][1]
    wavelen = args[0][2]
    masks = args[0][3]
    print("pid {} | shape {} | mean {}".format(
            metadata['pid'], spectra.shape, np.mean(spectra)))


def main():
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    data_path = os.path.join(os.getcwd(), "..", "data")

    filePath = os.path.join(data_path, "test_dataset2.h5")
    with HSDataset(filePath) as dataset:

        print(dataset.descr)

        print("\nFound %d groups in File" % len(dataset.groups))
        print("---------------------------------")
        for i, group in enumerate(dataset.groups):
            print(group)


        print("\nLoad spectral data")
        print("---------------------------------")
        start = timer()

        n = len(dataset.groups)
        data = np.zeros((n, 3))

        for i in range(65):
            metadata, spectra, wavelen, masks = dataset[i]
            print("pid {} | spectra {} | masks {}".format(
                metadata['pid'], np.mean(spectra), np.mean(masks)))
            data[i, 0] = metadata['pid']
            data[i, 1] = np.mean(spectra)
            data[i, 2] = np.mean(masks)

        print("\nElapsed time: %f sec" % (timer() - start))

    # refdata = np.zeros((n, 3))
    refdata = np.load(os.path.join(data_path, "test_dataset.npy"))

    print("Residual: {}".format(np.max(np.abs(data-refdata), axis=0)))


    print(data[:, 1]-refdata[:, 1])
        # for metadata, spectra, wavelen, masks in dataset.items():
        #     print("pid {} | shape {} | mean {}".format(
        #         metadata['pid'], spectra.shape, np.mean(spectra)))
        # print("\nElapsed time: %f sec" % (timer() - start))

        # print("\nLoad spectral data (parallel)")
        # print("---------------------------------")
        # start = timer()
        # with multiprocessing.Pool(processes=8) as pool:
        #     # pool.map(calc, list(dataset.items()))
        #     pool.map(calc, dataset.items())
        # print("\nElapsed time: %f sec" % (timer() - start))




if __name__ == '__main__':
    # fmt = "%(asctime)s %(filename)35s: %(lineno)-4d: %(funcName)20s(): " \
    #       "%(levelname)-7s: %(message)s"
    # logging.basicConfig(level='DEBUG', format=fmt)

    # requests_logger = logging.getLogger('hsi')
    requests_logger = logging.getLogger(__file__)
    requests_logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
            "%(asctime)s %(filename)35s: %(lineno)-4d: %(funcName)20s(): " \
              "%(levelname)-7s: %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    requests_logger.addHandler(handler)

    main()




