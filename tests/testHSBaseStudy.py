# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 12:38:38 2021

@author: kpapke
"""
import sys
import os.path
from timeit import default_timer as timer
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

import hsi
from hsi import HSDataset, HSImage
from hsi import cm, HSFormatFlag, HSIntensity
from hsi.analysis import HSBaseStudy

import logging

# LOGGING = True
LOGGING = False
logger = logging.getLogger(__name__)
logger.propagate = LOGGING



def main():
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    # data_path = os.path.join(os.getcwd(), "..", "data")
    data_path = os.path.join(os.getcwd(), "..", "..", "..", "amputation", "data")
    # data_path = os.path.join("c:", os.sep, "temp")

    # input file
    # filename = "test_dataset.h5"
    filename = "rostock_suedstadt_2018-2020.h5"
    # filename = "rostock_suedstadt_2018-2020_zip.h5"
    filepath = os.path.join(data_path, filename)



    # load dataset
    with HSDataset(filepath) as dataset:
        print(dataset.descr)
        print("\nFound %d of %d groups in File" % (
            len(dataset.groups), len(dataset)))

        study = HSBaseStudy("tivita_1") #, dataset)
        study.run(dataset)

    # data_path = os.path.join(data_path, "rostock_suedstadt_2018-2020", "tivita_1", "data")
    # filepath = os.path.join(data_path, "tivita_1.h5")
    # with HSDataset(filepath) as dataset:
    #     print(dataset.descr)
    #     print("\nFound %d of %d groups in File" % (
    #         len(dataset.groups), len(dataset)))
    #
    #     for i in range(7):
    #         spectra, wavelen, masks, metadata = dataset[i]
    #         print("pn {:10} | pid {:10} | hash {}".format(
    #             metadata['pn'], metadata['pid'], metadata['hash']))


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
