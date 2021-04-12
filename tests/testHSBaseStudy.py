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
import matplotlib.pyplot as plt

import hsi
from hsi import HSStore, HSImage
from hsi import cm, HSFormatFlag, HSIntensity
from hsi.analysis import HSBaseStudy
from hsi.log import logmanager

logger = logmanager.getLogger(__name__)


def main():
    # data_path = os.path.join(os.getcwd(), "..", "data")
    data_path = os.path.join(os.getcwd(), "..", "..", "..", "amputation", "data")
    # data_path = os.path.join("c:", os.sep, "temp")

    # input file
    # filename = "test_dataset.h5"
    filename = "rostock_suedstadt_2018-2020.h5"
    # filename = "rostock_suedstadt_2018-2020_zip.h5"
    filepath = os.path.join(data_path, filename)



    # load dataset
    with HSStore(filepath) as dataset:
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
    logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()
