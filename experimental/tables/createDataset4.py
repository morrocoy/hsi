# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:29:38 2021

@author: kpapke

.. _rostock_suedstadt_2018-2020_dataset:

Amputation data from the hospital of Rostock Suedstadt between 2018 and 2020
----------------------------------------------------------------------------

**Data Set Characteristics:**

    :Number of Instances: 65 (44 of healing and 21 of not healing)
    :Number of Attributes: 7 meta, 3 numeric, predictive and 2 class attributes
    :Attribute Information:
        - metadata:
            - pn: Patient number
            - pid: Patient ID
            - descr: Description of the wound
            - timestamp: Timestamp as string
            - format: Spectral format
            - hash: checksum of the spectral data using md5 hash
        - numeric predictive:
            - hsidata: Hyperspectral data
            - wavelen: Wavelengths at which the spectral information is sampled
            - masks: Selection masks applied on the hyperspectral images
        - class (target):
            - not healed with target index 0
            - healed with target index 1
"""
import sys
import os.path
import logging
from timeit import default_timer as timer

import numpy as np

from tables_utils import getDirPaths, loadPatientData, loadHSData
import hsi
from hsi import HSIntensity, HSDataset, HSPatientInfo
from hsi.log import logmanager


logger = logmanager.getLogger(__name__)

dirPaths = getDirPaths()

def main():
    # load metadata
    fileName = "181022_Resektionsgrenze_Südstadt_Auswertung_27.10.2020.xlsx"
    filePath = os.path.join(dirPaths['results'], fileName)
    project = "rostock"
    hsformat = HSIntensity

    patientData = loadPatientData(filePath, sheet_name=0, skiprows=1)
    patientData['hsformat'] = hsformat.key

    # create output file
    start = timer()

    fileName = "rostock_suedstadt_2018-2020_4.h5"
    filePath = os.path.join(dirPaths['data'], fileName)

    print(filePath)
    with HSDataset.open(
            filePath, mode="w", path="/records", descr=__doc__) as dataset:
        rows = len(patientData.index)

        tablePatient = dataset.createTable(
            name="patient",
            dtype=HSPatientInfo,
            title="Patient information",
            expectedrows=rows,
        )

        tableHSImage = dataset.createTable(
            name="hsimage",
            dtype=np.dtype([
                ("hsformat", "<S32"),
                ("wavelen", "<f8", (100, )),
                ("spectra", "<f4", (100, 480, 640))
            ]),
            title="Hyperspectral image data",
            expectedrows=rows,
        )

        tableMasks = dataset.createTable(
            name="masks",
            dtype=np.dtype([
                ("tissue", "<i1", (480, 640)),
                ("critical", "<i1", (480, 640)),
                ("wound", "<i1", (480, 640)),
                ("proximity", "<i1", (480, 640)),
            ]),
            title="Masks applied on image data",
            expectedrows=rows,

        )

        entryPatient = tablePatient.row
        entryHSImage = tableHSImage.row
        entryMasks = tableMasks.row
        for index, info in patientData.iterrows():
            # if index > 2:
            #     break

            print("Process index %d ..." % info["pn"])

            entryPatient["pn"] = info["pn"]  # i f'Particle: {i:6d}'
            entryPatient["pid"] = info["pid"]
            entryPatient["name"] = str.encode("")
            entryPatient["descr"] = str.encode(info["descr"])
            entryPatient["timestamp"] = str.encode(info["timestamp"])
            # entryPatient["hsformat"] = str.encode(hsformat.key)
            entryPatient["target"] = info["target"]
            entryPatient.append()

            baseName = info["timestamp"].replace('-', '_')
            pathName = os.path.join(dirPaths['data'], baseName)
            hsidata, masks = loadHSData(pathName, baseName, hsformat)

            entryHSImage["hsformat"] = str.encode(hsformat.key)
            entryHSImage["wavelen"] = hsidata.wavelen.astype("<f8")
            entryHSImage["spectra"] = hsidata.spectra.astype("<f4")
            entryHSImage.append()

            entryMasks["tissue"] = masks[0].astype("<i1")
            entryMasks["critical"] = masks[1].astype("<i1")
            entryMasks["wound"] = masks[2].astype("<i1")
            entryMasks["proximity"] = masks[3].astype("<i1")
            entryMasks.append()

        tablePatient.flush()
        tableHSImage.flush()
        tableMasks.flush()

    print("\nElapsed time: %f sec" % (timer() - start))


if __name__ == '__main__':
    if __name__ == '__main__':
        logmanager.setLevel(logging.DEBUG)
        logger.info("Python executable: {}".format(sys.executable))
        logger.info("Python hsi version: {}".format(hsi.__version__))
        main()
