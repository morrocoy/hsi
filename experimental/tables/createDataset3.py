# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:02:50 2021

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
            - hsformat: Spectral hsformat
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



from tables_utils import getDirPaths, loadPatientData, loadHSData
import hsi
from hsi import HSIntensity, HSStore
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

    fileName = "rostock_suedstadt_2018-2020_3.h5"
    filePath = os.path.join(dirPaths['data'], fileName)
    with HSStore.open(filePath, mode="w") as dataset:
        npatient = len(patientData.index)
        dataset.initTables("/records", descr=__doc__, expectedrows=npatient)

        for index, info in patientData.iterrows():
            # if index > 2:
            #     break

            print("Process index %d ..." % info["pn"])
            baseName = info["timestamp"].replace('-', '_')
            pathName = os.path.join(dirPaths['data'], baseName)
            hsidata, masks = loadHSData(pathName, baseName, hsformat)
            dataset.append(
                info=info,
                spectra=hsidata.spectra,
                wavelen=hsidata.wavelen,
                masks=masks
            )

    print("\nElapsed time: %f sec" % (timer() - start))


if __name__ == '__main__':
    if __name__ == '__main__':
        logmanager.setLevel(logging.DEBUG)
        logger.info("Python executable: {}".format(sys.executable))
        logger.info("Python hsi version: {}".format(hsi.__version__))
        main()
