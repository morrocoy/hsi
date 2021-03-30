# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 17:12:06 2021

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
            - timestamp: Timestamp of the record as string
            - format: Spectral format
            - hash: checksum of the spectral data using md5 hash
        - numeric predictive:
            - hsidata: Hyperspectral data
            - wavelen: Wavelengths at which the spectral information is sampled
            - masks: Selection masks applied on the hyperspectral images
        - class:
            - not healed with target index 0
            - healed with target index 1
"""
import os.path
from timeit import default_timer as timer

import pandas as pd
import numpy as np
import h5py

from tables_utils import getDirPaths, loadHSMetaData, loadHSData
from hsi import HSIntensity, HSAbsorption

dirPaths = getDirPaths()

# load metadata ..............................................................
fileName = "181022_Resektionsgrenze_Südstadt_Auswertung_27.10.2020.xlsx"
filePath = os.path.join(dirPaths['results'], fileName)
project = "rostock"
hsformat = HSIntensity

dfMetadata = loadHSMetaData(filePath, sheet_name=0, skiprows=1)
dfMetadata['hsformat'] = hsformat.key
# dfHSImages = dfMetadata.apply(lambda x: loadHSImage(x['timestamp']), axis=1)
# dfData = dfMetadata['timestamp'].apply(loadHSData)

# create output file .........................................................
fileName = "rostock_suedstadt_2018-2020_0.h5"
filePath = os.path.join(dirPaths['data'], fileName)

start = timer()

with h5py.File(filePath, 'w') as store:
    store['descr'] = __doc__
    for index, row in dfMetadata.iterrows():
        print("Process index %d ..." % row["pn"])

        group = store.create_group(row["timestamp"])
        baseName = row["timestamp"].replace('-', '_')
        pathName = os.path.join(dirPaths['data'], baseName)
        hsimage, masks = loadHSData(pathName, baseName, hsformat)
        k, m, n = hsimage.shape
        dsSpectra = group.create_dataset(
            name='hsidata', data=hsimage.spectra.astype(np.float32),
            dtype=np.float32, chunks=(k, m, n))
        dsWavelen = group.create_dataset(
            name='wavelen', data=hsimage.wavelen,
            dtype=np.float64, chunks=(k,))
        dsMasks = group.create_dataset(
            name='mask', data=masks,
            dtype=np.int8, chunks=(5, m, n))


# store metadata, hyperspectral image data and masks
with pd.HDFStore(filePath, 'a') as store:
    store.put('metadata', dfMetadata, format='table', data_columns=True)

print("\nElapsed time: %f sec" % (timer() - start))



