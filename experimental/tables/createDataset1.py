# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 8:52:39 2021

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
import os.path
from timeit import default_timer as timer

import pandas as pd
import numpy as np
import tables

from tables_utils import getDirPaths, loadPatientData, loadHSData
from hsi import HSIntensity, HSAbsorption


class Patient(tables.IsDescription):
    pn = tables.Int64Col()  # Signed 64-bit integer
    pid = tables.Int64Col()  # Signed 64-bit integer
    name = tables.StringCol(32)  # 32-byte character string (utf-8)
    descr = tables.StringCol(64)  # 64-byte character String (utf-8)
    timestamp = tables.StringCol(32)  # 32-byte character String (utf-8)
    hsformat = tables.StringCol(32)  # 32-byte character String (utf-8)
    wavelen = tables.Float64Col(shape=(100, ))  # array of 64-bit floats
    hsidata = tables.Float32Col(shape=(100, 480, 640))  # array of 32-bit floats
    mask = tables.Int8Col(shape=(5, 480, 640))  # array of bytes
    target = tables.Int32Col()  # Signed 32-bit integer


# Patient = np.dtype([
#     ("index", np.int64),  # Signed 64-bit integer
#     ("id", np.int64),  # Signed 64-bit integer
#     ("name", "S16"),  # 16-byte character string
#     ("descr", "S32"),  # 32-byte character string
#     ("timestamp", "S32"),  # 32-byte character string
#     ("hsformat", "S16"),  # 16-byte character string
#     ("wavelen", np.float64, (100, )),  # array of floats (single-precision)
#     ("hsi", np.float32, (100, 640, 480)),  # array of floats (single-precision)
#     ("mask", np.int8, (4, 640, 480))  # array of bytes
#     ("target", np.int32),  # Signed 32-bit integer
#     ])



dirPaths = getDirPaths()

# load metadata ..............................................................
fileName = "181022_Resektionsgrenze_Südstadt_Auswertung_27.10.2020.xlsx"
filePath = os.path.join(dirPaths['results'], fileName)
project = "rostock"
hsformat = HSIntensity

dfMetadata = loadPatientData(filePath, sheet_name=0, skiprows=1)
dfMetadata['hsformat'] = hsformat.key
# dfHSImages = dfMetadata.apply(lambda x: loadHSImage(x['timestamp']), axis=1)
# dfData = dfMetadata['timestamp'].apply(load_hsdata)

# create output file .........................................................
fileName = "rostock_suedstadt_2018-2020_1.h5"
h5file = tables.open_file(os.path.join(dirPaths['data'], fileName), mode="w")


# Create a new group
group = h5file.create_group("/", 'records', 'Clinical records')
group._v_attrs.descr = __doc__  # description of dataset

# Creating a new table
table = h5file.create_table(group, 'patient', Patient, "Patient information")

# fill table
n = len(dfMetadata.index)

# get a pointer to the Row
patient = table.row
for index, row in dfMetadata.iterrows():
    if index > 200:
        break

    print("Process index %d ..." % row["pn"])

    patient["pn"] = row["pn"]  # i f'Particle: {i:6d}'
    patient["pid"] = row["pid"]
    patient["name"] = str.encode("")
    patient["descr"] = str.encode(row["descr"])
    patient["timestamp"] = str.encode(row["timestamp"])
    patient["hsformat"] = str.encode(hsformat.key)
    patient["target"] = row["target"]

    baseName = row["timestamp"].replace('-', '_')
    pathName = os.path.join(dirPaths['data'], baseName)
    hsidata, mask = loadHSData(pathName, baseName, hsformat)

    patient["hsidata"] = hsidata.spectra.astype(np.float32)
    patient["wavelen"] = hsidata.wavelen.astype(np.float64)
    patient["mask"] = mask.astype(np.int8)

    # writes new particle record to the table I/O buffer
    patient.append()

# flush the table’s I/O buffer to write all this data to disk
table.flush()


# Finally, close the file (this also will flush all the remaining buffers!)
h5file.close()


# a = dfMetadata.iloc[0]["timestamp"]

