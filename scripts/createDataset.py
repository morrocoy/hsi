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
            - group: Group containing the hyperspectral data in the source file
            - timestamp: Timestamp of the record
            - format: Spectral format
            - hash: checksum of the spectral data using md5 hash
        - numeric predictive:
            - spectra: Hyperspectral data
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



from hsi import HSImage, HSIntensity, HSAbsorption
from hsi import genHash

def loadHSMetaData(filePath, sheet_name=0, columns=None,
                 skiprows=None, nrows=None):
    """ load meta data.
    """
    columns = None
    if columns is None:
        columns = {
            'pn': { 'index': 3, 'dtype': np.int64 },  # patient number
            'pid': { 'index': 4, 'dtype':  np.int64 },  # patient id
            'descr': { 'index': 6, 'dtype': str },  # details of the wound
            'target': { 'index': 7, 'dtype': int },  # healed or not
            'group': { 'index': 8, 'dtype': str },  # ref path within h5 file
            }
        
    names = columns.keys()
    usecols = [col['index'] for col in columns.values()]
    converters = { key: col['dtype'] for key, col in columns.items() }
    
        
    df = pd.read_excel(
        filePath, sheet_name=sheet_name, names=names, 
        usecols=usecols, skiprows=skiprows, nrows=nrows, converters=converters)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.astype(converters)
    
    # retrieve timestamp from group name
    df['timestamp'] = pd.to_datetime(df['group'], format="%Y-%m-%d-%H-%M-%S")
    
    return df
    

def loadHSData(pathName, baseName):
    """ load hyperspectral data.
    """
        
    filePath = os.path.join(pathName, baseName + "_SpecCube.dat")
    hsImage = HSImage(filePath)
    nwavelen, rows, cols = hsImage.shape

    # tissue and selection masks
    nmask = 4
    masks = np.zeros((nmask, rows, cols))
    masks[0, ...] = hsImage.getTissueMask([0.1, 0.9])

    filePath = os.path.join(pathName, baseName + "_Masks.npz")
    maskfile = np.load(filePath)
    for imask in range(1, nmask):
        key = "mask%d" % imask
        masks[imask, ...] = maskfile[key] * masks[0]
    
    return hsImage, masks



dataPath = os.path.join(os.getcwd(), "..", "data")
pictPath = os.path.join(os.getcwd(), "..", "pictures")
rsltPath = os.path.join(os.getcwd(), "..", "results")
# dataPath = os.path.join("c:", os.sep, "temp")


# load metadata ..............................................................
fileName = "181022_Resektionsgrenze_Südstadt_Auswertung_27.10.2020.xlsx"
filePath = os.path.join(rsltPath, fileName)
project = "rostock"
hsformat = HSIntensity

# load dataframes
# dfMetadata = loadHSMetaData(filePath, sheet_name=0, skiprows=1, nrows=2)
dfMetadata = loadHSMetaData(filePath, sheet_name=0, skiprows=1)
dfMetadata['format'] = hsformat.key
# dfHSImages = dfMetadata.apply(lambda x: loadHSImage(x['timestamp']), axis=1)
# dfData = dfMetadata['timestamp'].apply(loadHSData)


# create output file .........................................................
fileName = "rostock_suedstadt_2018-2020"
filePath = os.path.join(dataPath, fileName + ".h5")

start = timer()


n = len(dfMetadata.index)
checksum = np.zeros((n, 3), dtype='>f4')

hashes = []
with h5py.File(filePath, 'w') as store:

    store['descr'] = __doc__
    for i, key in enumerate(dfMetadata['group']):
        group = store.create_group(key)

        baseName = key.replace('-', '_')
        pathName = os.path.join(dataPath, project, baseName)
        hsimage, masks = loadHSData(pathName, baseName)
        hsimage.setFormat(hsformat)
        k, m, n = hsimage.shape
        dsSpectra = group.create_dataset(
            name='spectra', data=hsimage.spectra, dtype='>f4',
            chunks=(k, m, n))#, compression="gzip", compression_opts=4)
        dsWavelen = group.create_dataset(
            name='wavelen', data=hsimage.wavelen, dtype='f8',
            chunks=(k,))
        dsMasks = group.create_dataset(
            name='masks', data=masks, dtype='i4',
            chunks=(4,m, n))#, compression="gzip", compression_opts=4)
        # group.attrs['format'] = hsimage.format.key
        # dsSpectra.attrs['format'] = hsimage.format.key

        # checksum
        hashes.append(genHash(hsimage.spectra))

        checksum[i, 0] = dfMetadata['pid'][i]
        checksum[i, 1] = np.mean(hsimage.spectra)
        checksum[i, 2] = np.mean(masks)

dfMetadata['hash'] = hashes

# store metadata, hyperspectral image data and masks
with pd.HDFStore(filePath, 'a') as store:
      store.append('metadata', dfMetadata, format='table', data_columns=True)

# checksum
np.save(os.path.join(dataPath, fileName + "_cs.npy"),
        checksum)  # store checksum



print("\nElapsed time: %f sec" % (timer() - start))



