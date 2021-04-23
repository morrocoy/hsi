# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:45:53 2021

@author: kpapke

Example component fit analysis applied to hyperspectral image
"""
import sys
import os
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt

from hsi import cm
from hsi import HSAbsorption, HSIntensity, convert
from hsi import HSImage
from hsi.analysis import HSComponentFit


data_path = os.path.join(os.getcwd(), "..", "data")
pict_path = os.path.join(os.getcwd(), "..", "pictures")

# load hyperspectral image
# subfolder = "thyroid"
# timestamp = "2019_11_14_08_59_25"
subfolder = "occlusion"
timestamp = "2016_11_02_16_48_30"
filePath = os.path.join(data_path, subfolder, timestamp, timestamp)

hsImage = HSImage(filePath + "_SpecCube.dat")
hsImage.setFormat(HSAbsorption)

# apply filter
# hsImage.addFilter(mode='image', type='mean', size=5)
hsImage.addFilter(mode='image', type='gauss', sigma=1, truncate=4)

nwavelen, rows, cols = hsImage.shape
image = hsImage.getRGBValue()
spectra = hsImage.fspectra
wavelen = hsImage.wavelen
mask = hsImage.getTissueMask([0.1, 0.9])

# plot rgb image and gray out non-tissue regions .............................
red = image[:, :, 0]
green = image[:, :, 1]
blue = image[:, :, 2]

idx = np.nonzero(mask == 0)  # gray out region out of mask
gray = 0.2989 * red[idx] + 0.5870 * green[idx] + 0.1140 * blue[idx]
red[idx] = gray
green[idx] = gray
blue[idx] = gray

plt.imshow(image)
plt.show()

# component fit analysis .....................................................
analysis = HSComponentFit(format=HSAbsorption)

analysis.loadtxt("basevectors_3.txt", mode='all')
analysis.setData(spectra, wavelen, format=HSAbsorption)
analysis.setROI([500e-9, 995e-9])

# print base component names
print(analysis.keys)

# modify bounds for component weights
analysis.setVarBounds("hhb", [0, 0.05])
analysis.setVarBounds("ohb", [0, 0.05])
analysis.setVarBounds("wat", [0, 2.00])
analysis.setVarBounds("fat", [0, 1.00])
analysis.setVarBounds("mel", [0, 0.05])

# remove components
# analysis.removeComponent("mel")
print(analysis.keys)


analysis.prepareLSProblem()

a = analysis._anaSysMatrix
start = timer()
analysis.fit(method='bvls_f', mask=mask)
print("Elapsed time: %f sec" % (timer() - start))

# get dictionary of solutions (weights for each component
param = analysis.getSolution(unpack=True, clip=True)

# squared sum of residual over wavelength
res = analysis.getResiduals()

# residual of hyperspectral image
a = analysis._anaSysMatrix  # matrix of base vectors
b = analysis._anaTrgVector  # spectra to be fitted
x = analysis._anaVarVector  # vector of unknowns

res = b - a[:, :-1] @ x[:-1,:]
res = res.reshape(hsImage.shape)

wavelen = analysis.components["hhb"].xIntpData
spectrum = analysis.components["hhb"].yIntpData

# residual conversion from absorption to intensity
res_int = convert(HSIntensity, HSAbsorption, res, wavelen)


# post processing - calculate bood and oxygenation from HHB and O2HB .........
param['blo'] = param['hhb'] + param['ohb']
param['oxy'] = np.zeros(param['blo'].shape)
idx = np.nonzero(param['blo'])
param['oxy'][idx] = param['ohb'][idx] / param['blo'][idx]

# plot solution parameters ...................................................
cmap = cm.tivita()
keys = ['oxy', 'blo', 'wat', 'fat']
labels = ["Oxygenation", "Blood", "Water", "Fat"]

fig = plt.figure()
fig.set_size_inches(10, 8)
for i, key in enumerate(keys):
    ax = fig.add_subplot(2, 2, i + 1)
    if key in ('blo', 'ohb', 'hhb', 'mel'):
        pos = plt.imshow(param[key], cmap=cmap, vmin=0, vmax=0.05)
    elif key in ('oxy', 'fat'):
        pos = plt.imshow(param[key], cmap=cmap, vmin=0, vmax=1)
    elif key in ('wat'):
        pos = plt.imshow(param[key], cmap=cmap, vmin=0, vmax=1.6)
    # fig.colorbar(pos, ax=ax)
    ax.set_title(labels[i])

plt.show()