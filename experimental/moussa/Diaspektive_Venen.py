# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:43:47 2021

@author: mzomou
"""

import os
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import tkinter as tk
from tkinter import filedialog
from skimage.exposure import rescale_intensity


##### Cube Daten Einlesen

dim=3
dtype=np.float32
size = np.dtype(dtype).itemsize

root = tk.Tk()
root.withdraw()

data_path = os.path.join(os.getcwd(), "..", "..", "data")
pict_path = os.path.join(os.getcwd(), "..", "..", "pictures")

# load hyper spectral image data .........................................
# Pfad = filedialog.askopenfilename()
subfolder = "occlusion"
timestamp = "2016_11_02_16_48_30"
image_file_path = os.path.join(
    data_path, subfolder, timestamp, timestamp + "_SpecCube.dat")

# subfolder = "2021-06-29_aufnahmen_arm"
# timestamp = "2021_06_29_14_18_55"
# image_file_path = os.path.join(
#     data_path, subfolder, timestamp, timestamp + "_SpecCube.dat")

with open(image_file_path,'rb') as file:
    
        dtypeHeader = np.dtype(np.int32)
        dtypeHeader = dtypeHeader.newbyteorder('>')
        buffer = file.read(size*dim)
        header = np.frombuffer(buffer, dtype=dtypeHeader)
       
        dtypeData = np.dtype(dtype)
        dtypeData = dtypeData.newbyteorder('>')
        buffer = file.read()
        cubeData = np.frombuffer(buffer, dtype=dtypeData)
        
cubeData = cubeData.reshape(header, order='C')
cubeData = np.rot90(cubeData)

##### RGB Bild Erzeugen

rval = cubeData[:,:,27]
gval = cubeData[:,:,16]
bval = cubeData[:,:,8]

rgb = (np.stack([rval, gval, bval], axis = 2)).clip(0., 1.)

##### RGB mit gamma Korrektur

scale = 1
mid = 0.6
rgb_weights = [0.2989, 0.5870, 0.1140]
img_gray = np.dot(rgb, rgb_weights)
mean = np.mean(img_gray)
gamma = np.log(mid) / np.log(mean)
rgb_image = (scale * np.power(rgb, scale*gamma)).clip(0., 1.)

##### Absorption

Abs = -np.log(np.abs(cubeData))

#### Segmentierungsmaske Erzeugen

ABS =ndimage.uniform_filter(cubeData[:, :, 10], size=5) 
REF = ndimage.uniform_filter(cubeData[:, :, 30], size=5)

# NDI: Normalisierter Differenz Index            
Index = np.divide(np.subtract(REF, ABS), np.add(REF, ABS))

# Schwellwert
Index[Index>0.9]=0 # Bei Bedarf Kann geändert werden, um die Segmentirungsqualität zu verbessern.
Index[Index<0.2]=0 # Bei Bedarf Kann geändert werden, um die Segmentirungsqualität zu verbessern.
Index[Index>0]=1

# MM Operation
Index = ndimage.binary_fill_holes(Index)  

# Image to label
label_im, nb_labels = ndimage.label(Index)

# nicht relevante Bildobjekte nach Größe Filtern
sizes = ndimage.sum(Index, label_im, range(nb_labels + 1))
sizemax = max(sizes) 
mask_size = sizes < sizemax
remove_pixel = mask_size[label_im]
label_im[remove_pixel] = 0
label_im[label_im>1] = 1

#### Vorfilterung und SNV Transformation

Abs [Abs == np.inf] = 0
AbsMean = ndimage.uniform_filter(Abs, size=5)
mean = np.mean(AbsMean, axis=2)
std = np.std(AbsMean, axis=2)
SNV=(AbsMean - mean[..., np.newaxis])/std[..., np.newaxis]

##### Segmentiereng

RGB_Seg = np.multiply(rgb_image, label_im[:,:,None])
SNV = np.multiply(SNV, label_im[:,:,None])
cubeData = np.multiply(cubeData, label_im[:,:,None])

##### Venenmaske 1 Erstellen: Winkel Index SNV [625-720]

X = SNV[:,:,25:44] # SNV [625-720]
Y1 = np.zeros((480, 640))

for i in range(0, X.shape[0]):
    
    for j in range (0, X.shape[1]):
    
        x = X[i,j,:]
        y = range(len(x))
        angle = np.arctan2(y[-1] - y[0], x[-1] - x[0])
        Y1 [i,j] = angle
        
##### Venenmaske 2 Erstellen: Mittelwert von HS-Image in 750-950 nm

Y2 = np.mean(cubeData[:,:,50:90], axis=2)

##### Kombination von Venenmasken

Y1 [Y1 < 1.5707963267948966] = 1.5707963267948966
Y1 = rescale_intensity(Y1, (np.min(Y1), np.max(Y1)), (0, 100))

Y2 = rescale_intensity(Y2, (np.min(Y2), np.max(Y2)), (0, 100))

Y3 = ((100 - Y1) * Y2)
Y3 = rescale_intensity(Y3, (np.min(Y3), np.max(Y3)), (0, 100))

##### Darstellung
    
cmap = np.loadtxt('cmap_tivita.txt')
T_cmap = ListedColormap(cmap)

fig = plt.figure()

ax1 = fig.add_subplot(221, xticks = [], yticks = [], title = 'RGB Image')
Im1 = ax1.imshow(rgb_image , cmap = T_cmap)

ax2 = fig.add_subplot(222, xticks = [], yticks = [], title = 'Venenmaske 1: Winkel Index (625-720)')
Im2 = ax2.imshow(Y1, cmap = T_cmap)

ax3 = fig.add_subplot(223, xticks = [], yticks = [], title = 'Venenmaske 2: Mittelwerte der WL (750-950)')
Im3 = ax3.imshow(Y2, cmap = 'gray')

ax4 = fig.add_subplot(224, xticks = [], yticks = [], title = 'Venen Detektion')
Im4 = ax4.imshow(Y3, cmap = 'gray')

plt.tight_layout()
plt.show()