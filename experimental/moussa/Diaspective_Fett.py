# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 10:03:50 2021

@author: mzomou
"""
import os

from scipy.signal import savgol_filter
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import tkinter as tk
from tkinter import filedialog
from skimage.exposure import rescale_intensity

from load_cube import load_cube


##### Cube Daten Einlesen


root = tk.Tk()
root.withdraw()

# image_file_path = filedialog.askopenfilename()

data_path = os.path.join(os.getcwd(), "..", "..", "data")
pict_path = os.path.join(os.getcwd(), "..", "..", "pictures")

# load hyper spectral image data .........................................
subfolder = "occlusion"
timestamp = "2016_11_02_16_48_30"

subfolder = "thyroid"
timestamp = "2019_11_14_08_59_25"

subfolder = "Hautlappen"
timestamp = "2021_09_20_15_13_41"
timestamp = "2022_04_04_14_09_02"

image_file_path = os.path.join(
    data_path, subfolder, timestamp, timestamp + "_SpecCube.dat")


cubeData = load_cube(image_file_path)

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

##### Vorfilterung 2e Ableitung der Absorption

Abs [Abs == np.inf] = 0
Abs = ndimage.uniform_filter(Abs, size = 7)
Abs = np.pad (Abs, pad_width = ((0,0),(0,0),(1,1)), mode='symmetric')
Abs = np.diff (Abs, n = 2, axis = 2)
Abs = ndimage.uniform_filter(Abs, size = 7)
AbsDiff = savgol_filter(Abs, window_length= 9 , polyorder= 2, axis = 2, mode = 'mirror')

##### Segmentiereng

RGB_Seg = np.multiply(rgb_image, label_im[:,:,None])
AbsDiff = np.multiply(AbsDiff, label_im[:,:,None])

##### Methode 1: Fett - Winkel Index

# idx0 = range(80, 85)
# # x1 = AbsDiff[:, :, idx0]
# x1 = AbsDiff[:, :, 80:84]
# y1 = range(len(idx0))
#
# Y1 = np.arctan2(y1[-1] - y1[0], x1[-1] - x1[0])

X1 = AbsDiff[:,:,80:84] # 900-920 nm
Y1 = np.zeros((480, 640))

for i in range(0, X1.shape[0]):

    for j in range (0, X1.shape[1]):

        x1 = X1[i,j,:]
        y1 = range(len(x1))
        angle1 = np.arctan2(y1[-1] - y1[0], x1[-1] - x1[0])
        Y1 [i,j] = angle1

Y1 = rescale_intensity(Y1, (np.min(Y1), np.max(Y1)), (0, 100))
# Y1 = np.multiply(Y1, label_im)

##### Methode 2 und 3: Normalisierter Deffirenz Fett Indizes 875-925nm und 925-960nm

AbsDiff [AbsDiff == 0] = 1

ABSD = AbsDiff[:, :, 85]# 925nm

REFD1 = AbsDiff[:, :, 75]# 875nm
REFD2 = AbsDiff[:, :, 92]# 960nm

ABSD1 = ABSD + 1
REFD1 = REFD1 + 1
REFD2 = REFD2 + 1
            
Fett_Index1 = (REFD2 - ABSD1)/(REFD2 + ABSD1)
Fett_Index1 [Fett_Index1 == 0] = np.min(Fett_Index1)
Fett_Index1 = rescale_intensity(Fett_Index1, (np.min(Fett_Index1), np.max(Fett_Index1)), (0, 100)).astype('float64')

Fett_Index2 = (REFD1 - ABSD1)/(REFD1 + ABSD1)
Fett_Index2 [Fett_Index2 == 0] = np.min(Fett_Index2)
Fett_Index2 = rescale_intensity(Fett_Index2, (np.min(Fett_Index2), np.max(Fett_Index2)), (0, 100)).astype('float64')

#### Methode 4: 2e Ableitung der Absorption in 925nm: AbsDiff[925]

AbsDiff_925 = ABSD
AbsDiff_925 [AbsDiff_925 == 1] = np.min(AbsDiff_925)
AbsDiff_925 = rescale_intensity(AbsDiff_925, (np.min(AbsDiff_925), np.max(AbsDiff_925)), (0, 100))
AbsDiff_925 = 100 - AbsDiff_925
AbsDiff_925 [AbsDiff_925 == 100] = 0

#### Darstellung

#Tvita Color-map Einlesen
cmap = np.loadtxt('cmap_tivita.txt')
T_cmap = ListedColormap(cmap)

fig = plt.figure()

# ax1 = fig.add_subplot(231, xticks = [], yticks = [], title = 'RGB Bild')
# Im1 = ax1.imshow(rgb_image)
#
# ax2 = fig.add_subplot(232, xticks = [], yticks = [], title = 'Segmentiertes RGB Bild')
# Im2 = ax2.imshow(RGB_Seg)
#
# ax3 = fig.add_subplot(233, xticks = [], yticks = [], title = 'Winkel 900-920 nm')
# Im3 = ax3.imshow(Y1, cmap = T_cmap)
#
# ax4 = fig.add_subplot(234, xticks = [], yticks = [], title = 'Fett Index1: NDI 925/960 nm')
# Im4 = ax4.imshow(Fett_Index1, cmap = T_cmap)
#
# ax5 = fig.add_subplot(235, xticks = [], yticks = [], title = 'Fett Index2: NDI 875/925 nm')
# Im5 = ax5.imshow(Fett_Index2, cmap = T_cmap)
#
# ax6 = fig.add_subplot(236, xticks = [], yticks = [], title = 'Abs"[925 nm]')
# Im6 = ax6.imshow(AbsDiff_925, cmap = T_cmap)


ax3 = fig.add_subplot(221, xticks = [], yticks = [], title = 'Winkel 900-920 nm')
Im3 = ax3.imshow(Y1, cmap = T_cmap)

ax4 = fig.add_subplot(222, xticks = [], yticks = [], title = 'Fett Index1: NDI 925/960 nm')
Im4 = ax4.imshow(Fett_Index1, cmap = T_cmap)

ax5 = fig.add_subplot(223, xticks = [], yticks = [], title = 'Fett Index2: NDI 875/925 nm')
Im5 = ax5.imshow(Fett_Index2, cmap = T_cmap)

ax6 = fig.add_subplot(224, xticks = [], yticks = [], title = 'Abs"[925 nm]')
Im6 = ax6.imshow(AbsDiff_925, cmap = T_cmap)

plt.show()