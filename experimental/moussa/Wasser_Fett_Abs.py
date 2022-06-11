# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 13:02:36 2022

@author: mzomou
"""
import os.path
from scipy.signal import savgol_filter
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import tkinter as tk
from tkinter import filedialog
# from skimage.exposure import rescale_intensity

##### Cube Daten Einlesen

dim=3
dtype=np.float32
size = np.dtype(dtype).itemsize

root = tk.Tk()
root.withdraw()

data_path = os.path.join(os.getcwd(), "..", "..", "data")
pict_path = os.path.join(os.getcwd(), "..", "..", "pictures")

# load hyper spectral image data .........................................
subfolder = "occlusion"
timestamp = "2016_11_02_16_48_30"

# subfolder = "thyroid"
# timestamp = "2019_11_14_08_59_25"
#
# subfolder = "Hautlappen"
# timestamp = "2021_09_20_15_13_41"
# timestamp = "2021_09_27_15_42_49"
# timestamp = "2021_10_04_15_42_23"
# timestamp = "2021_11_15_16_17_29"
# timestamp = "2022_04_04_14_09_02"
image_file_path = os.path.join(
    data_path, subfolder, timestamp, timestamp + "_SpecCube.dat")

# Pfad = filedialog.askopenfilename()

Pfad = image_file_path
with open(Pfad,'rb') as file:
    
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

imgSize    = np.shape(cubeData)
NRows      = imgSize[0]
NCols      = imgSize[1]
NBands     = imgSize[2]

##### RGB Bild Erzeugen

rval = cubeData[:,:,27]
gval = cubeData[:,:,16]
bval = cubeData[:,:,8]

rgb = (np.stack([rval, gval, bval], axis = 2)).clip(0., 1.)

##### RGB mit gamma Korrektur
rgb_image = rgb
# scale = 1
# mid = 0.6
# rgb_weights = [0.2989, 0.5870, 0.1140]
# img_gray = np.dot(rgb, rgb_weights)
# mean = np.mean(img_gray)
# gamma = np.log(mid) / np.log(mean)
# rgb_image = (scale * np.power(rgb, scale*gamma)).clip(0., 1.)

##### Absorption

Abs = -np.log(np.abs(cubeData))

#### Segmentierungsmaske Erzeugen

ABS =ndimage.uniform_filter(cubeData[:, :, 10], size=5) 
REF = ndimage.uniform_filter(cubeData[:, :, 30], size=5)

# NDI: Normalisierter Differenz Index            
Index = np.divide(np.subtract(REF, ABS), np.add(REF, ABS))

# Schwellwert
Index[Index>0.9]=0 # Bei Bedarf Kann geändert werden, um die Segmentirungsqualität zu verbessern.
Index[Index<0.1]=0 # Bei Bedarf Kann geändert werden, um die Segmentirungsqualität zu verbessern.
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

X1 = AbsDiff[:,:,80:84] * 2000  # 900-920 nm
Y1 = np.zeros((NRows, NCols))

for i in range(0, X1.shape[0]):
    
    for j in range (0, X1.shape[1]):
    
        y1 = X1[i,j,:] 
        x1 = range(len(y1))
        angle1 = np.rad2deg(np.arctan2( y1[-1] - y1[0], x1[-1] - x1[0]))
        Y1 [i,j] = angle1
        
# print(np.max(Y1))
# print(np.min(Y1))

Wasser_Abs_Index = Y1.copy()
Fett_Abs_Index = - Y1.copy()

Wasser_Abs_Index [Wasser_Abs_Index < -15 ] = -14
Wasser_Abs_Index [Wasser_Abs_Index > 83 ] = 83
# Wasser_Abs_Index = rescale_intensity(Wasser_Abs_Index, (np.min(Wasser_Abs_Index), np.max(Wasser_Abs_Index)), (0, 100))
Wasser_Abs_Index = np.multiply(Wasser_Abs_Index, label_im)
Wasser_Abs_Index [Wasser_Abs_Index == 0 ] = -15

Fett_Abs_Index [Fett_Abs_Index < -20 ] = -19
Fett_Abs_Index [Fett_Abs_Index > 85 ] = 85
# Fett_Abs_Index = rescale_intensity(Fett_Abs_Index, (np.min(Fett_Abs_Index), np.max(Fett_Abs_Index)), (0, 100))
# Fett_Abs_Index = 105 - Fett_Abs_Index
Fett_Abs_Index = np.multiply(Fett_Abs_Index, label_im)
Fett_Abs_Index [Fett_Abs_Index == 0 ] = -20


#### Darstellung

#Tvita Color-map Einlesen
cmap = np.loadtxt('cmap_tivita.txt')
T_cmap = ListedColormap(cmap)

fig = plt.figure()

ax1 = fig.add_subplot(221, xticks = [], yticks = [], title = 'RGB Bild')
Im1 = ax1.imshow(rgb_image)

ax2 = fig.add_subplot(222, xticks = [], yticks = [], title = 'Segmentiertes RGB Bild')
Im2 = ax2.imshow(RGB_Seg)

ax3 = fig.add_subplot(223, xticks = [], yticks = [], title = 'Absoluter Wasser-Index')
Im3 = ax3.imshow(Wasser_Abs_Index, cmap = T_cmap, vmin = -15, vmax = 83)

ax4 = fig.add_subplot(224, xticks = [], yticks = [], title = 'Absoluter Fett-Index')
Im4 = ax4.imshow(Fett_Abs_Index, cmap = T_cmap, vmin = -20, vmax = 85)

plt.show()