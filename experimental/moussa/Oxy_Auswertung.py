# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:17:45 2022

@author: mzomou
"""
import os.path
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
subfolder = "occlusion"
timestamp = "2016_11_02_16_48_30"

# subfolder = "thyroid"
# timestamp = "2019_11_14_08_59_25"

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

X = SNV[:,:,26:43] # SNV [630-715]
Y1 = np.zeros((NRows, NCols))

for i in range(0, X.shape[0]):
    
    for j in range (0, X.shape[1]):
    
        y = X[i,j,:]
        x = range(len(y))
        angle = np.rad2deg (np.arctan2(y[-1] - y[0], x[-1] - x[0]))
        Y1 [i,j] = angle * 10
        

Y1 [Y1 < -34 ] = -33
Y1 [Y1 > -8 ] = -8
# Wasser_Abs_Index = rescale_intensity(Wasser_Abs_Index, (np.min(Wasser_Abs_Index), np.max(Wasser_Abs_Index)), (0, 100))
Y1 = np.multiply(Y1, label_im)
Y1 [Y1 == 0 ] = -35

print(np.max(Y1))
print(np.min(Y1))

##### Darstellung
    
cmap = np.loadtxt('cmap_tivita.txt')
T_cmap = ListedColormap(cmap)

fig = plt.figure()

# ax1 = fig.add_subplot(121, xticks = [], yticks = [], title = 'RGB Image')
# Im1 = ax1.imshow(rgb_image , cmap = T_cmap)

# ax2 = fig.add_subplot(122, xticks = [], yticks = [], title = 'Winkel Index (625-720)')
# Im2 = ax2.imshow(Y1, cmap = T_cmap, vmin = -34, vmax = -8)

ax2 = fig.add_subplot(111, xticks = [], yticks = [], title = 'Winkel Index (625-720)')
Im2 = ax2.imshow(Y1, cmap = T_cmap, vmin = -34, vmax = -8)


plt.tight_layout()
plt.show()