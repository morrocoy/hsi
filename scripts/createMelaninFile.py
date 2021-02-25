# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 12:18:38 2021

@author: kai
"""
import os.path
import numpy as np

from hsi.misc import getPkgDir

data_path = os.path.join(getPkgDir(), "materials")

# melanin by Jaques (https://omlc.org/spectra/melanin/mua.html):
# formula requires wavelength in nm

wavelen = np.arange(300, 1105, 5)
attcoef = 1.7e12 * wavelen ** -3.48
    
header ="melanin for skin by Jaques (https://omlc.org/spectra/melanin/mua.html):"
header += "\nmu_a = 1.7e12 * wavelen [nm] ** -3.48"
header += "\nlambda	absorption"
header += "\n[nm]	[cm-1]"
filePath = os.path.join(data_path, "Melanin by Jaques.txt")

np.savetxt(filePath, np.column_stack((wavelen, attcoef)), 
           fmt='%.1f %.5e', header=header)


header ="melanin for skin by Hermann (no reference):"
header += "\nmu_a = 519 * (wavelen [nm] / 500) ** -3.75"
header += "\nlambda	absorption"
header += "\n[nm]	[cm-1]"
filePath = os.path.join(data_path, "Melanin by Hermann.txt")

attcoef = 519. * (wavelen / 500) ** -3.75
np.savetxt(filePath, np.column_stack((wavelen, attcoef)), 
           fmt='%.1f %.5e', header=header)
