import os.path
import numpy as np

import matplotlib.pyplot as plt

import hsi
from hsi import HSIntensity, HSAbsorption, HSRefraction
from hsi import HSComponent, HSComponentFile
from hsi.analysis import HSCoFit



fpath = "basevectors_1.txt"

with HSComponentFile(fpath) as file:
    file.set_format(HSAbsorption)
    vectors, spec, wavelen = file.read()


spec_ddy = np.diff(spec["spec"], n=2, axis=0)
spec_ddx = wavelen[1:-1]

# retrieve basis spectra
fat = vectors["fat"]

# calculate 2nd derivative over the sampeled basis spectra
fat_ddy = np.diff(fat.yNodeData, n=2)
fat_ddx = fat.xNodeData[1:-1]

# create a component fit analysis using basis spectra of 2nd derivative
analysis = HSCoFit(spec_ddy, spec_ddx)
analysis.add_component(fat_ddy, fat_ddx, name="fat", label="fat", hsformat=HSAbsorption,
                       weight=fat.weight, bounds=fat.bounds)

# plot 2nd derivative together with basis spectra
fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(fat.xNodeData*1e9, fat.yNodeData, label="fat")
ax2.plot(fat_ddx*1e9, fat_ddy, label="fat 2nd der")

ax1.legend()
ax2.legend()

plt.show()
plt.close()


