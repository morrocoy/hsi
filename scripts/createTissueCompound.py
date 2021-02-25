# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:42:17 2021

@author: kai
"""
import os.path
import numpy as np

import matplotlib.pyplot as plt
from hsi import HSTissueCompound

data_path = os.path.join(os.getcwd(), "..", "data")
pict_path = os.path.join(os.getcwd(), "..", "pictures")


# create compound .............................................................
wavelen = np.linspace(500, 1000, 100, endpoint=False)
skintype = 'dermis'  # can be ('epidermis', 'dermis', 'bone', 'musle', 'mucosa')
portions = {
    'blo': 0.03,  # blood
    'ohb': 0.5,  # oxygenated hemoglobin (O2HB)
    'hhb': 0.5,  # deoxygenation (HHB) - should be (1 - 'ohb')
    'methb': 0.,  # methemoglobin
    'cohb': 0.,  # carboxyhemoglobin
    'shb': 0.,  # sulfhemoglobin
    'wat': 0.6,  # water
    'fat': 0.2,  # fat
    'mel': 0 * 0.025,  # melanin
}
# notes:
# - portion 'ohb' correspond to blood oxygenation (SpO2)
# - portion 'hhb' should be (1 - 'ohb')
# - sum of ('blo', 'wat', 'fat', 'mel') should be < 1

compound = HSTissueCompound(
    portions=portions, skintype=skintype, wavelen=wavelen)
compound.evaluate()

# retrieve optical parameters of tissue compound
mu_a = compound.absorption  # absorption coefficients
mu_sp = compound.rscattering  # reduced scattering coefficients
mu_s = compound.scattering  # scattering coefficients
g = compound.anisotropy  # anisotropy of scattering

# plot parameters .............................................................
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_yscale('log')

marker_options = {
    'markersize': 7,
    'markeredgewidth': 0.3,
    'markerfacecolor': 'none',
    'markevery':5,
}
ax.plot(wavelen, mu_a, label='absorption [cm-1]', marker='o', **marker_options)
ax.plot(wavelen, mu_s, label='scattering [cm-1]', marker='s', **marker_options)
ax.plot(wavelen, mu_sp, label='reduced scattering [cm-1]', marker='^', **marker_options)
ax.plot(wavelen, g, label='anisotropy', marker='*', **marker_options)

ax.set_xlabel("wavelength [nm]")
ax.set_ylabel("optical parameters")
ax.legend()

plt.savefig(os.path.join(pict_path, "tissue_compound_example.png"))
plt.show()
plt.close(fig)

# export optical tissue parameters ............................................
header ="tissue compound example"
header += "\nlambda mu_a mu_sp mu_s g"
header += "\n[nm] [cm-1] [cm-1] [cm-1] []"
np.savetxt(os.path.join(data_path, "tissue_compound_example.txt"),
           np.column_stack((wavelen, mu_a, mu_sp, mu_s, g)),
           fmt='%.1f %.5e %.5e %.5e %.5e', header=header)