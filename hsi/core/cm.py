# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 07:42:27 2021

@author: kai
"""
import os.path
import numpy as np
from matplotlib.colors import ListedColormap

from ..misc import getPkgDir

class cm:
    ''' Class for loading colormap.'''

    _maps = ["tivita"]

    def __init__(self):
        pass

    @staticmethod
    def tivita():
        colors = np.loadtxt(
            os.path.join(getPkgDir(), "data", "cmap_tivita.txt"))
        return ListedColormap(colors)
