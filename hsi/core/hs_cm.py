# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 07:42:27 2021

@author: kpapke
"""
import os.path
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from ..misc import getPkgDir

class cm:
    ''' Class for loading colormap.'''

    _maps = ["tivita"]

    def __init__(self):
        pass

    @staticmethod
    def tivita():
        cdict = {
            'red': [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.6, 1.0, 1.0],
                [0.8, 1.0, 1.0],
                [1.0, 100. / 255, 100. / 255],
            ],
            'green': [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.4, 1.0, 1.0],
                [0.6, 1.0, 1.0],
                [0.8, 0.0, 0.0],
                [1.0, 0.0, 0.0]
            ],
            'blue': [
                # [0.0, 100. / 255, 100. / 255],
                [0.0, 0.0, 0.0],
                [0.2, 1.0, 1.0],
                [0.4, 0.0, 0.0],
                [0.6, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        }
        cmap = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
        # colors = np.loadtxt(
        #     os.path.join(getPkgDir(), "data", "cmap_tivita.txt"))
        # cmap = ListedColormap(colors)
        return cmap
