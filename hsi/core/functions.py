import numpy as np


def snv(img):
    """
    standard normal variates (SNV) transformation of spectral data
    """
    mean = np.mean(img, axis=0)
    std = np.std(img, axis=0)
    return (img - mean[np.newaxis, ...])/std[np.newaxis, ...]



