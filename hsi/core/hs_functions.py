import numpy as np

from ..log import logmanager

logger = logmanager.getLogger(__name__)


def snv(img):
    """
    standard normal variates (SNV) transformation of spectral data
    """
    mean = np.mean(img, axis=0)
    std = np.std(img, axis=0)
    return (img - mean[np.newaxis, ...])/std[np.newaxis, ...]


def rescale_intensity(image, in_range, out_range):
    """Return image after stretching or shrinking its intensity levels.
    The desired intensity range of the input and output, `in_range` and
    `out_range` respectively, are used to stretch or shrink the intensity range
    of the input image. See examples below.
    Parameters
    ----------
    image : array
        Image array.
    in_range, out_range : 2-tuple
        Min and max intensity values of input and output image.
    Returns
    -------
    out : array
        Image array after rescaling its intensity. This image is the same dtype
        as the input image.
    """
    imin, imax = in_range
    omin, omax = out_range

    if np.any(np.isnan([imin, imax, omin, omax])):
        logger.debug(
            "WARNING: One or more intensity levels are NaN. Rescaling will "
            "broadcast NaN to the full image."
        )

    image = np.clip(image, imin, imax)

    if imin != imax:
        image = (image - imin) / (imax - imin)
        return np.asarray(image * (omax - omin) + omin)
    else:
        return np.clip(image, omin, omax)