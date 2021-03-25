# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:42:17 2021

@author: kpapke
"""
import os.path
import numpy as np

from scipy.interpolate import interp1d

from ..log import logmanager
from ..misc import getPkgDir

from .HSTissueComponent import HSTissueComponent

logger = logmanager.getLogger(__name__)

__all__ = ['HSTissueCompound']


class HSTissueCompound:
    """
        Class to represent a tissue compound.

        Objects of this class may be used to creat tissue compounds and to
        evaluate parameters such as total attenuation or anisotropy of
        scattering.

        Attributes
        ----------
        components : dict of :class:`TissueComponent`,
            A dictionary of tissue components.
        portions :  dict
            A dictionary of portions for each component to the compound.
        skintype :  ('epidermis', 'dermis', 'bone', 'musle', 'mucosa')
            The skin type.
        wavelen :  numpy.ndarray
            The wavelengths [nm] at which the spectral information for the
            tissue parameters is given.
        absorption : numpy.ndarray
            The mass attenuation coefficient of the tissue [cm-1].
        scattering : numpy.ndarray
            The scattering coefficient [cm-1].
        rscattering : numpy.ndarray
            The reduced scattering coefficient [cm-1].
        anisotropy : numpy.ndarray
            The anisotropy of scattering
        refraction : numpy.ndarray
            The refractive index.

        """

    def __init__(self, portions=None, skintype='epidermis', wavelen=None,
                 libdir=None):
        """Constructor.

        Parameters
        ----------
        portions :  dict
            A dictionary of portions for each component to the compound.
        skintype :  ('epidermis', 'dermis', 'bone', 'musle', 'mucosa')
            The skin type.
        wavelen :  numpy.ndarray
            The wavelengths [nm] at which the spectral information for the
            tissue parameters is given.
        libdir : str, optional
            The directory for all material input files
        """
        # directory for the parameter files
        if libdir is None:
            libdir = os.path.join(getPkgDir(), "materials")
        self.libdir = libdir

        # dictionary of TissueComponents
        self.components = {}

        # tissue composition
        self.portions = {
            'blo': 0.03,   # blood
            'ohb': 0.5,    # oxygenated hemoglobin (O2HB)
            'hhb': 0.5,    # deoxygenation (HHB)
            'methb': 0.,   # methemoglobin
            'cohb': 0.,    # carboxyhemoglobin
            'shb': 0.,     # sulfhemoglobin
            'wat': 0.6,    # water
            'fat': 0.2,    # fat
            'mel': 0*0.025,  # melanin
        }
        self.setComposition(portions)

        # set wavelength samples
        if wavelen is None:
            wavelen = np.linspace(500., 1000., num=100, endpoint=False)
        self.wavelen = wavelen  # wavelength [nm]

        # skin type ('epidermis', 'dermis', 'bone', 'musle', 'mucosa')
        if skintype in ('epidermis', 'dermis', 'bone', 'musle', 'mucosa'):
            self.skintype = skintype
        else:
            self.skintype = 'epidermis'

        # optical properties of tissues
        self.absorption = None  # mass attenuation coefficient of the tissue
        self.anisotropy = None  # anisotropy of scattering
        self.scattering = None  # scattering coefficient
        self.rscattering = None  # reduced scattering coefficient
        self.refraction = None  # refractive index

        # interpolator on reference data for anisotropy of scattering
        self._anisotropy = None

        # load mass attenuation coefficient of tissue components
        self.loadDefaultComponents()
        self.loadReferenceAnisotropy()


    def addComponentData(self, wavelen, attcoef, name,
                         xunit='nm', yunit='cm-1'):
        """Load spectral data for the mass attenuation coefficient from a file.

        Parameters
        ----------
        attcoef :  numpy.ndarray
            The sampled spectral information for the attenuation coefficient.
        wavelen :  numpy.ndarray
            The wavelengths [nm] at which the spectral information for the
            attenuation coefficient is sampled.
        name : str
            The component name.
        xunit : str, optional
            The unit for wavelength. Can be either of ('pm', 'nm', 'um', 'mm',
            'm'). The default is 'nm'.
        yunit : str, optional
            The unit for mass attenuation coefficient. Can be either of
            ('mm-1', 'cm-1', 'm-1'). The default is 'cm-1'.
        """

        # convert units to nm and cm-1
        if xunit == 'pm':
            xscale = 1e-3
        elif xunit == 'nm':
            xscale = 1
        elif xunit == 'um':
            xscale = 1e3
        elif xunit == 'mm':
            xscale = 1e6
        elif xunit == 'm':
            xscale = 1e9
        else:
            logger.debug("Unknown xunit {}.".format(xunit))
            return

        if yunit == 'm-1':
            yscale = 1e-2
        elif yunit == 'cm-1':
            yscale = 1
        elif yunit == 'mm-1':
            yscale = 10
        else:
            logger.debug("Unknown yunit {}.".format(yunit))
            return

        self.components[name] = HSTissueComponent(
            yscale*attcoef, xscale*wavelen, self.wavelen)


    def addComponentFile(self, filepath, name, skiprows=0, usecols=[0, 1],
                         **options):
        """Load spectral data for the mass attenuation coefficient from a file.

        Parameters
        ----------
        filePath : str
            The intput file path.
        name : str
            The component name.
        skiprows : int, optional
            DESCRIPTION. The default is 4.
        usecols : list of int, optional
            DESCRIPTION. The default is [0, 1].
        **options : dict, optional
            Additional options are forwarded to
            :func:`addComponentData()<TissueCompound.addComponentData>`.
        """
        wavelen, attcoef = np.loadtxt(
            filepath, skiprows=skiprows, usecols=usecols, unpack=True)

        self.addComponentData(wavelen, attcoef, name, **options)

    def evaluate(self):
        """Evaluate optical properties of tissue compound."""
        wavelen = self.wavelen
        portions = self.portions
        components = self.components

        # absorption from blood components
        absorption = np.zeros(wavelen.shape)
        rem = 1. # remainder portion
        for key in ('methb', 'cohb', 'shb'):
            absorption += portions[key] * components[key].absorption
            rem -= portions[key]
        for key in ('ohb', 'hhb'):
            absorption += rem * portions[key] * components[key].absorption
        absorption *= portions['blo']

        # absorption from components other than blood
        for key in ('wat', 'fat', 'mel'):
            absorption += portions[key] * components[key].absorption

        # anisotropy of scattering
        gref = self._anisotropy(wavelen)  # reference anisotropy
        anisotropy = np.zeros(wavelen.shape)
        anisotropy += gref
        anisotropy += (0.98 - gref) * portions['blo']
        anisotropy += (1.00 - gref) * portions['wat']

        # reduced scattering
        rscattering = np.zeros(wavelen.shape)
        rscattering += 22.0 * portions['blo'] * (wavelen / 500) ** -0.660
        rscattering += 13.7 * portions['fat'] * (wavelen / 500) ** -0.385
        # remainder portion
        rem = (1. - portions['blo'] - portions['wat'] - portions['fat'])
        if self.skintype == 'epidermis':
            rscattering += 68.7 * rem * (wavelen / 500) ** -1.161
        elif self.skintype == 'dermis':
            rscattering += 45.3 * rem * (wavelen / 500) ** -1.292
        elif self.skintype == 'bone':
            rscattering += 38.4 * rem * (wavelen / 500) ** -1.470
        elif self.skintype == 'musle':
            rscattering += 9.80 * rem * (wavelen / 500) ** -2.820
        elif self.skintype == 'mucosa':
            rscattering += 18.8 * rem * (wavelen / 500) ** -1.620
        else:
            raise("Unknown skin type {}.".format(self.skintype))

        # refraction
        # formula by Jaques, "Optical properties of biological tissues:
        # a review", Phys. Med. Biol. 58 R37, 2013
        n_dry = 1.514
        n_wat = 1.33
        refraction = np.ones(wavelen.shape)
        refraction *= n_dry - (n_dry - n_wat) * portions['wat']

        # store optical properties of tissue compound in member variables
        self.absorption = absorption
        self.anisotropy = anisotropy
        self.scattering = rscattering / (1. - anisotropy)
        self.rscattering = rscattering
        self.refraction = refraction


    def loadDefaultComponents(self):
        """Load attenuation coefficients for default components.

        The default components comprise:

        - Deoxygenated hemoglobin (HHB) by Gratzer et. al.
        - Oxygenated hemoglobin (O2HB) by Gratzer et. al.
        - Water by Hermann
        - Fat by van Veen et. al., 2004
        - Melanin for skin by Jaques (https://omlc.org/spectra/melanin/mua.html)
        - Methemoglobin by Hermann
        - Carboxyhemoglobin by Hermann
        - Sulfhemoglobin by Hermann

        """

        # deoxygenated hemoglobin (HHB) by Gratzer et. al.
        filePath = os.path.join(self.libdir, "Hemoglobin by Gratzer 2.txt")
        self.addComponentFile(filePath, 'hhb', skiprows=24, usecols=[0, 2])
        logger.debug("Load optical parameters for HHB from "
                     "{}.".format(filePath))

        # oxygenated hemoglobin (O2HB) by Gratzer et. al.
        filePath = os.path.join(self.libdir, "Hemoglobin by Gratzer 2.txt")
        self.addComponentFile(filePath, 'ohb', skiprows=24, usecols=[0, 1])
        logger.debug("Load optical parameters for O2HB from "
                     "{}.".format(filePath))

        # water by Hermann:
        filePath = os.path.join(self.libdir, "Water by Hermann.txt")
        self.addComponentFile(filePath, 'wat', skiprows=4)
        logger.debug("Load optical parameters for Water from "
                     "{}.".format(filePath))

        # water by Segelstein et. al.
        # filePath = os.path.join(self.libdir, "Water by Segelstein 1981.txt")
        # data = np.loadtxt(filePath, skiprows=4)
        # data[:, 1] = 4 * np.pi * data[:, 2] / data[:, 1]  # refractive index to mu_a
        # self.addComponentData(data[:, 0], data[:, 1], 'wat', xunit='um')
        # logger.debug("Load optical parameters for Water from "
        #              "{}.".format(filePath))

        # fat by van Veen et. al. 2004:
        filePath = os.path.join(self.libdir, "Fat by van Veen 2004.txt")
        self.addComponentFile(filePath, 'fat', skiprows=7, yunit='m-1')
        logger.debug("Load optical parameters for Fat from "
                     "{}.".format(filePath))

        # # melanin by Jaques:
        # filePath = os.path.join(self.libdir, "Melanin by Jaques.txt")
        # self.addComponentFile(filePath, 'mel', skiprows=4)
        # logger.debug("Load optical parameters for Melanin from "
        #              "{}.".format(filePath))

        # melanin by Hermann:
        filePath = os.path.join(self.libdir, "Melanin by Hermann.txt")
        self.addComponentFile(filePath, 'mel', skiprows=4)
        logger.debug("Load optical parameters for Melanin from "
                     "{}.".format(filePath))

        # methemoglobin by Hermann:
        filePath = os.path.join(self.libdir, "Methemoglobin by Hermann.txt")
        self.addComponentFile(filePath, 'methb', skiprows=4)
        logger.debug("Load optical parameters for Methemoglobin from "
                     "{}.".format(filePath))

        # carboxyhemoglobin by Hermann:
        filePath = os.path.join(self.libdir, "Carboxyhemoglobin by Hermann.txt")
        self.addComponentFile(filePath, 'cohb', skiprows=4)
        logger.debug("Load optical parameters for Carboxyhemoglobin from "
                     "{}.".format(filePath))

        # sulfhemoglobin  by Hermann
        filePath = os.path.join(self.libdir, "Sulfhemoglobin by Hermann.txt")
        self.addComponentFile(filePath, 'shb', skiprows=4)
        logger.debug("Load optical parameters for Sulfhemoglobin from "
                     "{}.".format(filePath))


    def loadReferenceAnisotropy(self, filePath=None, skiprows=4, usecol=[0, 1]):
        """Load rereference data for the anisotrop of scattering.

        Parameters
        ----------
        filePath : str, optional
            The absolute path to the input file.
        skiprows : int, optional
            DESCRIPTION. The default is 4.
        usecols : list of int, optional
            DESCRIPTION. The default is [0, 1].
        """
        if filePath is None:
            filePath = os.path.join(self.libdir, "g by Hermann.txt")
            skiprows = 4
            usecol = [0, 1]

        wavelen, anisotropy = np.loadtxt(
            filePath, skiprows=skiprows, usecols=usecol, unpack=True)
        self.setReferenceAnisotropy(wavelen, anisotropy, kind='linear')

        logger.debug("Load reference data for anisotropy of scattering "
                     "from {}.".format(filePath))


    def setComposition(self, portions):
        """Set portions for each tissue component to the compound.

        Parameters
        ----------
        portions : dict
            A dictionary of portions for each component to the compound.
        """
        if isinstance(portions, dict):
            for key, val in portions.items():
                if key in self.portions:
                    self.portions[key] = val
                    logger.debug(
                        "Set component portion {} to {}.".format(key, val))


    def setReferenceAnisotropy(self, wavelen, anisotropy, **options):
        """Set the reference data for anisotropy of scattering.

        Parameters
        ----------
        wavelen :  numpy.ndarray, optional
            The wavelengths [nm] at which the spectral information for the
            anisotropy of scattering is provided.
        anisotropy : numpy.ndarray
            The reference anisotropy of scattering.
        **options : dict, optional
            Options for the interpolation forwarded to
            :class:`scipy.interpolate.interp1d`.
        """
        self._anisotropy = interp1d(wavelen, anisotropy, **options)

        logger.debug("Apply interpolator on reference data for anisotropy of "
                     "scattering.")


    def setSkinType(self, skintype):
        """Set the skin type

        Parameters
        ----------
        skintype :  ('epidermis', 'dermis', 'bone', 'musle', 'mucosa')
            The skin type.
        """
        if skintype in ('epidermis', 'dermis', 'bone', 'musle', 'mucosa'):
            self.skintype = skintype
            logger.debug("Set skin type to {}.".format(skintype))
        else:
            logger.debug("Unknown skin type {}.".format(skintype))