# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:53:04 2021

@author: kpapke
"""
import sys
import os.path
from timeit import default_timer as timer
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

from ..misc import genHash, getPkgDir

from ..core.formats import HSFormatFlag, HSFormatDefault, convert
from ..core.HSDataset import HSDataset
from ..core.HSImage import HSImage
from ..core.cm import cm

from .HSTivita import HSTivita

import logging

LOGGING = True
# LOGGING = False
logger = logging.getLogger(__name__)
logger.propagate = LOGGING


__all__ = ['HSBaseStudy']

class HSBaseStudy(object):

    def __init__(self, name, dataset=None, format=None, keys=None):
        """Constructor.

        Parameters
        ----------
        name :  str
            The study name.
        dataset : :obj:`HSDataset<hsi.HSDataset>`, optional
            The dataset to by evaluated.
        format : :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The spectral format to be set. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`

        keys : dict
            A dictionary of selected output parameter keys.
        """
        self.name = name  # path to output file
        self._dirPaths = {}  # dictionary of output directory paths

        self.hsformat = None

        # set data if defined in arguments
        # if dataset is not None:
        #     self.setData(dataset)

        self.cmap = cm.tivita()
        self.fig_options = {
            'dpi': 300,  # resolution
            'pil_kwargs': {
                'quality': 10,  # for jpg
                'optimize': True,  # for jpg
            },
            'bbox_inches': 'tight',
            'pad_inches': 0.03,
        }

        self.keys = ['oxy', 'nir', 'thi', 'twi']
        self.labels = ["Oxygenation", "NIR-Perfusion", "THI", "TWI"]


    def mkdirFor(self, filePath):
        """Create output directories derived from the input filePath

        Parameters
        ----------
        filePath : str
            The path to the input file
        """
        path, fileName = os.path.split(filePath)
        self._dirPaths['main'] = os.path.join(
            path, fileName[:fileName.rfind('.')], self.name)
        self._dirPaths['data'] = os.path.join(
            self._dirPaths['main'], "data")
        self._dirPaths['images'] = os.path.join(
            self._dirPaths['main'], "images")
        self._dirPaths['results'] = os.path.join(
            self._dirPaths['main'], "results")

        for dir in self._dirPaths.values():
            if not os.path.exists(dir):
                os.makedirs(dir)


    def plotMasks(self, fileName, image, masks):
        """Create masked plots of the original rgb image.

        Parameters
        ----------
        fileName : str,
            The file name.
        image : np.ndarray
            The rgb image.
        mask :  dict
            A dictionarry of the mask arrays.
        """
        fig = plt.figure()
        fig.set_size_inches(10, 8)

        d, m = divmod(len(masks), 2)
        cols = d
        rows = d + m
        for i, (label, mask) in enumerate(masks.items()):
            mimage = image.copy()
            red = mimage[:, :, 0]
            green = mimage[:, :, 1]
            blue = mimage[:, :, 2]

            idx = np.nonzero(mask == 0)  # gray out region out of mask
            gray = 0.2989 * red[idx] + 0.5870 * green[idx] + 0.1140 * blue[idx]
            red[idx] = gray
            green[idx] = gray
            blue[idx] = gray

            ax = fig.add_subplot(cols, rows, i + 1)
            plt.imshow(mimage)
            ax.set_title(label)

        ext = fileName[fileName.rfind('.') + 1:]
        filePath = os.path.join(self._dirPaths['images'], fileName)
        fig.savefig(filePath, format=ext, **self.fig_options)
        plt.close(fig)


    def plotParam(self, fileName, param):
        """Create masked plots of the original rgb image.

        Parameters
        ----------
        fileName : str,
            The file name.
        param : dict
            A dictionary of solution parameters

        """
        fig = plt.figure()
        fig.set_size_inches(10, 8)

        d, m = divmod(len(self.keys), 2)
        cols = d
        rows = d + m
        for i, key in enumerate(self.keys):
            ax = fig.add_subplot(cols, rows, i + 1)
            pos = plt.imshow(param[key], cmap=self.cmap, vmin=0, vmax=1)
            fig.colorbar(pos, ax=ax)
            ax.set_title(self.labels[i])

        ext = fileName[fileName.rfind('.') + 1:]
        filePath = os.path.join(self._dirPaths['images'], fileName)
        fig.savefig(filePath, format=ext, **self.fig_options)
        plt.close(fig)


    def exportItem(self, filePath, results, spectra, wavelen, masks, metadata):
        """Export data of an evaluated item

        Parameters
        ----------
        filePath : str
            The path to the output file
        results : dict
            A dictionarry of the solution parameter arrays
        spectra :  numpy.ndarray
            The spectral data.
        wavelen :  numpy.ndarray
            The wavelengths at which the spectral data are sampled.
        masks :  numpy.ndarray
            Masks to be applied on the hyperspectral image.
        metadata : pd.Series
            The metadata of the original dataset item.
        """
        parr = np.array([mask for mask in results.values()])
        marr = np.array([mask for mask in masks.values()])

        hash = genHash(parr)  # checksum
        pn = metadata['pn']

        # if self.hsformat is None:
        series = metadata.replace({'pid': 44, 'hash': hash})
        df = series.to_frame()
        # else:
        #     df = metadata.replace(
        #         {'pid': 44, 'hash': hash, 'format': self.hsformat})

        # metadata.replace({'hash', hash})
        # if self.hsformat is not None:
        #     metadata.replace({'format',  self.hsformat.key})

        with pd.HDFStore(filePath, 'a') as store:
            store.append('metadata', df, format='table', data_columns=True)

        with h5py.File(filePath, 'r+') as store:
            group = store.create_group(metadata['group'])
            group.create_dataset(
                name='param', data=parr, dtype='f8',
                chunks=parr.shape)
            group.create_dataset(
                name='masks', data=marr, dtype='i4',
                chunks=marr.shape)



        # with pd.HDFStore(filePath, 'r+') as store:
        #     df = store['metadata']
        #
        #     index = df.index[df['pn'] == pn]
        #     df.loc[index, 'hash'] = hash
        #     if self.hsformat is not None:
        #         df.loc[index, 'format'] = self.hsformat.key
        #     df.loc[index, 'pid'] = 44


    def run(self, dataset, ):
        if not isinstance(dataset, HSDataset):
            return

        self.mkdirFor(dataset.filePath)

        # initiate output file
        filePath = os.path.join(self._dirPaths['data'], self.name + '.h5')
        with h5py.File(filePath, 'w') as store:
            store['descr'] = dataset.descr
        # with pd.HDFStore(filePath, 'w') as store:
        #     store.append('metadata', dataset.metadata, format='table',
        #                  data_columns=True)

        # run study in serial mode
        # print("\nnEvaluate spectral data (serial)")
        # print("---------------------------------")
        # start = timer()
        # for spectra, wavelen, masks, metadata in dataset.items():
        #     results = self.task(spectra, wavelen, masks, metadata)
        #     self.exportItem(
        #         filePath, results, spectra, wavelen, masks, metadata)
        # print("\nElapsed time: %f sec" % (timer() - start))

        # run study in parallel mode
        print("\nEvaluate spectral data (parallel)")
        print("---------------------------------")
        start = timer()

        nproc = 7
        nitems = 7 #len(dataset)
        with multiprocessing.Pool(processes=nproc) as pool:
            i = 0
            buffer = []
            while i < nitems:
                j = 0
                while i < nitems and j < nproc:
                    buffer.append(dataset[i])
                    i += 1
                    j += 1
                # results = [pool.apply_async(self.task, args) for args in buffer]
                # for spectra, wavelen, masks, metadata in enumerate(items):
                #     self.exportItem(filePath, results.get(timeout=10),
                #                     spectra, wavelen, masks, metadata)
                # [rst.get(timeout=10) for rst in mrst]

                results = pool.starmap(self.task, buffer)
                for j, (spectra, wavelen, masks, metadata) in enumerate(buffer):
                    self.exportItem(
                        filePath, results[j], spectra, wavelen, masks, metadata)
                buffer.clear()

        print("\nElapsed time: %f sec" % (timer() - start))


    def setFormat(self, format):
        """Set the format for the hyperspectral data.

        Parameters
        ----------
        format : :obj:`HSFormatFlag<hsi.HSFormatFlag>`
            The spectral format to be set. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`

        """
        if not HSFormatFlag.hasFlag(format):
            raise Exception("Unknown format '{}'.".format(format))
        self.hsformat = format



    def task(self, spectra, wavelen, masks, metadata):
        """Example task applied on each entry.

        Parameters
        ----------
        metadata : pd.Series
            Metadata of the record.
        spectra :  numpy.ndarray
            The spectral data.
        wavelen :  numpy.ndarray
            The wavelengths at which the spectral data are sampled.
        masks :  numpy.ndarray
            Masks to be applied on the hyperspectral image.

        Returns
        -------
        numpy.ndarray : Array of values for validation.

        """
        basename = "PN_%03d_PID_%07d_Date_%s" % (
            metadata["pn"], metadata["pid"],
            metadata["timestamp"].strftime("%Y-%m-%d-%H-%M-%S"))

        # checksum ............................................................
        hash = genHash(spectra)
        state = "valid" if hash == metadata['hash'] else "invalid"

        print(
            "pn {:10} | spectra {:<10.6f} | hash {} {}".format(
                metadata['pn'], np.mean(spectra), hash, state))
        msg = "pn {:10} | spectra {:<10.6f} | hash {} {}".format(
            metadata['pn'], np.mean(spectra), hash, state)

        # adapt format of spectral data if predefined .........................
        hsformat = HSFormatFlag.fromStr(metadata['format'])  # source format
        if self.hsformat is not None:
            spectra = convert(self.hsformat, hsformat, spectra, wavelen)
            hsformat = self.hsformat

        # image visualization .................................................
        hsImage = HSImage(spectra, wavelen, hsformat)
        image = hsImage.getRGBValue()
        self.plotMasks(basename + "_Masks.jpg", image, masks)

        # analysis ............................................................
        analyzer = HSTivita()
        analyzer.setData(spectra, wavelen, format=hsformat)
        analyzer.evaluate(mask=masks["tissue"])

        param = analyzer.getSolution(unpack=True, clip=True)
        self.plotParam(basename + "_Param.jpg", param)

        # return array of only selected parameters and masks
        rslt = np.array([param[key] for key in self.keys])
        rslt = {key: param[key] for key in self.keys}

        return rslt
        # return msg




