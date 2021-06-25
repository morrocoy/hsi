# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:43:35 2021

@author: kpapke
"""
import sys
import os.path
import logging

import time
from timeit import default_timer as timer
import multiprocessing
import queue


from multiprocessing import Process, Pipe


import pandas as pd
import numpy as np
import tables

from tables_utils import getDirPaths, plotMasks, plotParam

import hsi
from hsi import HSImage, HSIntensity, HSAbsorption, HSFormatFlag
from hsi import genHash
from hsi.analysis import HSTivita
from hsi.log import logmanager

logger = logmanager.getLogger(__name__)



# All access to the file goes through a single instance of this class.
# It contains several queues that are used to communicate with other
# processes.
# The read_queue is used for requests to read data from the HDF5 file.
# A list of result_queues is used to send data back to client processes.
# The write_queue is used for requests to modify the HDF5 file.
# One end of a pipe (shutdown) is used to signal the process to terminate.
class FileAccess(multiprocessing.Process):

    def __init__(self, filePath, readQueue, resultQueues, writeQueue,
                 shutdown):
        self.h5file = None
        self.filePath = filePath
        self.readQueue = readQueue
        self.resultQueues = resultQueues
        self.writeQueue = writeQueue
        self.shutdown = shutdown
        self.blockPeriod = .01
        super().__init__()

    def run(self):
        self.h5file = tables.open_file(self.filePath, 'r+')

        # self.group = self.h5file.root.records
        # self.table = self.h5file.root.records.patient
        self.table = self.h5file.get_node('/records/patient')

        another_loop = True
        while another_loop:

            # Check if the process has received the shutdown signal.
            if self.shutdown.poll():
                another_loop = False

            # Check for any data requests in the read_queue.
            try:
                row_num, proc_num = self.readQueue.get(
                    True, self.blockPeriod)
                # look up the appropriate result_queue for this data processor
                # instance
                result_queue = self.resultQueues[proc_num]
                print('processor {} reading from row {}'.format(proc_num,
                                                                  row_num))
                result_queue.put(self.read_row(row_num))
                another_loop = True
            except queue.Empty:
                pass

            # Check for any write requests in the write_queue.
            try:
                row_num, data = self.writeQueue.get(True, self.blockPeriod)
                print('writing row', row_num)
                self.write_row(row_num, data)
                another_loop = True
            except queue.Empty:
                pass

        # close the HDF5 file before shutting down
        self.h5file.close()

    def read_row(self, row_num):
        return self.array[row_num, :]

    def write_row(self, row_num, data):
        self.array[row_num, :] = data


# This class represents a process that does work by reading and writing to the
# HDF5 file.  It does this by sending requests to the FileAccess class instance
# through its read and write queues.  The data results are sent back through
# the result_queue.
# Its actions are logged to a text file.
class DataProcessor(multiprocessing.Process):

    def __init__(self, read_queue, result_queue, write_queue, proc_num,
                 array_size, output_file):
        self.read_queue = read_queue
        self.result_queue = result_queue
        self.write_queue = write_queue
        self.proc_num = proc_num
        self.array_size = array_size
        self.output_file = output_file
        super().__init__()

    def run(self):
        self.output_file = open(self.output_file, 'w')
        # read a random row from the file
        row_num = np.random.random(self.array_size)
        self.read_queue.put((row_num, self.proc_num))
        self.output_file.write(str(row_num) + '\n')
        self.output_file.write(str(self.result_queue.get()) + '\n')

        # modify a random row to equal 11 * (self.proc_num + 1)
        row_num = np.random.random(self.array_size)
        new_data = (np.zeros((1, self.array_size), 'i8') +
                    11 * (self.proc_num + 1))
        self.write_queue.put((row_num, new_data))

        # pause, then read the modified row
        time.sleep(0.015)
        self.read_queue.put((row_num, self.proc_num))
        self.output_file.write(str(row_num) + '\n')
        self.output_file.write(str(self.result_queue.get()) + '\n')
        self.output_file.close()


def task(patient):
    hsformat = HSFormatFlag.from_str(patient["hsformat"].decode())

    print("%8d | %8d | %-20s | %-20s | %-10s | %3d |" % (
        patient["pn"],
        patient["pid"],
        patient["descr"].decode(),
        patient["timestamp"].decode(),
        hsformat.key,
        patient["target"]
    ))

    hsImage = HSImage(
        spectra=patient["hsidata"], wavelen=patient["wavelen"], hsformat=hsformat)
    image = hsImage.as_rgb()

    keys = [
        "tissue",
        "critical wound region",
        "wound region",
        "wound and proximity",
        "wound proximity"
    ]
    mask = {key: val for (key, val) in zip(keys, patient["mask"])}

    fileName = "PN_%03d_PID_%07d_Date_%s_Masks.jpg" % (
        patient["pn"], patient["pid"], patient["timestamp"])
    # plotMasks(fileName, image, mask)

    analysis = HSTivita(hsformat=HSIntensity)
    analysis.set_data(hsImage.spectra, hsImage.wavelen, hsformat=hsformat)
    analysis.evaluate(mask=mask["tissue"])
    param = analysis.get_solution(unpack=True, clip=True)
    # param = None
    fileName = "PN_%03d_PID_%07d_Date_%s_Tivita.jpg" % (
        patient["pn"], patient["pid"], patient["timestamp"])
    # plotParam(fileName, param)

    return param



def main():

    dirPaths = getDirPaths()

    start = timer()

    # open file in (r)ead mode
    fileName = "rostock_suedstadt_2018-2020_1.h5"
    h5file = tables.open_file(os.path.join(dirPaths['data'], fileName), mode="r")

    group = h5file.root.records
    table = h5file.root.records.patient

    # print(group._v_attrs.descr)
    # print(repr(table))

    # print("\nSerial evaluation")
    # print("---------------------")
    # for patient in table.iterrows():
    #     task(patient)

    print("\nParallel evaluation")
    print("---------------------")

    nproc = 7
    nitems = table.nrows
    print("Items: ", nitems)

    parmap(task, table.iterrows())

    # with multiprocessing.Pool(processes=nproc) as pool:

    #     # apply_async with unlimited pool size
    #     # mrst = [
    #     #     pool.apply_async(task, (buf.fetch_all_fields(),))
    #     #     for buf in table.iterrows()]
    #     # [rst.get(timeout = 3) for rst in mrst]
    #
    #     # apply_async with limited pool size
    #     for i in range(0, nitems, nproc):
    #         mrst = [
    #             pool.apply_async(task, (table[i + j],))
    #             for j in range(nproc if i + nproc < nitems else nitems - i)
    #         ]
    #         # [rst.get(timeout=10) for rst in mrst]
    #         [rst.get() for rst in mrst]

        # starmap with unlimited pool size
        # rst = pool.starmap(task, [(buf.fetch_all_fields(),)
        #     for buf in table.iterrows()])

        # for i in range(0, nitems, nproc):
        #     rst = pool.starmap(task, [
        #         (table[i + j],)
        #         for j in range(nproc if i + nproc < nitems else nitems - i)
        #     ])

        # i = 0
        # j = 0
        # buffer = []
        # while i < nitems:
        #     while i < nitems and j < nproc:
        #         patient = table[i]
        #
        #         metadata = {
        #             key: patient[key] for key in
        #             ["pn", "pid", "descr", "timestamp", "hsformat", "target"]
        #         }
        #         hsidata =  patient["hsidata"]
        #         wavelen = patient["wavelen"]
        #         mask = patient["mask"]
        #         buffer.append((patient, hsidata, wavelen, mask))
        #
        #         # buffer.append((patient,))
        #         i += 1
        #         j += 1
        #
        #     rst = pool.starmap(task_split, buffer)
        #     buffer.clear()
        #     j = 0



    # Finally, close the file (this also will flush all the remaining buffers!)
    h5file.close()

    print("\nElapsed time: %f sec" % (timer() - start))



if __name__ == '__main__':
    # logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()
