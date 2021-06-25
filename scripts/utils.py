# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:12:11 2020

@author: papkai
"""
import numpy as np
import xlrd


# excel stuff .................................................................
def read_excel(file, sheet, rows, cols, zerochars=None, nanchars=None):

    workbook = xlrd.open_workbook(file)

    if isinstance(sheet, int):
        worksheet = workbook.sheet_by_index(sheet)
    else:
        worksheet = workbook.sheet_by_name(sheet)

    if zerochars is None:
        zerochars = []
    if nanchars is None:
        nanchars = []

    data = []
    for i in (range(len(rows))):
        line = []
        if worksheet.nrows <= rows[i]:
            break
        for j in range(len(cols)):
            readout = worksheet.cell_value(rows[i], cols[j])
            # check for number
            if isinstance(readout, (int, float)):
                if readout == int(readout):
                    line.append(int(readout))
                else:
                    line.append(readout)

            # check for characters which shall be associated with zero
            elif readout in zerochars:
                line.append(0)

            # check for characters which shall be associated with zero
            elif readout in nanchars:
                line.append(np.nan)

            # append readout as string
            else:
                line.append(readout)
        data.append(line)
    return data
