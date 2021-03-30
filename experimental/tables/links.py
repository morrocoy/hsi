import os.path

import tables
import numpy as np


class Particle(tables.IsDescription):
    name = tables.StringCol(16)   # 16-character String
    idnumber = tables.Int64Col()      # Signed 64-bit integer
    ADCcount = tables.UInt16Col()     # Unsigned short integer
    TDCcount = tables.UInt8Col()      # unsigned byte
    grid_i = tables.Int32Col()      # 32-bit integer
    grid_j = tables.Int32Col()      # 32-bit integer
    pressure = tables.Float32Col()    # float  (single-precision)
    energy = tables.Float64Col()    # double (double-precision)



data_path = os.path.join(os.getcwd(), "..", "..", "data")
pict_path = os.path.join(os.getcwd(), "..", "..", "pictures")

# Open a file in "w"rite mode
fileName = "links1.h5"
h5file = tables.open_file(os.path.join(data_path, fileName), mode = "w")

# create groups
g1 = h5file.create_group('/', 'g1')
g2 = h5file.create_group(g1, 'g2')

# put some datasets
a1 = h5file.create_carray(g1, 'a1', tables.Int64Atom(), shape=(10000,))
t1 = h5file.create_table(
    g2, 't1', {'f1': tables.IntCol(), 'f2': tables.FloatCol()})

# create another group with hard
gl = h5file.create_group('/', 'gl')
ht = h5file.create_hard_link(gl, 'ht', '/g1/g2/t1')  # ht points to t1
print(f"``{ht}`` is a hard link to: ``{t1}``")

t1.remove()
print(f"table continues to be accessible in: ``{h5file.get_node('/gl/ht')}``")

# Finally, close the file (this also will flush all the remaining buffers!)
h5file.close()