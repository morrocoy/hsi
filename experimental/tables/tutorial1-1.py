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
fileName = "tutorial1.h5"
h5file = tables.open_file(os.path.join(data_path, fileName), mode = "w")

# Create a new group
group = h5file.create_group("/", 'detector', 'Detector information')
# Creating a new table
table = h5file.create_table(group, 'readout', Particle, "Readout example")

# print info
print(h5file)

# get a pointer to the Row
particle = table.row


# fill table
for i in range(10):
    # particle['name']  = f'Pßarticle: {i:6d}'
    particle['name'] = b'Particle'
    particle['TDCcount'] = i % 256
    particle['ADCcount'] = (i * 256) % (1 << 16)
    particle['grid_i'] = i
    particle['grid_j'] = 10 - i
    particle['pressure'] = float(i*i)
    particle['energy'] = float(particle['pressure'] ** 4)
    particle['idnumber'] = i * (2 ** 34)

    # writes new particle record to the table I/O buffer
    particle.append()

# flush the table’s I/O buffer to write all this data to disk
table.flush()


# read data in a table
table = h5file.root.detector.readout
pressure = [
    x['pressure'] for x in table.iterrows()
    if x['TDCcount'] > 3 and 20 <= x['pressure'] < 50
]
print(pressure)

# read data in a table using in-kernel selection
names = [
    x['name'] for x in table.where(
        """(TDCcount > 3) & (20 <= pressure) & (pressure < 50)""")
]
print(names)


# separate the selected data from the mass of detector data
gcolumns = h5file.create_group(h5file.root, "columns", "Pressure and Name")
h5file.create_array(
    gcolumns, 'pressure', np.array(pressure), "Pressure column selection")
h5file.create_array(gcolumns, 'name', names, "Name column selection")

# Finally, close the file (this also will flush all the remaining buffers!)
h5file.close()