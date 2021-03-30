import os.path

import tables
import numpy as np


data_path = os.path.join(os.getcwd(), "..", "..", "data")
pict_path = os.path.join(os.getcwd(), "..", "..", "pictures")

# Open a file in "w"rite mode
fileName = "tutorial1.h5"
h5file = tables.open_file(os.path.join(data_path, fileName), mode="a")

# access table object through the standard attribute attrs in Leaf nodes
table = h5file.root.detector.readout
pressureObject = h5file.root.columns.pressure
nameObject = h5file.root.columns.name

# get a pointer to the Row
particle = table.row

# appending data to an existing table
for i in range(10,15):
    particle['name']  = f'Particle: {i:6d}'
    particle['TDCcount'] = i % 256
    particle['ADCcount'] = (i * 256) % (1 << 16)
    particle['grid_i'] = i
    particle['grid_j'] = 10 - i
    particle['pressure'] = float(i*i)
    particle['energy'] = float(particle['pressure'] ** 4)
    particle['idnumber'] = i * (2 ** 34)
    particle.append()
table.flush()


for r in table.iterrows():
    print("%-16s | %11.1f | %11.4g | %6d | %6d | %8d |" % (
        r['name'], r['pressure'], r['energy'], r['grid_i'], r['grid_j'],
        r['TDCcount']))


# modifying data in tables
print("Before modifying-->", table[0:5])
table.cols.energy[0] = 2  # modify cell
table.cols.TDCcount[2:5] = [20, 30, 40] # modify range of TDCcount
print("After modifying first row of ADCcount and energy-->", table[0:5])

#  modify complete sets of second and fifth row
table.modify_rows(start=1, step=3,
                  rows=[(1, 2, 3.0, 4, 5, 6, 'Particle:   None', 8.0),
                        (2, 4, 6.0, 8, 10, 12, 'Particle: None*2', 16.0)])
print("After modifying the complete fifth row-->", table[0:5])

# modify columns via table iterators and Row.update() method
for row in table.where('TDCcount <= 2'):
    row['energy'] = row['TDCcount']*2
    row.update()
print("After modifying energy column (where TDCcount <=2)-->", table[0:4])

# modifying data in arrays (you cannot use negative values for step)
print("Before modif-->", pressureObject[:])
pressureObject[-1] = 2
print("First modif-->", pressureObject[:])
pressureObject[1:3] = [2.1, 3.5]
print("Second modif-->", pressureObject[:])
pressureObject[::2] = [1, 2]
print("Third modif-->", pressureObject[:])

print("Before modif-->", nameObject[:])
nameObject[0] = b'Particle:   None'
print("First modif-->", nameObject[:])
nameObject[1:3] = [b'Particle:      0', b'Particle:      1']
print("Second modif-->", nameObject[:])
nameObject[::2] = [b'Particle:     -3', b'Particle:     -5']
print("Third modif-->", nameObject[:])


# remove rows 5 to 9
table.remove_rows(5,10)

# Finally, close the file (this also will flush all the remaining buffers!)
h5file.close()