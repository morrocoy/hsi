import os.path

import tables
import numpy as np


data_path = os.path.join(os.getcwd(), "..", "..", "data")
pict_path = os.path.join(os.getcwd(), "..", "..", "pictures")

# Open a file in "w"rite mode
fileName = "tutorial1.h5"
h5file = tables.open_file(os.path.join(data_path, fileName), mode="a")

# print all nodes
for node in h5file:
    print(node)

# print groups only
for group in h5file.walk_groups():
    print(group)

# print all arrays by walking through groups and list nodes (classname filter)
for group in h5file.walk_groups(where="/"):  # / - root tree
    for array in h5file.list_nodes(group, classname='Array'):
        print(array)

# print all arrays in the tree by walking through nodes (classname filter)
for array in h5file.walk_nodes(where="/", classname="Array"):
    print(array)
        

# access table object through the standard attribute attrs in Leaf nodes
table = h5file.root.detector.readout
# access table object using File.get_node() method
table = h5file.get_node("/detector", "readout")

# setting and getting attributes
table.attrs.gath_date = "Wed, 06/12/2003 18:33"
table.attrs.temperature = 18.4
table.attrs.temp_scale = "Celsius"

print(table.attrs.gath_date)
print(table.attrs.temperature)
print(table.attrs.temp_scale)

print(table.attrs._f_list("all"))
print(table.attrs._f_list("sys"))
print(table.attrs._f_list("user"))

print(repr(table.attrs))

# getting object's metadata
print("Object:", table)
print("Table name:", table.name)
print("Table title:", table.title)
print("Number of rows in table:", table.nrows)
print("Table variable names with their type and shape:")
for name in table.colnames:
    print(name, ':= %s, %s' % (table.coldtypes[name], table.coldtypes[name].shape))

# examine metadata in /columns/pressure array using File.get_node() method
pressureObject = h5file.get_node("/columns", "pressure")
print("Info on the object:", repr(pressureObject))
print("  shape: ==>", pressureObject.shape)
print("  title: ==>", pressureObject.title)
print("  atom: ==>", pressureObject.atom)

# Reading data from Array objects
pressureArray = pressureObject.read()
print(pressureArray)
print("pressureArray is an object of type:", type(pressureArray))  # np.ndarray

nameArray = h5file.root.columns.name.read()
print("nameArray is an object of type:", type(nameArray))  # list

print("Data on arrays nameArray and pressureArray:")
for i in range(pressureObject.shape[0]):
    print(nameArray[i], "-->", pressureArray[i])

# Finally, close the file (this also will flush all the remaining buffers!)
h5file.close()