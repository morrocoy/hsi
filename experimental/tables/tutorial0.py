import os.path
import tables as tb


class Particle(tb.IsDescription):
    identity = tb.StringCol(itemsize=22, dflt=" ", pos=0)  # character String
    idnumber = tb.Int16Col(dflt=1, pos = 1)  # short integer
    speed    = tb.Float32Col(dflt=1, pos = 2)  # single-precision


data_path = os.path.join(os.getcwd(), "..", "..", "data")
pict_path = os.path.join(os.getcwd(), "..", "..", "pictures")

# Open a file in "w"rite mode
fileName = "../../data/objecttree.h5"
fileh = tb.open_file(os.path.join(data_path, fileName), mode = "w")

# Get the HDF5 root group
root = fileh.root

# Create the groups
group1 = fileh.create_group(root, "group1")
group2 = fileh.create_group(root, "group2")

# Now, create an array in root group
array1 = fileh.create_array(root, "array1", ["string", "array"], "String array")

# Create 2 new tables in group1
table1 = fileh.create_table(group1, "table1", Particle)
table2 = fileh.create_table("/group2", "table2", Particle)

# Create the last table in group2
array2 = fileh.create_array("/group1", "array2", [1,2,3,4])

# Now, fill the tables
for table in (table1, table2):
    # Get the record object associated with the table:
    row = table.row

    # Fill the table with 10 records
    for i in range(10):
        # First, assign the values to the Particle record
        row['identity']  = f'This is particle: {i:2d}'
        row['idnumber'] = i
        row['speed']  = i * 2.

        # This injects the Record values
        row.append()

    # Flush the table buffers
    table.flush()

# Finally, close the file (this also will flush all the remaining buffers!)
fileh.close()