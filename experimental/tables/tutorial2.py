import os.path

import tables
import numpy as np


# Describe a particle record
# class Particle(tables.IsDescription):
#     name        = tables.StringCol(itemsize=16)  # 16-character string
#     lati        = tables.Int32Col()              # integer
#     longi       = tables.Int32Col()              # integer
#     pressure    = tables.Float64Col(shape=(2, 3)) # array of floats (double-precision)
#     temperature = tables.Float32Col(shape=(640, 480)) # array of doubles (single-precision)

Particle = np.dtype([
    ("name", "S16"),  # 16-character string
    ("lati", np.int32),             # integer
    ("longi", np.int32),              # integer
    ("pressure", np.float32, (2, 3)), # array of floats (double-precision)
    ("temperature", np.float32, (640, 480)) # array of doubles (single-precision)
    ])

# Native NumPy dtype instances are also accepted
Event = np.dtype([
    ("name"     , "S16"),
    ("TDCcount" , np.uint8),
    ("ADCcount" , np.uint16),
    ("xcoord"   , np.float32),
    ("ycoord"   , np.float32)
    ])

# And dictionaries too (this defines the same structure as above)
# Event = {
#     "name"     : tb.StringCol(itemsize=16),
#     "TDCcount" : tb.UInt8Col(),
#     "ADCcount" : tb.UInt16Col(),
#     "xcoord"   : tb.Float32Col(),
#     "ycoord"   : tb.Float32Col(),
#     }

data_path = os.path.join(os.getcwd(), "..", "..", "data")
pict_path = os.path.join(os.getcwd(), "..", "..", "pictures")

# Open a file in "w"rite mode
fileName = "tutorial2.h5"
fileh = tables.open_file(os.path.join(data_path, fileName), mode = "w")

# Get the HDF5 root group
root = fileh.root

# Create the groups:
for groupname in ("Particles", "Events"):
    group = fileh.create_group(root, groupname)

# Now, create and fill the tables in Particles group
gparticles = root.Particles

# Create 3 new tables
for tablename in ("TParticle1", "TParticle2", "TParticle3"):
    # Create a table
    table = fileh.create_table(
        "/Particles", tablename, Particle, "Particles: " + tablename)

    # Get the record object associated with the table:
    particle = table.row

    # Fill the table with 257 particles
    for i in range(257):
        # First, assign the values to the Particle record
        particle['name'] = f'Particle: {i:6d}'
        particle['lati'] = i
        particle['longi'] = 10 - i

        # incorrect shape detected by sanity check
        # particle['pressure'] = np.array(i * np.arange(2 * 4)).reshape((2, 4))
        # correct shape
        particle['pressure'] = np.array(i * np.arange(2 * 3)).reshape((2, 3)) # Correct

        particle['temperature'] = i ** 2     # Broadcasting

        # This injects the Record values
        particle.append()

    # Flush the table buffers
    table.flush()

# Now, go for Events:
for tablename in ("TEvent1", "TEvent2", "TEvent3"):
    # Create a table in Events group
    table = fileh.create_table(root.Events, tablename, Event, "Events: "+tablename)

    # Get the record object associated with the table:
    event = table.row

    # Fill the table with 257 events
    for i in range(257):
        # First, assign the values to the Event record
        event['name']  = f'Event: {i:6d}'
        event['TDCcount'] = i % (1<<8)   # Correct range


        # incorrect attribute key detected by sanity check
        # event['xcoor'] = float(i ** 2)
        # correct attribute detected
        event['xcoord'] = float(i ** 2)
        # incorrect attribute type detected by sanity check
        # event['ADCcount'] = "sss"
        # correct attribute type
        event['ADCcount'] = i * 2

        event['ycoord'] = float(i) ** 4

        # This injects the Record values
        event.append()

    # Flush the buffers
    table.flush()

# Read the records from table "/Events/TEvent3" and select some
table = root.Events.TEvent3
e = [ p['TDCcount'] for p in table
      if p['ADCcount'] < 20 and 4 <= p['TDCcount'] < 15 ]

print(f"Last record ==> {table[-1]}")
print(f"Selected values ==> {e}")
print(f"Total selected records ==> {len(e)}")

# Finally, close the file (this also will flush all the remaining buffers!)
fileh.close()