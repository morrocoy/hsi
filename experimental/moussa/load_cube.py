import numpy as np

dim=3
dtype=np.float32
size = np.dtype(dtype).itemsize

def load_cube(image_file_path):
    with open(image_file_path, 'rb') as file:
        dtypeHeader = np.dtype(np.int32)
        dtypeHeader = dtypeHeader.newbyteorder('>')
        buffer = file.read(size * dim)
        header = np.frombuffer(buffer, dtype=dtypeHeader)

        dtypeData = np.dtype(dtype)
        dtypeData = dtypeData.newbyteorder('>')
        buffer = file.read()
        cubeData = np.frombuffer(buffer, dtype=dtypeData)

    cubeData = cubeData.reshape(header, order='C')
    cubeData = np.rot90(cubeData)

    return cubeData