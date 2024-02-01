
import numpy as np
import matplotlib.pyplot as plt


class DataCube:
    def __init__(self, data):
        self.data = data
        self.shape = self.data.shape
        self.memory_mb = self.data.nbytes / (1024 ** 2)  # Convert bytes to megabytes

    def __repr__(self):
        return (f"DataCube(shape={self.shape}, "
                f"total_values={np.prod(self.shape)}, "
                f"memory={self.memory_mb:.2f} MB)")

if __name__ == "__main__":
    data_array = np.random.rand(10, 10, 10)
    data_cube = DataCube(data_array)
    print(data_cube)
