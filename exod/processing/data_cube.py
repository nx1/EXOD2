
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm


class DataCube:
    def __init__(self, data):
        self.data = data
        self.shape = self.data.shape
        self.memory_mb = self.data.nbytes / (1024 ** 2)  # Convert bytes to megabytes

    def __repr__(self):
        return (f"DataCube(shape={self.shape}, "
                f"total_values={np.prod(self.shape)}, "
                f"memory={self.memory_mb:.2f} MB)")

    def video(self):
        fig, ax = plt.subplots()
        img = ax.imshow(self.data[:, :, 0].T, cmap='hot', animated=True, interpolation='none',
                        origin='lower') # norm=LogNorm())
        colorbar = fig.colorbar(img, ax=ax)

        def update(frame):
            ax.set_title(f'{self}\n{frame}/{num_frames}')
            img.set_array(self.data[:, :, frame])
            return img,
    
        num_frames = self.shape[2]
        ani = FuncAnimation(fig, update, frames=num_frames, interval=10)
        plt.show()

if __name__ == "__main__":
    data_array = np.random.rand(10, 10, 10)
    data_cube = DataCube(data_array)
    data_cube.video()
    print(data_cube)
