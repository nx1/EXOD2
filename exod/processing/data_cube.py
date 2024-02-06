from scipy.stats import binned_statistic_dd

from exod.utils.logger import logger

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


class DataCubeXMM(DataCube):
    def __init__(self, event_list, size_arcsec, time_interval):
        self.event_list = event_list
        self.size_arcsec   = size_arcsec
        self.time_interval = time_interval
        self.extent        = 51840 # Initial Cube Size
        self.pixel_size    = size_arcsec / 0.05
        self.n_pixels      = int(self.extent / self.pixel_size)
        self.bin_x = np.linspace(0, self.extent, self.n_pixels+1)
        self.bin_y = np.linspace(0, self.extent, self.n_pixels+1)
        self.bin_t = self.calc_time_bins()

        self.data = self.bin_event_list()
        self.bbox_img = self.get_cube_bbox()
        self.crop_data_cube()
        super().__init__(self.data)

    def calc_time_bins(self):
        t_lo = self.event_list.time_min
        t_hi = self.event_list.time_max
        t_i  = self.time_interval
        n_time_bins = int((t_hi - t_lo) / t_i)
        time_stop = t_lo + n_time_bins * t_i
        time_bins = np.arange(t_lo, time_stop + 1, t_i)
        return time_bins

    def bin_event_list(self):
        data = self.event_list.data
        sample = data['X'], data['Y'], data['TIME']
        bins = [self.bin_x, self.bin_y, self.bin_t]
        cube, bin_edges, bin_number = binned_statistic_dd(sample=sample,
                                                          values=None,
                                                          statistic='count',
                                                          bins=bins)
        return cube

    def crop_data_cube(self):
        """Crop the surrounding areas of the datacube that are empty."""
        bbox_img = self.bbox_img

        logger.info(f'Cropping data cube between bbox_img: {bbox_img}')
        self.data = self.data[bbox_img[0]:bbox_img[1], bbox_img[2]:bbox_img[3]]

        # Calculate the new bins
        self.bin_x = self.bin_x[bbox_img[0]:bbox_img[1]]
        self.bin_y = self.bin_y[bbox_img[2]:bbox_img[3]]

    def get_cube_bbox(self):
        """Get the Bounding Box corresponding to the cube's image plane."""
        idx_nonempty = np.where(np.sum(self.data, axis=2) > 0)
        bbox_img = (np.min(idx_nonempty[0]), np.max(idx_nonempty[0]) + 1,
                    np.min(idx_nonempty[1]), np.max(idx_nonempty[1]) + 1)
        return bbox_img

    @property
    def info(self):
        info = {'event_list'    : repr(self.event_list),
                'shape'         : self.shape,
                'size_arcsec'   : self.size_arcsec,
                'time_interval' : self.time_interval,
                'extent'        : self.extent,
                'pixel_size'    : self.pixel_size,
                'n_pixels'      : self.n_pixels,
                'bbox_img'      : self.bbox_img}
        for k, v in info.items():
            logger.info(f'{k:<13} : {v}')
        return info


if __name__ == "__main__":
    data_array = np.random.rand(10, 10, 10)
    data_cube = DataCube(data_array)
    data_cube.video()
    print(data_cube)

