from re import A
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from exod.utils.path import data_raw, data_processed

# Load Events list
obsid = '0001730201'
obsid = '0304080501'
obsid = '0510010701'
obsid = '0863401101'

event_file = data_raw / obsid / 'P0001730201PNU002PIEVLI0000.FTZ'
event_file = data_processed / obsid / 'PN_pattern_clean.fits'


instrument = fits.open(event_file)[0].header['INSTRUME'] # ['EMOS1', 'EMOS2', 'EPN']
tab        = Table.read(event_file, hdu=1)
df         = tab.to_pandas()
print(df)


# Filter only 1 CCD and set the start time to 0
df = df[df['CCDNR'] == 4]
df['TIME'] = df['TIME'] - df['TIME'].min()

# Only include columns we need
df = df[['TIME', 'RAWX', 'RAWY', 'PI']]
print(df)

t_bin_size = 1000
box_size   = 3
x_max = 69
y_max = 203

t_0, t_f   = df['TIME'].min(), df['TIME'].max()
t_bins     = t_bins = np.arange(t_0, t_f + t_bin_size, t_bin_size)
x_bins     = np.arange(1, x_max+box_size, box_size)
y_bins     = np.arange(1, y_max+box_size, box_size)

N_t_bins   = len(t_bins)
N_y_bins   = len(y_bins)
N_x_bins   = len(x_bins)

cube_size = N_t_bins * N_y_bins * N_x_bins
print(N_t_bins, N_y_bins, N_x_bins, cube_size)


# Group photons into time windows, and boxes
df['RAWX_GROUP'] = pd.cut(df['RAWX'], bins=x_bins)
df['RAWY_GROUP'] = pd.cut(df['RAWY'], bins=y_bins)
df['XY_BOX']     = df['RAWX_GROUP'].astype(str) + '_' + df['RAWY_GROUP'].astype(str)
df['TIME_BIN']   = pd.cut(df['TIME'], bins=t_bins, right=False)
print(df)

df = df[~df['RAWX_GROUP'].isna()]

# Count the number of photons in each time_window + bin combination
# Using observed=True  will only return those groups that had values
# Using observed=False will return all combinations even if they did not have counts
df_sub = df[['TIME_BIN','XY_BOX', 'PI']]
box_counts = df_sub.groupby(['TIME_BIN', 'XY_BOX'], observed=True).count().reset_index()
print(box_counts)


# Extract X, Y low and high values for each photon
box_counts_split = box_counts['XY_BOX'].str.extract(r'\((\d+), (\d+)\]\_\((\d+), (\d+)\]').astype(int)
box_counts_split.columns = ['X_LO', 'X_HI', 'Y_LO', 'Y_HI']
box_counts_split['VAL'] = box_counts['PI'] # Add column with number of detected photons
print(box_counts_split)


image_arrays = []
for time_bin in box_counts['TIME_BIN'].unique():
    image_size = (y_max, x_max)
    image_array = np.zeros(image_size, dtype=int)
    
    box_counts_time_bin = box_counts_split[box_counts['TIME_BIN'] == time_bin]
    for index, row in box_counts_time_bin.iterrows():
        image_array[row['Y_LO']:row['Y_HI'], row['X_LO']:row['X_HI']] = row['VAL']
    image_arrays.append(image_array)
    #plt.title(time_bin)
    #plt.imshow(image_array)
    #plt.show()
image_arrays = np.array(image_arrays)

print(f'Number of image frames={len(image_arrays)}')


c_max = np.max(image_arrays, axis=0)
c_median = np.median(image_arrays, axis=0)
c_min = np.min(image_arrays, axis=0)
c_median_nonzero = np.where(c_median == 0, 1, c_median)
V = np.maximum(c_max - c_median, c_median - c_min)

plt.imshow(V,  interpolation='none') # norm=LogNorm(),
plt.xlim(0,64)
plt.ylim(0,200)
plt.show()

plt.imshow(c_max, norm=LogNorm(), interpolation='none')
plt.show()
plt.imshow(c_median, norm=LogNorm(), interpolation='none')
plt.show()
plt.imshow(c_min, norm=LogNorm(), interpolation='none')
plt.show()
