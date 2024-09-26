from scipy.stats import binned_statistic_dd

from exod.utils.path import data_util
from exod.processing.data_cube import DataCube

import numpy as np
from astropy.table import Table


tab_wfi  = Table.read(data_util / 'evt_wfi.fits')
tab_xifu = Table.read(data_util / 'evt_xifu.fits')

tab = tab_wfi
x = 'RAWX'
y = 'RAWY'


print(tab[x].min(), tab[x].max())
print(tab[y].min(), tab[y].max())
print(tab['TIME'].min(), tab['TIME'].max())

bin_x = np.linspace(tab[x].min(), tab[y].max(), num=300)
bin_y = np.linspace(tab[x].min(), tab[y].max(), num=300)
bin_t = np.arange(tab['TIME'].min(), tab['TIME'].max(), 5)
bins = [bin_x, bin_y, bin_t]
sample = [tab[x], tab[y], tab['TIME']]
cube, bin_edges, bin_number = binned_statistic_dd(sample=sample, values=None, statistic='count', bins=bins)

dc = DataCube(cube)
dc.video()
