import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt

from exod.utils.plotting import use_scienceplots
from exod.utils.path import data_plots, savepaths_util

if __name__ == "__main__":
    use_scienceplots()
    tab = Table.read(savepaths_util['4xmm_dr14_cat'])
    tab_var = tab[tab['SC_VAR_FLAG']]
    tab_var2 = tab[tab['VAR_FLAG']]


    plt.figure()
    plt.hist(np.log10(tab['EP_8_CTS']), bins=100, histtype='step', color='black', label=f'DR14 All ({len(tab):,})', density=True)
    plt.hist(np.log10(tab_var['EP_8_CTS']), bins=100, histtype='step', label=f'DR14 Src Var ({len(tab_var):,})', density=True)
    plt.hist(np.log10(tab_var2['EP_8_CTS']), bins=100, histtype='step', label=f'DR14 Det Var ({len(tab_var2):,})', density=True)
    plt.xlabel(r'$log_{10}$(0.2 - 12.0 keV Counts)')
    plt.ylabel('Normalized Counts')
    plt.legend(loc='upper right')
    plt.savefig(data_plots / 'dr14_cts_hist.png')
    plt.savefig(data_plots / 'dr14_cts_hist.pdf')
    plt.show()
