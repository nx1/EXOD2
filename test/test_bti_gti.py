from exod.processing.bti import get_bti, get_bti_bin_idx, plot_bti
from numpy import sin, cos, exp
import numpy as np

def test_gti_bti():
    time  = np.arange(0,10000,100) 
    time2 = np.arange(0,10000,33.33) # Different time sampling
   
    noise = 0.1*np.random.random(size=len(time)) 

    lcs = {
    'starts_above'          : cos((2*3.14 / 5000)*time)* exp(-time/4000) + 1 + noise,
    'ends_above'            : cos((2*3.14 / 7000)*time-600) + 1 + noise,
    'starts_and_ends_above' : cos((2*3.14 / 5000)*time) + 1 + noise,
    'starts_and_ends_below' : sin((2*3.14 / 5000)*time) + 1 + noise,
    'high_freq'             : sin((2*3.14 / 500)*time) + 1 + noise,
    'all_below'             : 0.3*cos((2*3.14 / 5000)) + 1 + noise,
    'all_above'             : cos((2*3.14 / 5000)*time-200) + 5 + noise,
    }
       
    threshold = 1.5
    
    for key, data in lcs.items():
        threshold = 1.5
        bti = get_bti(time, data, threshold)
        rejected_idx = get_bti_bin_idx(bti, time2)
        plot_bti(time, data, threshold, bti)
