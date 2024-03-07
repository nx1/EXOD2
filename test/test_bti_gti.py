from exod.pre_processing.bti import get_bti, get_bti_bin_idx, plot_bti
import numpy as np

def test_gti_bti():
    time  = np.arange(0,10000,100)
    time2 = np.arange(0,10000,33.33)
    
    starts_above          = np.cos((2*3.14 / 5000)*time)* np.exp(-time/4000) + 1 + 0.1*np.random.random(size=len(time)) 
    ends_above2           = np.flip(starts_above)
    ends_above            = np.cos((2*3.14 / 7000)*time-600) + 1 + 0.1*np.random.random(size=len(time))
    starts_above2         = np.flip(ends_above)
    starts_and_ends_above = np.cos((2*3.14 / 5000)*time) + 1 + 0.1*np.random.random(size=len(time))
    starts_and_ends_below = np.sin((2*3.14 / 5000)*time) + 1 + 0.1*np.random.random(size=len(time))
    high_freq             = np.sin((2*3.14 / 500)*time) + 1 + 0.1*np.random.random(size=len(time))
    all_below             = 0.3*np.cos((2*3.14 / 5000)) + 1 + 0.1*np.random.random(size=len(time))
    all_above             = np.cos((2*3.14 / 5000)*time-200) + 5 + 0.1*np.random.random(size=len(time))
    
    test_cases = [starts_above, ends_above2, ends_above, starts_above2, starts_and_ends_above, starts_and_ends_below, high_freq, all_below, all_above]
    
    threshold = 1.5
    
    for data in test_cases:
        threshold = 1.5
        bti = get_bti(time, data, threshold)
        rejected_idx = get_bti_bin_idx(bti, time2)
        plot_bti(time, data, threshold, bti)
