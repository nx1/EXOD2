import numpy as np
import pandas as pd
from skopt.space import Real, Integer
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_evaluations, plot_objective

def model(amplitude, time_binning):
    width = 0.1
    distance = amplitude - (3 * time_binning - 2)
    res = 1 / (1 + np.exp(-distance / width))
    return res

def objective_function(params):
    amplitude, time_binning = params
    detection_fraction = model(amplitude=amplitude, time_binning=time_binning)
    if 0 < detection_fraction < 1:
        score = abs(detection_fraction-0.5)
    else:
        score = 10
    return score

search_space = [
        Real(0.01, 30.0, name='Burst Count Rate (ct/s)'),
        Real(0.01, 30.0, name='Time Bin (s)'),
        ]

result = gp_minimize(
        func=objective_function,
        dimensions=search_space,
        n_calls=30,
        n_initial_points=4,
        acq_func='EI'
    )

plot_evaluations(result)
plot_convergence(result)
plot_objective(result)
plt.show()
