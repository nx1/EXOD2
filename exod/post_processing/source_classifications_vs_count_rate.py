from numpy import save
import pandas as pd

from exod.utils.path import savepaths_combined

import pandas as pd                                     
import numpy as np                                      
                                                        
# Generate a random dataset                             
np.random.seed(42)                                      
n_sources = 1000                                        
df = pd.DataFrame({                                     
    'counts': np.random.randint(0, 100, n_sources),     
    'class': np.random                                  
})                                                      
                                                        
# Bin the sources                                       
bins = np.linspace(0, 100, 10)                          
bin_sources(df, bins)                                   

