import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import poisson

# Function to update the histogram and display PMF
def update(frame):
    plt.clf()  # Clear the current figure
    data = np.random.poisson(lam=frame, size=1000)
    bins = np.arange(0, 20, 1)
    plt.hist(data, bins=bins, color='blue', edgecolor='black', density=True)
    plt.title(fr'Poisson Distribution ($\lambda$={frame:.1f}) k=1000 events')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')

    # Plot Poisson PMF as an orange dotted line
    x = np.arange(0, 20)
    pmf_values = poisson.pmf(x, frame)
    plt.plot(x, pmf_values, color='red', linestyle='--', linewidth=2)

    pmf_text = r'$P(X=k) = \frac{e^{-\lambda}\lambda^k}{k!}$'                                                  
    plt.text(0.4, 0.95, pmf_text, transform=ax.transAxes, fontsize=25, verticalalignment='top')                
                                                                                                               


    plt.xlim(0, 15)
    plt.ylim(0, 0.4)
    bin_edges = bins
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.xticks(bin_centers, [str(int(center)) for center in bin_centers])

# Set up the figure and axis
fig, ax = plt.subplots()

# Create the animation
lambda_values = np.concatenate([np.arange(0.1, 10, 0.1), np.arange(10, 0, -0.1)])
animation = FuncAnimation(fig, update, frames=lambda_values, interval=100)
animation.save('poission.gif')

plt.show()

