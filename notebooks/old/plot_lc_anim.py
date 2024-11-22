import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# Function to update the plot
def update(frame):
    ax[0].clear()
    ax[1].clear()

    lc = lightcurve[:frame + 1]
    minimum = np.min(lc)
    maximum = np.max(lc)
    mean    = np.mean(lc)
    median  = np.median(lc)
    std     = np.std(lc)

    minimum_l.append(minimum)
    maximum_l.append(maximum)
    mean_l.append(mean)
    median_l.append(median)
    std_l.append(std)



    # Plot the lightcurve
    ax[0].plot(lc, label="Lightcurve", color="black", lw=1.0)

    # Plot horizontal lines for statistics
    ax[0].axhline(y=minimum, color="green", linestyle="--", label="Min", lw=1.0)
    ax[0].axhline(y=maximum, color="green", linestyle="--", label="Max", lw=1.0)
    ax[0].axhline(y=mean, color="orange", linestyle="--", label="Mean", lw=1.0)
    ax[0].axhline(y=median, color="purple", linestyle="--", label="Median", lw=1.0)
    ax[0].axhline(y=mean - std, color="red", linestyle="--", label="Mean - Std", lw=1.0)
    ax[0].axhline(y=mean + std, color="red", linestyle="--", label="Mean + Std", lw=1.0)

    ax[0].set_ylabel("Counts")
    ax[0].set_xlim(0,xlim)
    ax[0].set_ylim(0,h)

    # Display statistics
    ax[0].set_title(
        f"Min: {minimum:.2f} Max: {maximum:.2f} Mean: {mean:.2f} Median: {median:.2f} Std: {std:.2f}"
    )
    ax[0].legend(loc='lower right', bbox_to_anchor=(1.3, 0)
)

    ax[1].plot(minimum_l, color='green', label='Min', lw=1.0)
    ax[1].plot(maximum_l, color='green', label='Max', lw=1.0)
    ax[1].plot(mean_l, color='orange', label='Mean', lw=1.0)
    ax[1].plot(median_l, color='purple', label='Median', lw=1.0)
    ax[1].plot(std_l, color='red', label='Std', lw=1.0)
    ax[1].legend(loc='lower right', bbox_to_anchor=(1.25, 0))
    ax[1].set_xlabel("Time Window (50s)")
    ax[1].set_ylabel("Value")
    ax[1].set_xlim(0,xlim)
    plt.tight_layout()




    


minimum_l =[]
maximum_l =[]
mean_l =[]
median_l =[]
std_l =[]

df = pd.read_csv('/home/nkhan/EXOD2/data/results/0886121001/lcs.csv')
df = pd.read_csv('/home/nkhan/EXOD2/data/results/0101440401/lcs.csv') # src_0
df = pd.read_csv('/home/nkhan/EXOD2/data/results/0305570101/lcs.csv') # src_0
df = pd.read_csv('/home/nkhan/EXOD2/data/results/0831790701/lcs.csv') # src_0

df = df[~df['bti']]
lightcurve = df['src_0']
n = len(lightcurve)
xlim = n
h = max(lightcurve) + 5

# Set up the plot
fig, ax = plt.subplots(2,1, sharex=True, figsize=(7.5,5))

# Set the number of frames
num_frames = n - 1

# Create the animation
animation = FuncAnimation(fig, update, frames=num_frames, interval=50, repeat=False)
animation.save('QPE.gif')
# Show the plot
plt.show()

