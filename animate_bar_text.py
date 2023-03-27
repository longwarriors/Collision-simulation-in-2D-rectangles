import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Generate some random data
data = np.random.normal(0, 1, 1000)

# Create a figure and axis object
plt.style.use(["science", "notebook", "grid"])
fig, ax = plt.subplots()

# Set up the histogram
n_bins = 50
hist, bins = np.histogram(data, bins=n_bins)
width = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2

# Create the histogram plot
bar = ax.bar(center, hist, align='center', width=width, ec="yellow", fc="blue", alpha=0.5)

# Add the timer text
timer_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, ha='right', va='top')

# Define the animation function
def animate(i):
    # Generate new data
    data = np.random.normal(0, 1, 1000)
    
    # Update the histogram
    hist, bins = np.histogram(data, bins=n_bins)
    for rect, h in zip(bar, hist):
        rect.set_height(h)
    
    # Update the timer
    timer_text.set_text('Time: {:.2f}s'.format(i*0.1))
    
    return bar, timer_text

# Create the animation object
# blit must be False!
ani = animation.FuncAnimation(fig, animate, frames=100, interval=100, blit=False)

# Show the plot
plt.show()