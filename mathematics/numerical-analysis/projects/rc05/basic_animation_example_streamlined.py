"""
=========================
Simple animation example
=========================

Animation of a random walk plot.

Adapted from a simple example from matplotlib documentation,
with input from stack overflow 
https://stackoverflow.com/questions/25333732/matplotlib-animation-not-working-in-ipython-notebook-blank-plot

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


###############################################################################

# Create data with points to show in each frame
# Fixing random state for reproducibility
# np.random.seed(19680801)
data = np.random.rand(2, 25)
# Alternate list of data is a parabola
# data = np.array([np.linspace(0,1,10),np.linspace(0,1,10)**2])

# Create the figure, axis, and plot
fig, ax = plt.subplots()
l, = ax.plot([], [], 'r:o')
# adjust axis parameters to make it "pretty"
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('test')
plt.grid()

# Define update function as a lambda function (idea from stack overflow)
update = lambda fnum: l.set_data(data[..., :fnum])

# run a frame for every column
r,c = np.shape(data)

# Set up the animation
ani = animation.FuncAnimation(fig, update, frames=c, interval=1000)

plt.show()