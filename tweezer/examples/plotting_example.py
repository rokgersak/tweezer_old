import numpy as np
import tweezer.plotting as plt

# Example of unpacking the data
path = 'some_path.dat'
time, traps, trajectories = plt.read_file(path, 1)
data = trajectories[:, 0:2]
averaging_time = 0.1

# Example of using trajectory plot
plt.trajectory_plot(time, data, averaging_time)