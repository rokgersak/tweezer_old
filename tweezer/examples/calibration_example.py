import numpy as np
import tweezer.calibration as cal
import tweezer.examples.calibration_generate_data as gen
import matplotlib.pyplot as plt

# Parameters for drawing positions
k = (1.e-06, 2.e-6)
phi = 0.4
center = (6., 7.)

# Drawing positions
xdata, ydata = gen.generate(k, phi=phi, center=center)
time = gen.generate_time()

# Example of using calibration.calibrate
k_estimated, phi_estimated, center_estimated = cal.calibrate(time, xdata, ydata)
print(k_estimated, phi_estimated)

# Example of using calibration.plot
cal.plot(time, xdata, ydata)

# Example of using calibration.plot_drift
cal.plot_drift(time, xdata, ydata, averaging_time = 5)

#Example of using calibarion.potential
positions, values, phi = cal.potential(time, xdata, ydata)

#Example of using calibration.plot_potential
cal.plot_potential(time, xdata, ydata)