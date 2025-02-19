"""
lab4.py

This script processes experimental data from a laser optics experiment. It reads data from a CSV file,
performs necessary data transformations, and fits the data to a Gaussian beam width model. The script
then plots the fitted data and saves the plots as images.

Author: Arnav Menon
Date: 2/19/2025
Dependencies: math, pandas, scipy, numpy, matplotlib

Usage:
    Ensure the CSV file 'EXP04_ALL.csv' is located in the specified directory.
    Run the script to generate and save plots of the Gaussian beam width fits.

Notes:
    - The laser wavelength is set to 633 nm.
    - Data is divided into 'near' and 'far' distance procedures for analysis.
    - The script assumes the CSV file is formatted correctly with columns 'save_dist' and 'save_xwidth'.

"""

# Importing dependencies
import math
import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# We need the laser's wavelength in meters
laser_wavelength = 633e-9

# All of the lab data taken is messily stored in EXP04.csv
# We need to slice the dataset according to the correct experiment sections
df = pd.read_csv('Lab 4/EXP04_ALL.csv')

# Grabbing the data for the small distance procedure
near_df = (df.iloc[25:56,:]).sort_values(by='save_dist') # sorted by increasing distance

# Grabbing the data for the far distance procedure
far_df = (df.iloc[58:,]).sort_values(by='save_dist') # sorted by increasing distance

# We expect a sqrt(1 + x**2) relationship in the data,
# so it is useful to put the minimum at x = 0
min_xwidth_near = near_df['save_xwidth'].min()
min_dist_near = near_df.loc[near_df['save_xwidth'] == min_xwidth_near, 'save_dist'].iloc[0]
near_df['save_dist'] -= min_dist_near # moving everything so that the minimum is at zero

min_xwidth_far = far_df['save_xwidth'].min()
min_dist_far = far_df.loc[far_df['save_xwidth'] == min_xwidth_far, 'save_dist'].iloc[0]
far_df['save_dist'] -= min_dist_far # moving everything so that the minimum is at zero

near_df['save_dist'] *= 1e-3 # converting to meters
near_df['save_xwidth'] *= 1e-6 # converting to meters

far_df['save_dist'] *= 1e-3 # converting to meters
far_df['save_xwidth'] *= 1e-6 # converting to meters

# Convert to numpy arrays
x_values_near = np.array(near_df['save_dist'])
y_values_near = np.array(near_df['save_xwidth'])

x_values_far = np.array(far_df['save_dist'])
y_values_far = np.array(far_df['save_xwidth'])

# Now we fit to the Gaussian beam width equation
# There's only one fit parameter, w0, so the initial guess is the minimum of the dataset
def beam_width(z, w0):
    zR = (np.pi * (w0**2)) / laser_wavelength
    return w0 * np.sqrt(1 + (z / zR)**2)

params, covariance = curve_fit(beam_width, x_values_near, y_values_near, p0 = [min_xwidth_near])
y_fit_near = beam_width(x_values_near, *params)

params, covariance = curve_fit(beam_width, x_values_far, y_values_far, p0 = [min_xwidth_far])
y_fit_far = beam_width(x_values_far, *params)

# Plotting and saving the near fit to near_fit.png
plt.figure(figsize=(10, 6))
plt.scatter(x_values_near * 1e3, y_values_near * 1e6, label='Data', color='blue') # redoing units to mm and um
plt.plot(x_values_near * 1e3, y_fit_near * 1e6, label='Fit', color='red') # redoing units to mm and um

# Display the near fit equation on the plot
fit_eq_near = f"Gaussian Fit: w(z) = {params[0]:.2e} * sqrt(1 + (z/zR)^2)"
plt.text(0.5, 0.95, fit_eq_near, transform=plt.gca().transAxes, fontsize=10, horizontalalignment = 'center',
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.xlabel('Distance (mm)')
plt.ylabel('Beam Width (um)')
plt.title('Gaussian Beam Width Fit Near the Laser')
plt.legend()
plt.grid(True)
plt.savefig('Lab 4/near_fit.png')

# Plotting and saving the far fit to far_fit.png
plt.figure(figsize=(10, 6))
plt.scatter(x_values_far * 1e3, y_values_far * 1e6, label='Data', color='blue') # redoing units to mm and um
plt.plot(x_values_far * 1e3, y_fit_far * 1e6, label='Fit', color='red') # redoing units to mm and um

# Display the far fit equation on the plot
fit_eq_far = f"Gaussian Fit: w(z) = {params[0]:.2e} * sqrt(1 + (z/zR)^2)"
plt.text(0.5, 0.95, fit_eq_far, transform=plt.gca().transAxes, fontsize=10, horizontalalignment = 'center',
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.xlabel('Distance (mm)')
plt.ylabel('Beam Width (um)')
plt.title('Gaussian Beam Width Fit Far from the Laser')
plt.legend()
plt.grid(True)
plt.savefig('Lab 4/far_fit.png')
