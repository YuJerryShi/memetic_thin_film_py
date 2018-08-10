import sys
sys.path.append('../algorithm')
import numpy as np
from DispData import *
from MAoptim import *

#%% 
# Materials used (material data must be contained in ../material_database)
materials = ['Al2O3', 'MgF2', 'TiO2', 'SiC', 'SiN', 'SiO2', 'HfO2', 'Ag']

# Wavelength sampled. Format: array([[wvlen_min1, wvlen_max1, N_samples1], ...])
wvlen_samples = np.array([0.5, 1, 100])

# Finely sampled wavelength for plotting. Format: array([wvlen_min, wvlen_max, N_fine])
wvlen_fine = np.array([0.3, 1.5, 1000])

# Generate dispersion data for materials 
disp_data = DispData(materials, wvlen_samples, wvlen_fine)

#%%
# (Optional): plot the dispersion data for a given material
disp_data.disp_plot('SiO2')

#%% Define optimization targets

# Target reflection spectrum input array
# Format: array([[wvlen_min1, wvlen_max1, refl1, weight1], ...])
r_target_input = np.array([0.5, 1, 1, 1]) # Reflect from 0.5 to 1 micron

# List of the materials used
list_materials = ['Al2O3', 'MgF2', 'TiO2', 'SiC', 'SiN', 'SiO2', 'HfO2']

material_in = 1 # Incident material. Can either be a material name or a scalar
material_sub = 1 # Substrate material. Can either be a material name or a scalar
num_population = 500 # Number of population in the memetic algorithm
num_layers = 10 # Number of material layers

#%% Set up the solver
# Instantiate the memetic algorithm object
ma = MAoptim(disp_data, list_materials, material_in, material_sub, num_layers, num_population, r_target_input)

# Generate the population
ma.generate_population() 

# OPTIONAL memetic algorithm parameters
optim_params = {'max_iter':60, 'mutation_rate':0.05}
ma.set_optim_params(**optim_params)

#%% Perform the memetic algorithm optimization
ma.ma_optimize()

#%% Display and plot the best structure
ma.plot_spectrum()

