import sys
sys.path.append('../algorithm')
import numpy as np
from DispData import *
from MAoptim import *

#%% 
# Materials used (material data must be contained in ../material_database)
materials = ['Al2O3', 'MgF2', 'TiO2', 'SiC', 'SiN', 'SiO2', 'HfO2', 'Ag']

# Wavelength sampled. Format: array([[wvlen_min1, wvlen_max1, N_samples1], ...])
wvlen_samples = np.array([[0.35, 1.8, 100], [6.00, 20.0, 150]])

# Finely sampled wavelength for plotting. Format: array([wvlen_min, wvlen_max, N_fine])
wvlen_fine = np.array([0.3, 20, 2000])

# Generate dispersion data for materials 
disp_data = DispData(materials, wvlen_samples, wvlen_fine)

#%%
# (Optional): plot the dispersion data for a given material
disp_data.disp_plot('SiO2')

#%% Define optimization targets

# Target reflection spectrum input array
# Format: array([[wvlen_min1, wvlen_max1, refl1, weight1], ...])
r_target_input = np.array([[0.3, 1.8, 1, 5], [8, 13, 0, 1], [13, 25, 1, 1]])
    # Reflects 0.3-1.8 and 13-25 microns. Antireflects 8-13 microns

# List of the materials used
list_materials = ['Al2O3', 'MgF2', 'TiO2', 'SiC', 'SiN', 'SiO2', 'HfO2']

material_in = 1 # Incident material. Can either be a material name or a scalar
material_sub = 'Ag' # Substrate material. Can either be a material name or a scalar
num_population = 1000 # Number of population in the memetic algorithm
num_layers = 6 # Number of material layers

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

