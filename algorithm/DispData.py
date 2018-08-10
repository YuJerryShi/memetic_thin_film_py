import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import *
from disp_interp import *

"""
This class generates the material dispersion data used for optimization and 
plotting.
""" 

class DispData:    
    def __init__(self, material_key, wvlen_sample, wvlen_fine):
        """
        Class that generates the sampled dispersion of given materials.
        
        Parameters: 
            
            material_key: list of material names
            wvlen_sample: sampled wavelengths for optimization
                format: array([[wvlen_min1, wvlen_max1, N_samples1], ...])
            wvlen_fine: finely sampled wavelengths for plotting
                format: array([wvlen_min, wvlen_max, N_fine])
        
        
        Attributes: 
            
            material_key: list of material names
            N_wvlen: the number of sampled wavelengths for optimization
            N_wvlen_fine: the number of finely sampled wavelengths for plotting
            wvlen_vec: array that contains the sampled wavelengths for optimization
            wvlen_vec_fine: array that contains finely sampled wavelength for plotting
            ref_ind: array that contains the material refractive indices at the sampled wavelengths 
                convention: column - wavelength; row - material
            ref_ind_fine: array that contains the material refractive indices at the finely sampled material refractive index
                convention: column - wavelength; row - material
                
        """
        
        #%% Store the list of material names
        self.material_key = material_key 
        
        #%% Initialize wavelength vectors
        if len(wvlen_sample.shape) > 1:
            self.N_wvlen = int(sum(wvlen_sample[:, 2]))
        else:
            self.N_wvlen = int(wvlen_sample[2])
        
        self.wvlen_vec = self.wvlen_gen(wvlen_sample)
        
        self.N_wvlen_fine = int(wvlen_fine[2])
        self.wvlen_vec_fine = self.wvlen_gen_fine(wvlen_fine)
        
        #%% Generate the dispersion for each material
        self.ref_ind, self.ref_ind_fine = self.disp_generate(material_key, self.wvlen_vec, self.wvlen_vec_fine)

    
    #%% Function that generates the sampled wavelengths 
    def wvlen_gen(self, wvlen_sample):
        """
        Generates the (coarsely sampled) wavelength vectors used in optimization. 
        
        This function generates an array of sampled wavelengths according to 
        wvlen_sample. 
        
        Parameters: 
            wvlen_sample: sampled wavelengths for optimization
                format: array([[wvlen_min1, wvlen_max1, N_samples1], ...])
        
        Output: 
            wvlen_vec: output wavelength vector sampled according to wvlen_sample
        
        """
        
        # Check if there are 1 or more intervals in wvlen_sample
        if len(wvlen_sample.shape) > 1:
            N_samples = len(wvlen_sample[:, 0])
        else:
            N_samples = 1
        
        # Initialize the output wvlen_vec
        wvlen_vec = np.array([])
        
        # Generate the wvlen_vec from different ranges of wvlen_sample
        if N_samples == 1:
            # Case with just one wavelength sampling region
            wvlen_low_temp = wvlen_sample[0]
            wvlen_high_temp = wvlen_sample[1]
            Nsample_temp = int(wvlen_sample[2])
                   
            wvlen_vec = np.linspace(wvlen_low_temp, wvlen_high_temp, Nsample_temp)
        else: 
            # Case with just multiple wavelength sampling regions
            for i in range(N_samples):
                wvlen_low_temp = wvlen_sample[i, 0]
                wvlen_high_temp = wvlen_sample[i, 1]
                Nsample_temp = int(wvlen_sample[i, 2])
                       
                wvlen_vec_temp = np.linspace(wvlen_low_temp, wvlen_high_temp, Nsample_temp)
                
                wvlen_vec = np.concatenate([wvlen_vec, wvlen_vec_temp])

        # Output wvlen_vec
        return wvlen_vec
    
    #%% Function that generates finely sampled wavelengths for plotting
    def wvlen_gen_fine(self, wvlen_fine):
        """
        Generates the finely sampled wavelength vectors for plotting. 
        
        Parameters: 
            wvlen_fine: finely sampled wavelength input
                format: array([wvlen_min, wvlen_max, N_fine])
        
        Output: 
            wvlen_vec_fine: output finely sampled wavelength vector
        
        """
        
        # Parse the minimum, maximum, and N_fine from wvlen_fine
        wvlen_min = wvlen_fine[0]
        wvlen_max = wvlen_fine[1]
        N_fine = int(wvlen_fine[2])
        
        # Calculate the min and max frequencies 
        omega_max = 2*np.pi*c0 / wvlen_min
        omega_min = 2*np.pi*c0 / wvlen_max
        
        # Sample evenly in frequency
        omega_vec_fine = np.linspace(omega_min, omega_max, N_fine)
        wvlen_vec_fine = 2*np.pi*c0 / omega_vec_fine
        
        # Reconstruct finely sampled wavelength 
        wvlen_vec_fine = np.flip(wvlen_vec_fine, 0)
        
        return wvlen_vec_fine
    
    #%% Function that generates the interpolated refractive index dispersion
    def disp_generate(self, material_key, wvlen_vec, wvlen_vec_fine):
        """
        Generates the material dispersion for materials in material_key
        
        Parameters: 
            material_key: list of material names
            wvlen_vec: sampled wavelength vector for optimization
            wvlen_vec_fine: finely sampled wavelength vector for plotting
            
        
        Output: 
            ref_ind: array that contains the material refractive indices at the sampled wavelengths 
                convention: column - wavelength; row - material
            ref_ind_fine: array that contains the material refractive indices at the finely sampled material refractive index
                convention: column - wavelength; row - material
        
        """
        
        # Initialize the number of materials and the refractive index arrays
        num_materials = len(material_key)
        ref_ind = np.zeros([self.N_wvlen, num_materials], dtype = complex)
        ref_ind_fine = np.zeros([self.N_wvlen_fine, num_materials], dtype = complex)
        
        #%% Read in the dispersion data of each material
        for i in range(num_materials):
            file_name = '../material_database/mat_' + material_key[i] + '.xlsx'
            
            try: 
                A = array(pd.read_excel(file_name))
            except NameError:
                print('The material database does not contain', material_key[i])
            
            ref_ind[:, i] = disp_interp(self.wvlen_vec, A)
            ref_ind_fine[:, i] = disp_interp(self.wvlen_vec_fine, A)
        
        return ref_ind, ref_ind_fine
    
    #%% Function that plots the refractive index for a given layer
    def disp_plot(self, material_name):
        """
        Plots the dispersion curve of a given material
        
        This function plots the real and imaginary parts of a material's 
        refractive index according to wavelength
        
        Parameters: 
            material_key: name of a given material
        
        """
        
        mat_index = self.material_key.index(material_name)
        
        fig1, ax1 = plt.subplots(); 
        plt.plot(self.wvlen_vec_fine, self.ref_ind_fine[:, mat_index].real, 'b', label='real(index)')
        plt.plot(self.wvlen_vec_fine, self.ref_ind_fine[:, mat_index].imag, 'r--', label='imag(index)')
        plt.legend(loc='upper right')
        plt.xlabel('wavelength ($\mu$m)')
        plt.ylabel('refractive index')
        
        plt.show()
    
            
        