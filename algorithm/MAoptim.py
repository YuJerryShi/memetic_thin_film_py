import numpy as np
import matplotlib as plt
import pandas as pd
import copy
from constants import *
from DispData import *
import matplotlib.pyplot as plt
from refl_disp import *
from scipy.optimize import minimize

"""
This class optimizes the spectrum of a multi-layer structure according to a target spectrum.
"""

#%% 
class MAoptim:
    def __init__(self, disp_data, list_material_names, material_in, material_sub, num_layers, num_population, r_target_input, \
                 theta_in=0, d_max=DEFAULT_D_MAX, max_iter=DEFAULT_MAX_ITER):
        """
        Constructor for generating a multi-layer optimization instance
        
        Parameters: 
            
            disp_data: an object from DispData containing the optimization wavelengths and materials of interest
            list_material_names: a list of material names used in the multi-layer
            material_in: material of incidence medium
            material_sub: material of substrate medium
            num_layers: number of layers
            num_population: size of the memetic algorithm population
            r_input_target: array that contains the target spectrum's information
                format: array([[wvlen_min1, wvlen_max1, R, weight], ...])
            theta_in: incidence angle
            d_max: maximum layer thickness
            max_iter: maximum number of memetic iterations
                
        Attributes: 
            disp_data: an object from DispData
            N: size of the memetic algorithm population
            K: number of layers
            d_max: maximum layer thickness
            
            list_material_names: a list of material names used in the multi-layer
            num_materials: number of materials used 
            list_materials: list_material_names represented by integers
            material_in: material of incidence medium
            material_sub: material of substrate medium
            n_in: refractive index incidence material
            n_in_fine: finely sampled refractive index incidence material
            n_sub: refractive index of substrate
            n_sub_fine: finely sampled refractive index of substrate
            theta_in: incidence angle
            
            r_target_input: array input of target reflection spectrum
                format: array([[wvlen_min1, wvlen_max1, R, weight], ...])
            merit_function: merit function 
            r_target: target reflection spectrum
            
            max_iter: maximum number of memetic iterations
            rate_top: reselection rate - top percentages to keep
            rate_good: reselection rate - percentages selected from the top half of population
            mutation_rate: mutation rate of population
            refine_period: number of iterations between population refinement
            refine_num: number of individuals refined
            
            fitness_generation: fitness of the current generation (size 2N)
            fitness_best: best fitness of the generation 
            fitness_avg: average fitness of the generation
            fitness_ind: sorted fitness indices according to structure
            fitness_best_so_far: best fitness of the algorithm so far
            
            n_material_best: materials of the best structure (size K)
            d_best: thicknesses of the best structure (size K)
            n_material_generation: materials of each generation (size 2N by K)
            d_generation: thicknesses of each generation (size 2N by K)
            n_material_parents: materials of parents (size N by K)
            d_parents: thicknesses of parents (size N by K)
            n_material_childern: materials of childern (size N by K)
            d_childern: thicknesses of childern (size N by K)
            
            optim_R_spectrum_fine: spectrum of the optimum structure
            
        """
        
        #%% Initialize a memetic algorithm's population
        self.disp_data = disp_data
        self.N = num_population # N = number of population size
        self.K = num_layers # K = number of layers
        self.list_material_names = list_material_names
        self.num_materials = len(self.list_material_names)
        self.list_materials = self.matname2list(self.list_material_names)
        self.d_max = d_max
        
        #%% Incidence and substrate materials 
        self.material_in = material_in
        self.material_sub = material_sub
        
        self.n_in, self.n_in_fine = self.material_to_ind(self.material_in)
        self.n_sub, self.n_sub_fine = self.material_to_ind(self.material_sub)
        
        self.theta_in = theta_in; 
        
        
        #%% Initialize the merit function
        self.r_target_input = r_target_input # [[wvlen_min1, wvlen_min2, R, weight], ...]
        self.merit_function, self.r_target = self.generate_mf()

        
        #%% Initialize the memetic operator values
        self.max_iter = max_iter
        self.rate_top = DEFAULT_RATE_TOP
        self.rate_good = DEFAULT_RATE_GOOD
        self.mutation_rate = DEFAULT_MUTATION_RATE
        self.refine_period = DEFAULT_REFINE_PERIOD
        self.refine_num = DEFAULT_REFINE_NUM
        
        
        #%% Initialize state variables
        self.fitness_generation = []
        self.fitness_best = np.zeros(self.max_iter)
        self.fitness_avg = np.zeros(self.max_iter)
        self.fitness_ind = []
        self.fitness_best_so_far = 1e9; 
        
        self.n_material_best = []
        self.d_best = []
        
        self.n_material_generation = []
        self.d_generation = []
        
        self.n_material_parents = []
        self.d_parents = []
        
        self.n_material_childern = []
        self.d_childern = []
        
        
        self.optim_R_spectrum_fine = np.zeros([disp_data.N_wvlen_fine])
        
        
    #%% Convert material names to integers
    def matname2list(self, list_material_names):
        """
        Converts material names to integer representations. 
        
        Parameters: 
            list_material_names: list of material names
        
        Output: 
            list_materials: list of integers, each representing a material
        
        """
        
        list_materials = []
        
        for i in range(self.num_materials):
            list_materials.append(self.disp_data.material_key.index(list_material_names[i]))
            
        
        return list_materials
    
    #%% Function that generates the population
    def generate_population(self):
        """
        Generates a population for memetic algorithm
        
        This method updates self.n_material_parents and self.d_parents 
        so that it produces N number of K-layer structures randomly
            
        """
        
        # num_population, num_layers, list_materials, d_max
        max_index = len(self.list_materials)
        temp_ordering = np.random.randint(0, max_index, [self.N, self.K])
        
        
        self.n_material_parents = np.zeros((self.N, self.K)).astype(int)
        for i in range(self.N):
            for j in range(self.K):
                self.n_material_parents[i, j] = self.list_materials[temp_ordering[i, j]]
        
        
        self.d_parents = self.d_max/2 * np.ones((self.N, self.K)) \
                      + self.d_max * (np.random.rand(self.N, self.K) - 0.5)
                    
        return
        
    #%% Function that performs crossover
    def crossover(self):
        """
        Performs crossover amongst parents
        
        This method randomly pairs up two parent structures, perform crossover 
        to produce two new childern structures. This doubles the size of the 
        individuals in the generation. 
            
        """
        
        # Initialize childern arrays
        self.n_material_childern = np.empty([self.N, self.K]).astype(int)
        self.d_childern = np.empty([self.N, self.K])
        
        # Shuffle the layers
        shuffle_index = np.random.permutation(self.N)
        
        self.n_material_parents = self.n_material_parents[shuffle_index, :]
        self.d_parents = self.d_parents[shuffle_index, :]
        
        # Perform crossover for each pair of parents
        for j in range(int(self.N/2)):
            n_material_parent1 = copy.deepcopy(self.n_material_parents[2*j, :])
            n_material_parent2 = copy.deepcopy(self.n_material_parents[2*j+1, :])
            
            d_parent1 = copy.deepcopy(self.d_parents[2*j, :])
            d_parent2 = copy.deepcopy(self.d_parents[2*j+1, :])
            
            # Choose a random crossover point
            cross_pt = np.random.randint(self.K-1)+1
            
            self.n_material_childern[2*j, :] = np.concatenate((n_material_parent1[0:cross_pt], n_material_parent2[cross_pt:self.K]), axis=0)
            self.n_material_childern[2*j+1, :] = np.concatenate((n_material_parent2[0:cross_pt], n_material_parent1[cross_pt:self.K]), axis=0)
            
            self.d_childern[2*j, :] = np.concatenate((d_parent1[0:cross_pt], d_parent2[cross_pt:self.K]), axis=0)
            self.d_childern[2*j+1, :] = np.concatenate((d_parent2[0:cross_pt], d_parent1[cross_pt:self.K]), axis=0)
            
        # Generation now double in size
        self.n_material_generation = np.concatenate((self.n_material_parents, self.n_material_childern), axis=0)
        self.d_generation = np.concatenate((self.d_parents, self.d_childern), axis=0)
        
        return
    
    #%%
    def mutation(self):
        """
        Performs mutations on a small number of individuals
        
        This method randomly selects individuals with probability mutation_rate 
        to perform mutation. When mutation happens to an individual, it changes 
        a random layer's material and thickness
            
        """
        
        for n in range(2*self.N):
            if np.random.rand() < self.mutation_rate:
                layer_change = np.random.randint(0, self.K)
                self.d_generation[n, layer_change] = self.d_max/2. + self.d_max * (np.random.rand() - 0.5)
                
                index_max = len(self.list_materials)
                temp_index = np.random.randint(0, index_max)
                self.n_material_generation[n, layer_change] = self.list_materials[temp_index]
        return
                
    #%% Evaluate the fitness of the population
    def eval_fitness(self):
        """
        Evaluates the fitness of the individuals in a generation
            
        """
        
        if self.fitness_generation == []:
            self.fitness_generation = np.zeros([2*self.N])
            
        for n in range(2*self.N):
            n_generation = self.disp_data.ref_ind[:, self.n_material_generation[n, :]]
            self.fitness_generation[n] = self.merit_function(n_generation, self.d_generation[n, :])
        
        return
    
    #%%
    def sort_population(self):
        """
        Sorts the individuals in a generation according to their merit function.
            
        """
        self.fitness_ind =sorted(range(2*self.N), key = lambda k : self.fitness_generation[k])
        self.fitness_ind = np.array(self.fitness_ind)
        
        # Arrange materials according to fitness_ind (input passed by reference)
        self.n_material_generation = self.n_material_generation[self.fitness_ind, :]
        self.d_generation = self.d_generation[self.fitness_ind, :]
        self.fitness_generation = self.fitness_generation[self.fitness_ind]
        
        return
    
    #%%
    def reselection(self):
        """
        Performs reselection on a generation's population
        
        This method selects the parent individuals for the next generation of 
        memetic optimization according to the reselection rule. 
        
        Reselection rule: 
            Keep top rate_top of the population as parents 
            Select the rest of the rate_good of the parents from the top-half of population
            Select the remaining rate_poor of the parents from the bottom half
            
        """
        # 
        N_top = round(self.N * self.rate_top)
        N_good = round(self.N * self.rate_good - self.rate_top)
        N_poor = self.N - N_top - N_good
        
        N_good_pool = self.N - N_top
        
        # Keep the N_top best structures
        selected_ind_top = np.array(range(N_top))
        
        # Select from the good structures
        temp_rand_index = N_top + np.random.permutation(N_good_pool)
#        temp_selected_good = temp_rand_index[0:N_good]
#        selected_ind_good = self.fitness_ind[temp_selected_good]
        selected_ind_good = temp_rand_index[0:N_good]
        
        # Select from the poor structures
        temp_rand_index = np.random.permutation(self.N) + self.N
#        selected_ind_poor = self.fitness_ind[temp_rand_index[0:N_poor]]
        selected_ind_poor = temp_rand_index[0:N_poor]
        
        # Concatenate the selected indices
        selected_inds = np.concatenate((selected_ind_top, selected_ind_good, selected_ind_poor)).astype(int)
                
#        print(selected_inds)
        self.n_material_parents = copy.deepcopy(self.n_material_generation[selected_inds, :])
        self.d_parents = copy.deepcopy(self.d_generation[selected_inds, :])
        
        return
    
    #%% Keep and display the best structure
    def keep_best(self, iter):
        """
        Keeps the best structures up to the current optimization step
            
        """
        
        self.fitness_best[iter] = self.fitness_generation[0]
        self.fitness_avg[iter] = np.mean(self.fitness_generation)
        
        if self.fitness_best[iter] < self.fitness_best_so_far:
            print('Found a temporary best structure in iteration ', iter)
            
            self.n_material_best = self.n_material_generation[0, :]
            self.d_best = self.d_generation[0, :]
            
            # Display the best structure so far and its fitness
            self.fitness_best_so_far = self.fitness_best[iter]
            print('Best fitness so far is ', self.fitness_best_so_far)
            self.print_struct(self.n_material_best, self.d_best)
            print('\n')
    
    #%% Perform local optimization
    def refinement(self, iter):
        """
        Performs refinement of a few structures
        
        This method performs a local optimization on the thicknesses of some 
        structures periodically. 
            
        """
        
        if np.mod(iter, self.refine_period) == self.refine_period-1:
            fitness_parents = np.zeros([self.N])
            # Resort fitness
            for n in range(self.N):
                n_parents = self.disp_data.ref_ind[:, self.n_material_parents[n, :]]
                fitness_parents[n] = self.merit_function(n_parents, self.d_parents[n, :])
        
            fitness_ind_parents = sorted(range(self.N), key = lambda k : fitness_parents[k])
            self.n_material_parents = self.n_material_parents[fitness_ind_parents, :]
            self.d_parents = self.d_parents[fitness_ind_parents, :]
            
            # Perform local optimization
            for m in range(self.refine_num):
                n_layers_m = self.disp_data.ref_ind[:, self.n_material_parents[m, :]]
                
                obj_fun = lambda x: self.merit_function(n_layers_m, x)
                
                x0 = copy.deepcopy(self.d_parents[m, :])
                
                x_opt = minimize(obj_fun, x0)
                # print(x_opt.x)
                
                self.d_parents[m, :] = copy.deepcopy(x_opt.x)
                print('Merit function of individual = ', obj_fun(x_opt.x))
        
    
    #%%  Print the material 
    def print_struct(self, n_index_layers, d_layers):
        """
        This method prints a structure's material composition and thicknesses.
        
        Parameters: 
            n_index_layers: an array of integers representing materials
            d_layers: thickness of each layer
            
        """
        
        struct_name = []  
        
        for i in range(self.K):
            struct_name.append(self.disp_data.material_key[n_index_layers[i]])
            
        d = {'material' : struct_name, 'thickness' : d_layers}
        df = pd.DataFrame(data=d)
        
        print(df)
    
    #%% Set optimization rates
    
    
    #%% Public function: optimize the population given 
    def ma_optimize(self):
        """
        Performs the memetic algorithm optimization
        
        This method calls all the memetic algorithm operators to perform 
        optimization and obtain the optimum structure
            
        """
        
        #%% Convergence curve
        fig_conv = plt.figure()
        ax_conv = fig_conv.add_subplot(111)
        
        for i in range(self.max_iter):
            if np.mod(i, 5) == 0:
                print('iter = ', i, '\n')
                
            # Single point crossover
            self.crossover()
            
            # Perform mutation
            self.mutation()
            
            # Evaluate fitness and sort the population
            self.eval_fitness()
            self.sort_population()
            
            # Keep best structure
            self.keep_best(i)
            
            # Perform reselection
            self.reselection()
            
            # Plot
            self.plot_convergence(i, fig_conv)
            
            # Refinement 
            self.refinement(i)
            
            
        return []
        
    #%% 
    def plot_spectrum(self): 
        """
        Plots the spectrum of the most optimum structure
            
        """
        
        
        n_material_final = self.n_material_best
        d_final = self.d_best
        
        print('The optimum structure found was: ')
        self.print_struct(n_material_final, d_final)
        
        n_final_fine = self.disp_data.ref_ind_fine[:, n_material_final]
        
        # Display the best structure (including the fixed materials above the substrate)
#        structure_best = convert2struct(material_key, n_material_final, d_final)
        
        # Compute the optimized structure's spectrum
        self.optim_R_spectrum_fine = refl_disp(self.n_in_fine, self.n_sub_fine, n_final_fine, d_final, self.disp_data.wvlen_vec_fine, self.theta_in)
        
        # Plot the optimized spectrum
        fig_spect = plt.figure()
        ax_spect = fig_spect.add_subplot(111)
        plt.plot(self.disp_data.wvlen_vec_fine, self.optim_R_spectrum_fine, 'b', label='Optimized spectrum')
        plt.plot(self.disp_data.wvlen_vec, self.r_target, 'g.', label='Target')
        
        plt.axis([min(self.disp_data.wvlen_vec_fine), max(self.disp_data.wvlen_vec_fine), -0.01, 1.01])
        plt.xlabel('wavelength ($\mu$m)')
        plt.ylabel('reflectivity')
        plt.legend()
        
        plt.show()
        
        
    
    
    #%%
    def plot_r_target(self):
        """
        Plot the target reflection spectrum
        
        """
        fig_tgt = plt.figure()
        ax_tgt = fig_tgt.add_subplot(111)
        plt.plot(self.disp_data.wvlen_vec, self.r_target, 'g.', label='Target')
        plt.axis([min(self.disp_data.wvlen_vec_fine), max(self.disp_data.wvlen_vec_fine), -0.01, 1.01])
        plt.xlabel('wavelength ($\mu$m)')
        plt.ylabel('reflectivity')
        plt.legend()
        
        plt.show()
        
    
    
    #%% Plot the convergence
    def plot_convergence(self, i, fig_conv):
        """
        Plots the convergence curve
        
        This method plots the convergence of the best fitness and average 
        fitness of the population over each generation. 
        
        Parameters: 
            i: current iteration
            fig_conv: figure that refers to the convergence plot
            
        """
        
        if i == 1:
            plt.semilogy([i-1, i], [self.fitness_best[i-1], self.fitness_best[i]], 'b', label='Best merit function')
            plt.semilogy([i-1, i], [self.fitness_avg[i-1], self.fitness_avg[i]], 'r--', label='Average merit function')
            plt.legend(loc='upper right')
            fig_conv.show()
            plt.pause(0.05)
        
        elif i > 1:
            plt.semilogy([i-1, i], [self.fitness_best[i-1], self.fitness_best[i]], 'b')
            plt.semilogy([i-1, i], [self.fitness_avg[i-1], self.fitness_avg[i]], 'r--')
            fig_conv.show()
            plt.pause(0.05)
            
        return
    
    #%% function
    def generate_mf(self):
        """
        Generates the merit function 
        
        This method generates the a function handle that represents the merit
        function of a given structure according to its material composition and 
        layer thicknesses. 
        
        Output: 
            mf: merit function handle
            r_target: target spectrum
            
        """
        
        r_target = np.zeros([self.disp_data.N_wvlen])
        weight = np.zeros([self.disp_data.N_wvlen])
                        
        if len(self.r_target_input.shape) == 1:
            wvlen_min_temp = self.r_target_input[0]
            wvlen_max_temp = self.r_target_input[1]
            r_temp = self.r_target_input[2]
            w_temp = self.r_target_input[3]
            
            indicator_interval = (self.disp_data.wvlen_vec >= wvlen_min_temp) & (self.disp_data.wvlen_vec < wvlen_max_temp)
            r_target = r_temp * indicator_interval
            weight = w_temp * indicator_interval
            
            
        else:
            N_intervals = len(self.r_target_input[:, 0])
            
            for i in range(N_intervals):
                # Parse input parameters 
                wvlen_min_temp = self.r_target_input[i, 0]
                wvlen_max_temp = self.r_target_input[i, 1]
                r_temp = self.r_target_input[i, 2]
                w_temp = self.r_target_input[i, 3]
                
                # Generate the target spectrum and weight function
                indicator_interval = (self.disp_data.wvlen_vec >= wvlen_min_temp) & (self.disp_data.wvlen_vec < wvlen_max_temp)
                r_target += r_temp * indicator_interval
                weight += w_temp * indicator_interval
            
            
                    
        mf = lambda n_layers, d: np.linalg.norm(weight * (refl_disp(self.n_in, self.n_sub, n_layers, abs(d), self.disp_data.wvlen_vec, self.theta_in) - r_target))**2
        
        return mf, r_target
    
    
    #%% This function converts a material name to its dispersive refractive indices
    def material_to_ind(self, material_input):
        """
        Converts a material's name to its refractive indices
        
        Parameters: 
            material_input: a single material input name
        
        Outputs: 
            ref_ind_output: refractive index of the material_input
            ref_ind_output_fine: finely sampled refractive index of the material input
            
        """
        
        if type(material_input) == str:
            material_index = self.disp_data.material_key.index(material_input)
            ref_ind_output = self.disp_data.ref_ind[:, material_index]
            ref_ind_output_fine = self.disp_data.ref_ind_fine[:, material_index]
            
        elif type(material_input) == float or type(material_input) == int:
            ref_ind_output = material_input 
            ref_ind_output_fine = material_input
            
        return ref_ind_output, ref_ind_output_fine
    
    #%% Set the memetic optimization parameters
    def set_optim_params(self, **kwargs):
        """
        Adjust the optimization parameter to users' input
        
        Parameters: 
            **kwargs: key item that contains {'optim_param', value}
            'optim_params' include:
                'max_iter'
                'rate_top'
                'rate_good'
                'mutation_rate'
                'refine_period'
                'refine_num'
            
        """
        
        for key, value in kwargs.items(): 
            if key == 'max_iter': 
                self.max_iter = value
            elif key == 'rate_top': 
                self.rate_top = value
            elif key == 'rate_good': 
                self.rate_good = value
            elif key == 'mutation_rate': 
                self.mutation_rate = value
            elif key == 'refine_period':
                self.refine_period = value
            elif key == 'refine_num':
                self.refine_num = value
            else: 
                raise ValueError('"%s" is not a valid optimization parameter' % key)
                
        
        