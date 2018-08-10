from numpy import pi, sqrt, array, zeros, sin, cos, tan, arcsin

#%%
def refl_disp_norm(n_in, n_out, n_disp, d, wvlen):
    # This program calculates the normal reflection spectrum of a multi-layer structure
    # Inputs: 
    # n_in - scalar/vector that specifies the incident refractive index
    # n_out - scalar/vector that specifies the output refractive index
    # n_disp - matrix that specifies the dispersive refractive indices of layers
    # d - vector that specifies the thickness of each layer
    # wvlen - vector that specfies the wavelengths of interest
    
    #%%
    K = len(d)
    K
    
    Z_out = 1 / n_out
    Z_in = 1 / n_in
    
    # Check if n_disp is dispersive
    flag_disp = len(n_disp.shape) # Returns 1 if not dispersive, 2 if dispersive
    
    #%% Iteratively use the impedance method
    Z_inter = Z_out
    
    for i in range(K):
        j = K - i - 1
        
        if flag_disp == 2:
            nj = n_disp[:, j]
        else:
            nj = n_disp[j]
        
        dj = d[j]
        
        Z_inter = 1 / nj * (Z_inter + 1j / nj * tan(2*pi*nj / wvlen * dj)) / (1/nj + 1j*Z_inter * tan(2*pi*nj/wvlen * dj))
    
    #%% 
    R = abs((Z_inter - Z_in) / (Z_inter + Z_in))**2
    return R
    
    
#%%
def refl_disp_TE(n_in, n_out, n_disp, d, wvlen, theta_in):
    # This program calculates the normal reflection spectrum of a multi-layer structure
    # Inputs: 
    # n_in - scalar/vector that specifies the incident refractive index
    # n_out - scalar/vector that specifies the output refractive index
    # n_disp - matrix that specifies the dispersive refractive indices of layers
    # d - vector that specifies the thickness of each layer
    # wvlen - vector that specfies the wavelengths of interest
    # theta_in - scalar angle of incidence in radians
    
    #%%
    K = max(len(d), len(d.transpose()))
    
    theta_out = (arcsin(n_in * sin(theta_in) / n_out)).transpose()
    
    Z_out = 1 / n_out / cos(theta_out)
    Z_in = 1 / n_in / cos(theta_in)
    
    # Check if n_disp is dispersive
    flag_disp = len(n_disp.shape) # Returns 1 if not dispersive, 2 if dispersive
    
    #%% Iteratively use the impedance method
    Z_inter = Z_out
    
    for i in range(K):
        j = K - i - 1
        
        if flag_disp == 2:
            nj = n_disp[:, j]
        else:
            nj = n_disp[j]
            
        dj = d[j]
        
        if i == 0 :
            theta_j = arcsin(n_out * sin(theta_out) / nj)
        else: 
            theta_j = arcsin(n_in * sin(theta_in) / nj)
        
        Z_inter = 1 / nj / cos(theta_j) * (Z_inter + 1j / nj / cos(theta_j) * tan(2*pi*nj / wvlen * cos(theta_j) * dj)) / (1/nj/cos(theta_j) + 1j*Z_inter * tan(2*pi*nj/wvlen * cos(theta_j) * dj))
    
    #%% 
    R_TE = abs((Z_inter - Z_in) / (Z_inter + Z_in))**2
    return R_TE
    
#%%
def refl_disp_TM(n_in, n_out, n_disp, d, wvlen, theta_in):
    # This program calculates the normal reflection spectrum of a multi-layer structure
    # Inputs: 
    # n_in - scalar/vector that specifies the incident refractive index
    # n_out - scalar/vector that specifies the output refractive index
    # n_disp - matrix that specifies the dispersive refractive indices of layers
    # d - vector that specifies the thickness of each layer
    # wvlen - vector that specfies the wavelengths of interest
    # theta_in - scalar angle of incidence in radians
    
    #%%
    K = max(len(d), len(d.transpose()))
    
    theta_out = (arcsin(n_in * sin(theta_in) / n_out)).transpose()
    
    Z_out = 1 / n_out * cos(theta_out)
    Z_in = 1 / n_in * cos(theta_in)
    
    # Check if n_disp is dispersive
    flag_disp = len(n_disp.shape) # Returns 1 if not dispersive, 2 if dispersive
    
    #%% Iteratively use the impedance method
    Z_inter = Z_out
    
    for i in range(K):
        j = K - i - 1
        
        if flag_disp == 2:
            nj = n_disp[:, j]
        else:
            nj = n_disp[j]
            
        dj = d[j]
        
        if i == 0 :
            theta_j = arcsin(n_out * sin(theta_out) / nj)
        else: 
            theta_j = arcsin(n_in * sin(theta_in) / nj)
        
        Z_inter = 1 / nj * cos(theta_j) * (Z_inter + 1j / nj * cos(theta_j) * tan(2*pi*nj / wvlen * cos(theta_j) * dj)) / (1/nj*cos(theta_j) + 1j*Z_inter * tan(2*pi*nj/wvlen * cos(theta_j) * dj))
    
    #%% 
    R_TM = abs((Z_inter - Z_in) / (Z_inter + Z_in))**2
    return R_TM
    
#%% 
def refl_disp(n_in, n_out, n_disp, d, wvlen, theta_in):
    if theta_in == 0:
        R = refl_disp_norm(n_in, n_out, n_disp, d, wvlen)
        
    else:
        R_TM = refl_disp_TM(n_in, n_out, n_disp, d, wvlen, theta_in)
        R_TE = refl_disp_TE(n_in, n_out, n_disp, d, wvlen, theta_in)
        
        R = 0.5 * (R_TM + R_TE)
        
    return R