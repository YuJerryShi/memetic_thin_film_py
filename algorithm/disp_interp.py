from numpy import pi, sqrt, array, zeros, sin, cos, tan, linspace, \
                  concatenate, flip, interp               
                  
#%%
def disp_interp(wvlen_in, data):
    N1 = len(wvlen_in)
    N2 = len(data[:, 0])
    
    wvlen_data = data[:, 0]
    n0 = data[:, 1]
    k0 = data[:, 2]
    
    n_interp = interp(wvlen_in, wvlen_data, n0)
    k_interp = interp(wvlen_in, wvlen_data, k0)
    
    ref_ind = n_interp - 1j * k_interp
    
    return ref_ind
    