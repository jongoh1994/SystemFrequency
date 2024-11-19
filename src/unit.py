'''
Use this code to execute matlab code for the unit test data.
'''

import matlab.engine  
import pandas as pd
import numpy as np

import h5py

def write_3d_array_to_hdf5(array, filename):
    """Write a 3D NumPy array to an HDF5 file."""
    with h5py.File(filename, 'w') as f:
        f.create_dataset("array", data=array)

def read_3d_array_from_hdf5(filename):
    """Read a 3D NumPy array from an HDF5 file."""
    with h5py.File(filename, 'r') as f:
        array = f["array"][:]
    return array

def get_matlab_output(fname_steam, fname_gas, order, t_sim, fname_results="data/OUTPUT"):
    '''Then run this with py 3.10 for matlab tools'''

    # Define simulation parameters
    eng = matlab.engine.start_matlab()

    # Call the MATLAB function
    output = eng.Call4Simulink_test(fname_steam, fname_gas, order, t_sim)

    # Write incase we want to save
    for field in output:
        dat = np.array(output[field]) # pd.DataFrame(output[field])
        fname = f'{fname_results}_{field}.csv'
        #dat.to_csv(fname, index=False)
        write_3d_array_to_hdf5(dat, fname)

        yield dat

    # Close the MATLAB engine
    eng.quit()

def read_matlab_output(fname="data/OUTPUT"):
    '''Then run this with py 3.10 for matlab tools'''

    fields = ['ramp', 'ramp_v', 'unitTestLoadlimit_ggov1', 'unitTest']

    # Write incase we want to save
    for field in fields:
        yield read_3d_array_from_hdf5(f'{fname}_{field}.csv')
        #yield pd.read_csv(f'{fname}_{field}.csv').T.reset_index(drop=True).T



