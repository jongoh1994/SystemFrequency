import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import unit
import datagen
import readgen
import time

# Generator Data File Names
fname_steam, fname_gas, fname_gen = 'data/IEEEG1_data.csv', 'data/GGOV1_data.csv', 'data/GEN_data.csv'

# Parameters
order, t_sim = 6, 10

# Get Matlab Output and Generator Information Dataframes
output_gen_info_org, output_gov_gasInfo_org, gen_data = readgen.get_gen_params(fname_steam, fname_gas, fname_gen)

fname_output = "data/OUTPUT"
rerun = True
if rerun:
    ramp, ramp_v, unittest_gas, unittest  = unit.get_matlab_output(fname_steam, fname_gas, order, t_sim, fname_output)
else:
    ramp, ramp_v, unittest_gas, unittest  = unit.read_matlab_output(fname_output)

print(unittest_gas.shape)