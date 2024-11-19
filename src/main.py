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
rerun = False
if rerun:
    ramp, ramp_v, unittest_gas, unittest  = unit.get_matlab_output(fname_steam, fname_gas, order, t_sim, fname_output)
else:
    ramp, ramp_v, unittest_gas, unittest  = unit.read_matlab_output(fname_output)


# Start Timer
start_time = time.time()

# Generator information
event_gen=pd.DataFrame({
    'BusNum' : [7098, 7099],
    'GenID' : [1, 1]
})

output_gen_info=output_gen_info_org.copy()
output_gov_gasInfo=output_gov_gasInfo_org.copy()

output_gen_info['TSVmax']=0.87
output_gov_gasInfo['TSLdref']=0.835
output2=datagen.DataGenerate(gen_data, output_gen_info, output_gov_gasInfo, event_gen)

gen_data_active=output2[0]
gov_gasInfo = output2[1]


numGen=gen_data_active.shape[0]
P_event=2558/sum(gen_data_active['GenMVABase'])
Hsys=sum(gen_data_active['TSH']*100)/sum(gen_data_active['GenMVABase'])

## Coefficient for Polynomial Fitting
# Lambda matrix for integral of ramp polynomials

# Initialize variables
coeffi = 1 / (np.arange(1, order + 2))  # Coefficients for lambda matrix
lamda = np.eye(order + 1) - np.diag(coeffi)

# Proportional coefficient for governor output (based on machine size)
alpha = gen_data_active['GenMVABase'] / sum(gen_data_active['GenMVABase'])

# Initialization
mask_index_nolimit = pd.Series(data=np.ones(numGen, dtype=bool), index=gen_data_active.index)
index_limit = []
index_limit_valve = []
index_limit_load = []
index_limit_valveAfterLoad = []

# Polynomial fitting of ramp response within the fitting time window
t_fit = 5
time10 = np.linspace(0, t_sim, 1000)

# Time window for fitting
time_fit = time10[time10 <= t_fit]
time_fit_len = len(time_fit)

# Filter the columns in output_unittest['ramp_v'] that match the 'Index' values from gen_data_active
ramp = pd.DataFrame(ramp)
ramp_v = pd.DataFrame(ramp_v)
ramp = ramp.loc[:,gen_data_active.index]
ramp_v = ramp_v.loc[:,gen_data_active.index]


rampResponse_fit = ramp.iloc[0:time_fit_len, :]
v_rampResponse_fit = ramp_v.iloc[0:time_fit_len, :]

# Initialize lists to hold polynomial coefficients
p_gov_total = []
p_valve_total = []

# Polynomial fitting for each generator
for i in range(numGen):
    # Fit polynomials & reverse coefficient order for later evaluation
    p_gov_total.append(np.polyfit(time_fit,rampResponse_fit.iloc[:, i], order)[::-1])  # Flipping the polynomial coefficients
    p_valve_total.append(np.polyfit(time_fit,  v_rampResponse_fit.iloc[:, i], order)[::-1])  # Flipping the polynomial coefficients

# Convert the lists to numpy arrays (optional)
p_gov_total = pd.DataFrame(p_gov_total, index=gen_data_active.index)
p_valve_total = pd.DataFrame(p_valve_total, index=gen_data_active.index)

# Frequency Nadir Calculation - No Limit Violation
t_fit_step = 1
Pm_limit = np.zeros(time_fit_len)

# Initialize arrays with zeros
limitCheck_old = pd.Series(data=np.zeros(numGen, dtype=bool), index=gen_data_active.index)
valve_limitCheck_old = pd.Series(data=np.zeros(numGen, dtype=bool), index=gen_data_active.index)
load_limitCheck_old = pd.Series(data=np.zeros(numGen, dtype=bool), index=gen_data_active.index)
limitCheck_old_valveAfterLoad = pd.Series(data=np.zeros(numGen, dtype=bool), index=gen_data_active.index)

# Filter time_fit based on the condition (time10 <= t_fit)
limit_start = 1

from scipy.linalg import pinv
from scipy.special import factorial

# Calculate nadir time by checking error
Pout_factor = P_event / (2 * Hsys)
t_powers = np.power(time_fit[:, None], np.arange(order + 1))
t_powers_1 = np.power(time_fit[:, None], np.arange(1, order + 2))
t_powers_2 = np.power(time_fit[:, None], np.arange(order + 2))

zeros_column = pd.DataFrame(np.zeros((p_valve_total.shape[0], 1)), index=gen_data_active.index)

# Insert zero-filled columns at the beginning and end
ramp_p_integ = pd.concat([zeros_column, p_valve_total], axis=1)
ramp_p = pd.concat([p_valve_total, zeros_column], axis=1)

ramp_p_integ.columns = range(ramp_p_integ.shape[1])
ramp_p.columns = range(ramp_p.shape[1]) 

diagonal_matrix = np.zeros((order+2, order+2))
np.fill_diagonal(diagonal_matrix[1:, 1:], coeffi)

# Define variables accordingly
# Initialize required arrays, variables, and constants here

#unittest_gas = unittest_gas.to_numpy() # np.array(output_unittest['unitTestLoadlimit_ggov1'])
#unittest = unittest.to_numpy() # np.array(output_unittest['unitTest'])

# Precompute limit values for faster access
lim_u = gen_data_active['valve_lim_u']

const_Lref = gov_gasInfo['TSLdref'] / gov_gasInfo['TSKturb'] + gov_gasInfo['TSWfnl']
const_v_0 =  gov_gasInfo['GenMW'] / gov_gasInfo['GenMVABase'] / gov_gasInfo['TSKturb'] + gov_gasInfo['TSWfnl']
inner_limit = 1 - const_v_0
temp = (const_Lref - const_v_0) * gov_gasInfo['TSKpload']

numSteam=len(output_gen_info[output_gen_info['Type'] == 'steam'])

temp_matrix1=ramp_p_integ @ diagonal_matrix
temp_matrix2=t_powers_1 @ np.diag(coeffi) @ p_valve_total.T
temp_matrix3=t_powers @ p_valve_total.T

while True:
    
    # Frequency nadir calculation
    while True:
        
        p_gov_part = p_gov_total.loc[mask_index_nolimit]
        p_valve_part = p_valve_total.loc[mask_index_nolimit]
                  
        # Multiply each row of `p_gov_part_filtered` by the corresponding element in `alpha_filtered`
        p_gov_sys = (p_gov_part.T * alpha[mask_index_nolimit]).T.sum()

        # NOTE this can be simplified dramatically
        # Find Frequency Nadir Time by Minimizing Error
        error_save = 100 * np.ones((time_fit_len, 2))
        error_save[:, 0] = time_fit

        if not index_limit:
            error_save[:, 1] = np.abs(t_powers @ lamda @ p_gov_sys - 2 * Hsys)
        else:
            error_save[:, 1] = np.abs(Pout_factor * t_powers @ lamda @ p_gov_sys + Pm_limit - P_event)

        num = np.argmin(error_save[:, 1])
        t_n_star = error_save[num, 0]

        if t_fit > t_n_star:
            break
        else:
            t_fit += t_fit_step
            time_fit = time10[time10 <= t_fit]
            time_fit_len = len(time_fit)
            
            t_powers = np.power(time_fit[:, None], np.arange(order + 1))
            t_powers_1 = np.power(time_fit[:, None], np.arange(1, order + 2))
            t_powers_2 = np.power(time_fit[:, None], np.arange(order + 2))
            
            rampResponse_fit = ramp.iloc[0:time_fit_len, :]
            v_rampResponse_fit = ramp_v.iloc[0:time_fit_len, :]

            p_gov_total = []
            p_valve_total = []
            
            # Polynomial fitting for each generator
            for i in range(numGen):
                # Fit polynomials & reverse coefficient order for later evaluation
                p_gov_total.append(np.polyfit(time_fit,rampResponse_fit.iloc[:, i], order)[::-1])  # Flipping the polynomial coefficients
                p_valve_total.append(np.polyfit(time_fit,  v_rampResponse_fit.iloc[:, i], order)[::-1])  # Flipping the polynomial coefficients
            
            # Convert the lists to numpy arrays (optional)
            p_gov_total = pd.DataFrame(p_gov_total, index=gen_data_active.index)
            p_valve_total = pd.DataFrame(p_valve_total, index=gen_data_active.index)
            
            # NOTE make these *empty* sets not zero.
            zeros_column = pd.DataFrame(np.zeros((p_valve_total.shape[0], 1)), index=gen_data_active.index)
            
            # Insert zero-filled columns at the beginning and end
            ramp_p_integ = pd.concat([zeros_column, p_valve_total], axis=1)
            ramp_p = pd.concat([p_valve_total, zeros_column], axis=1)
            
            ramp_p_integ.columns = range(ramp_p_integ.shape[1])
            ramp_p.columns = range(ramp_p.shape[1]) 
            
            diagonal_matrix = np.zeros((order+2, order+2))
            np.fill_diagonal(diagonal_matrix[1:, 1:], coeffi)
            
            temp_matrix1=ramp_p_integ @ diagonal_matrix
            temp_matrix2=t_powers_1 @ np.diag(coeffi) @ p_valve_total.T
            temp_matrix3=t_powers @ p_valve_total.T

    a = t_powers_1 @ np.diag(coeffi)

    # Valve coefficient approximation   
    v_coeffifient_total = -P_event / (2 * Hsys) * (temp_matrix1 / t_n_star - ramp_p)
    v_cal = -P_event / (2 * Hsys) * (1 / t_n_star * (temp_matrix2) - temp_matrix3)
    a = -P_event / (2 * Hsys) * (1 / 2.5125 * (temp_matrix2) - temp_matrix3)

    # Load limit response at no limit governors
    nolimit_index_gas = (gen_data_active.index[
        (gen_data_active['TSGenGovernorName'] == 'GGOV1') & (mask_index_nolimit)]).tolist()
    
    v_coefficient_limit = v_coeffifient_total.loc[nolimit_index_gas]

    loadlimiter_output_initial = pd.DataFrame(np.zeros((time_fit_len, len(nolimit_index_gas)))
        , columns=nolimit_index_gas)

    loadlimiter_results = np.zeros((time_fit_len, len(nolimit_index_gas)))
    
    '''
    3 Dimensional Array
    '''
    for idx, i in enumerate(nolimit_index_gas):
        # Extract the correct slice from unittest_gas
        slice_gas = unittest_gas[:time_fit_len, :, i - numSteam]
        
        # Perform matrix multiplication and add temp[i]
        # loadlimiter_output_initial[i]  = slice_gas @ v_coefficient_limit.loc[i, :] + temp[i]
        
        result=slice_gas @ v_coefficient_limit.loc[i].values + temp[i]
        loadlimiter_results[:, idx]  = result
        
    loadlimiter_output_initial.loc[:, nolimit_index_gas]=loadlimiter_results
    
    # Initialize the limit check DataFrames with boolean values for faster operations  
    valve_limitCheck = pd.Series(data=np.zeros(numGen, dtype=bool), index=gen_data_active.index)
    load_limitCheck = pd.Series(data=np.zeros(numGen, dtype=bool), index=gen_data_active.index)
      
    # Create boolean masks for `index_nolimit` and `nolimit_index_gas` to apply conditions across all time steps
    mask_nolimit_index_gas1 = pd.Series(gen_data_active.index.isin(nolimit_index_gas), index=gen_data_active.index)
    
    
    column = mask_index_nolimit.index[mask_index_nolimit]
    
    # Creating an empty array with shape (len_time_fit, len_column)
    test_lim_u=pd.DataFrame(np.zeros((time_fit_len, len(column))), columns=column)
    test_lim_u.loc[:, column]=lim_u[column].values
    
    mask_valve_limit = (v_cal.loc[:, mask_index_nolimit] >= test_lim_u)
    first_true_row = mask_valve_limit.idxmax()
    non_zero_values_valv = first_true_row[first_true_row != 0]

    ''' Valve Limit Check'''
    if not non_zero_values_valv.empty:
        min_time_valv = non_zero_values_valv.min()
        min_index_valv = non_zero_values_valv.idxmin()
        
    else:
        min_time_valv=time_fit_len-1
        min_index_valv=99999
        
    
    ''' Load Limit Check - 3D Array Used Here'''

    column = mask_nolimit_index_gas1.index[mask_nolimit_index_gas1]
    
    # Creating an empty array with shape (len_time_fit, len_column)
    mask_load_limit = (v_cal.loc[:, column] >= loadlimiter_output_initial)
    first_true_row = mask_load_limit.idxmax()
    non_zero_values_load = first_true_row[first_true_row != 0]
    
    if not non_zero_values_load.empty:   
        min_time_load = non_zero_values_load.min()
        min_index_load = non_zero_values_load.idxmin()
    else:
        min_time_load=time_fit_len-1
        min_index_load=99999
               
        
        
    
    if min_time_valv < min_time_load:
        valve_limitCheck.loc[mask_index_nolimit] = valve_limitCheck.loc[mask_index_nolimit] | mask_valve_limit.loc[min_time_valv]    
        k_iter=min_time_valv
        status = 1
    elif min_time_valv >= min_time_load:
        load_limitCheck.loc[mask_index_nolimit] = load_limitCheck.loc[mask_index_nolimit] | mask_load_limit.loc[min_time_load]    
        k_iter=min_time_load
        status = 2
    else:
        k_iter=time_fit_len-1
        status = 0
    
    
    limit_start = min(k_iter, time_fit_len)
    t_limit = time_fit[limit_start]

    # Update indices based on limit checks
    index_limit_old = index_limit
    index_limit_old_valve = index_limit_valve
    index_limit_old_load = index_limit_load
    
    # Convert to boolean (True where value is non-zero, False where value is zero)
    index_limit_valve = valve_limitCheck_old.index[(valve_limitCheck_old) | (valve_limitCheck)].tolist()
    index_limit_load = load_limitCheck_old.index[(load_limitCheck_old) | (load_limitCheck)].tolist()

    index_limit_new_valve = np.setdiff1d(index_limit_valve, index_limit_old_valve).tolist()
    index_limit_new_load = np.setdiff1d(index_limit_load, index_limit_old_load).tolist()
    
    mask_index_nolimit = (~limitCheck_old) & (~valve_limitCheck) & (~load_limitCheck)

    index_limit = limitCheck_old.index[(limitCheck_old) | (valve_limitCheck) | (load_limitCheck)].tolist()

    index_limit_new = np.setdiff1d(index_limit, index_limit_old).tolist()
    
    limitCheck_old = (limitCheck_old) | (valve_limitCheck) | (load_limitCheck)
    valve_limitCheck_old = (valve_limitCheck_old) | (valve_limitCheck) 
    load_limitCheck_old = (load_limitCheck_old) | (load_limitCheck) 
    

    if not len(index_limit_new) or t_limit > t_n_star:
        break

    '''Load Limit Violation'''
    if status == 2:

        v_coefficient_limit = v_coeffifient_total.loc[index_limit_new_load,:]
        
        # Preallocate for speed
        loadlimiter_output = pd.DataFrame(np.zeros((time_fit_len, 
                                                    len(index_limit_new_load))), columns=index_limit_new_load)
        loadlimiter_shiftedValve=pd.DataFrame(np.zeros((time_fit_len, 
                                                    len(index_limit_new_load))), columns=index_limit_new_load)
        loadlimiter_shiftedUnitstep=pd.DataFrame(np.zeros((time_fit_len, 
                                                    len(index_limit_new_load))), columns=index_limit_new_load)
        unitTest_temp=np.zeros((time_fit_len,order+2));
        second_loadlimit=pd.DataFrame(np.zeros((time_fit_len, 
                                                    len(index_limit_new_load))), columns=index_limit_new_load)
        
        for i in index_limit_new_load:
            # Adjust index to handle the steam generator offset
            unitTest_temp = unittest_gas[:time_fit_len, :, i - len(output_gen_info[output_gen_info['Type'] == 'steam'])]
            loadlimiter_output.loc[:, i] = np.dot(unitTest_temp[:time_fit_len, :], v_coefficient_limit.loc[i, :])
        
            temp_loadlimiter_shiftedValve_2_save = np.zeros((time_fit_len, order + 2))
        
            # Initialize loadlimiter_shiftedUnitstep DataFrame
            if limit_start == 1:
                loadlimiter_shiftedUnitstep.loc[:, i] = (temp[i]) * unitTest_temp[:, 0]
            else:
                loadlimiter_shiftedUnitstep.loc[:, i] = (temp[i]) * np.concatenate([np.zeros(limit_start), unitTest_temp[:time_fit_len - limit_start, 0]])
        
            for ii in range(1, order + 3):
                temp_loadlimiter_shiftedValve_2 = np.zeros(time_fit_len)
        
                for iii in range(ii):
                    factor = (t_limit ** iii * factorial(ii - 1) / (factorial(iii) * factorial(ii - 1 - iii)))
                    shifted_valve_component = unitTest_temp[:, ii - iii] if limit_start == 1 else np.concatenate(
                        [np.zeros(limit_start), unitTest_temp[:time_fit_len - limit_start, ii - iii -1]]
                    )
                    temp_loadlimiter_shiftedValve_2 += factor * shifted_valve_component
        
                temp_loadlimiter_shiftedValve_2_save[:, ii - 1] = temp_loadlimiter_shiftedValve_2
        
            loadlimiter_shiftedValve.loc[:, i] = np.dot(temp_loadlimiter_shiftedValve_2_save, v_coefficient_limit.loc[i, :])
        
            # Polynomial fitting for load limiter
            poly_fit_coeffs=np.polyfit(time_fit[limit_start:], 
            loadlimiter_output.loc[limit_start:, i] - loadlimiter_shiftedValve.loc[limit_start:, i] + loadlimiter_shiftedUnitstep.loc[limit_start:, i], 
            order + 1)[::-1]
            
            b = np.dot(temp_loadlimiter_shiftedValve_2_save, poly_fit_coeffs)
        
            A = np.concatenate([
                np.zeros((limit_start, order + 2)),
                t_powers_2[:time_fit_len - limit_start, :] - unitTest_temp[:time_fit_len - limit_start, :]
            ])
        
            # Use pseudoinverse to solve for c
            c = np.dot(pinv(A), b)
        
            second_loadlimit.loc[:, i] = np.dot(
                np.concatenate([np.zeros((limit_start, order + 2)), t_powers_2[:time_fit_len - limit_start, :]]), c
            )
            
        # First part of the code to calculate `first_loadlimit`, `v_cal_load`
        first_loadlimit = np.array(loadlimiter_output) - loadlimiter_shiftedValve + loadlimiter_shiftedUnitstep
               
        # Creating an empty array with shape (len_time_fit, len_column)
        temp_1=pd.DataFrame(np.zeros((time_fit_len, len(index_limit_new_load))), columns=index_limit_new_load)
        temp_1.loc[:, index_limit_new_load]=temp[index_limit_new_load].values
                
        v_cal_load = first_loadlimit + second_loadlimit + temp_1

        # Calculate limited Pm response considering load limiter
        Pm_limit_new1 = np.zeros((time_fit_len, len(index_limit_new_load)))
        unitTest_temp = np.zeros((time_fit_len, order + 2))
        
        valve_coeffifient_limit = v_coeffifient_total.loc[index_limit_new_load, :]
        load_coeffifient_limit = np.zeros((len(index_limit_new_load), order + 2))
        
        for idx, i in enumerate(index_limit_new_load):
            # load_coeffifient_limit[idx, :] = Polynomial.fit(time_fit, v_cal_load.loc[:, i], order + 1).convert().coef[::-1]
            load_coeffifient_limit[idx, :] = np.polyfit(time_fit, v_cal_load.loc[:, i], order + 1)[::-1]
        
            
        for idx, i in enumerate(index_limit_new_load):
            unitTest_temp = unittest[:time_fit_len, :, i]
            gov_output = np.dot(unitTest_temp, valve_coeffifient_limit.loc[i, :])
        
            gov_shiftedValve = np.zeros(time_fit_len)
            gov_shiftedload = np.zeros(time_fit_len)
        
            for ii in range(1, order + 3):
                temp_gov_shiftedValve_2 = np.zeros(time_fit_len)
                temp_gov_shiftedload_2 = np.zeros(time_fit_len)
        
                for iii in range(ii):
                    factor = t_limit ** iii * factorial(ii - 1) / (factorial(iii) * factorial(ii - 1 - iii))
        
                    shifted_valve_component = unitTest_temp[:, ii - iii] if limit_start == 1 else np.concatenate(
                        [np.zeros(limit_start), unitTest_temp[:time_fit_len - limit_start, ii - iii -1]]
                    )
                    shifted_load_component = shifted_valve_component
        
                    temp_gov_shiftedValve_2 += factor * shifted_valve_component
                    temp_gov_shiftedload_2 += factor * shifted_load_component
        
                gov_shiftedValve += valve_coeffifient_limit.loc[i, ii - 1] * temp_gov_shiftedValve_2
                gov_shiftedload += load_coeffifient_limit[idx, ii - 1] * temp_gov_shiftedload_2
        
            Pm_limit_new1[:, idx] = alpha[i] * (gov_output - gov_shiftedValve + gov_shiftedload)
        
        # Initialize the limit check DataFrames with boolean values for faster operations  
        valveAfterLoad_limitCheck = pd.Series(data=np.zeros(numGen, dtype=bool), index=gen_data_active.index)
         
        
        # Creating an empty array with shape (len_time_fit, len_column)
        test_lim_u=pd.DataFrame(np.zeros((time_fit_len, len(index_limit_new_load))), columns=index_limit_new_load)
        test_lim_u.loc[:, index_limit_new_load]=lim_u[index_limit_new_load].values
        
        mask_valve_limit_load = (v_cal_load.loc[:, index_limit_new_load] >= test_lim_u)
        first_true_row = mask_valve_limit_load.idxmax()
        non_zero_values_valv = first_true_row[first_true_row != 0]

        if not non_zero_values_valv.empty:
            status2 = 1
            k_iter = non_zero_values_valv.min()
            min_index_valv = non_zero_values_valv.idxmin()
            
        else:
            status2 = 0
            k_iter=time_fit_len-1
            min_index_valv=99999
        
        

        # Calculate additional limited Pm response considering valve limiter after load limiter
        if status2 == 1:
            limit_start_valveAfterLoad = min(k_iter, time_fit_len)
            t_limit_valveAfterLoad = time_fit[limit_start_valveAfterLoad]
        
            # Update indices for limits
            index_limit_old_valveAfterLoad = index_limit_valveAfterLoad
            index_limit_valveAfterLoad = np.where(np.logical_or(limitCheck_old_valveAfterLoad, valveAfterLoad_limitCheck))[0]
            index_limit_new_valveAfterLoad = np.setdiff1d(index_limit_valveAfterLoad, index_limit_old_valveAfterLoad)
            limitCheck_old_valveAfterLoad = np.logical_or(limitCheck_old_valveAfterLoad, valveAfterLoad_limitCheck)
        
            _, temp_index, _ = np.intersect1d(index_limit_new_load, index_limit_new_valveAfterLoad, return_indices=True)
            load_coeffifient_valveAfterLoad = load_coeffifient_limit[temp_index, :]
        
            v_limit_value = lim_u[index_limit_new_valveAfterLoad]
            Pm_limit_new2 = np.zeros((time_fit_len, len(index_limit_new_valveAfterLoad)))
        
            for i, idx in enumerate(index_limit_new_valveAfterLoad):
                unitTest_temp = unittest[:time_fit_len, :, idx]
        
                if limit_start_valveAfterLoad == 1:
                    gov_shiftedUnitstep = v_limit_value[i] * unitTest_temp[:, 0]
                else:
                    gov_shiftedUnitstep = v_limit_value[i] * np.concatenate(
                        (np.zeros(limit_start_valveAfterLoad), unitTest_temp[:time_fit_len - limit_start_valveAfterLoad, 0])
                    )
        
                gov_shiftedload = np.zeros(time_fit_len)
                for ii in range(1, order + 3):
                    temp_gov_shiftedload_2 = np.zeros(time_fit_len)
        
                    for iii in range(ii):
                        factor = t_limit_valveAfterLoad ** iii * factorial(ii - 1) / (factorial(iii) * factorial(ii - 1 - iii))
        
                        shifted_load_component = (
                            unitTest_temp[:, ii - iii] if limit_start_valveAfterLoad == 1 else
                            np.concatenate(
                                (np.zeros(limit_start_valveAfterLoad),
                                 unitTest_temp[:time_fit_len - limit_start_valveAfterLoad, ii - iii -1])
                            )
                        )
                        temp_gov_shiftedload_2 += factor * shifted_load_component
        
                    gov_shiftedload += load_coeffifient_valveAfterLoad[i, ii - 1] * temp_gov_shiftedload_2
        
                Pm_limit_new2[:, i] = alpha[idx] * (-gov_shiftedload + gov_shiftedUnitstep)
        
            # Combine results
            Pm_limit_new = np.hstack((Pm_limit_new1, Pm_limit_new2))
        
        else:
            Pm_limit_new = Pm_limit_new1

    # Valve Limit Violation
    elif status == 1:
        v_coeffifient_limit = v_coeffifient_total.loc[index_limit_new_valve, :]
    
        Pm_limit_new = np.zeros((time_fit_len, len(index_limit_new_valve)))
        v_limit_value = lim_u[index_limit_new_valve]
        
        
        for idx, i in enumerate(index_limit_new_valve):
            
            ''' 3 Dimensional Array '''
            unitTest_temp = unittest[:time_fit_len, :, i]
            gov_output = unitTest_temp @ v_coeffifient_limit.loc[i, :]
    
            if limit_start == 1:
                gov_shiftedUnitstep = v_limit_value[i] * unitTest_temp[:, 0]
            else:
                gov_shiftedUnitstep = v_limit_value[i] * np.concatenate(
                    (np.zeros(limit_start), unitTest_temp[:time_fit_len - limit_start, 0])
                )
    
            gov_shiftedValve = np.zeros(time_fit_len)
            for ii in range(1, order + 3):
                temp_gov_shiftedValve_2 = np.zeros(time_fit_len)
    
                for iii in range(ii):
                    factor = t_limit ** iii * factorial(ii - 1) / (factorial(iii) * factorial(ii - 1 - iii))
    
                    shifted_valve_component = (
                        unitTest_temp[:, ii - iii -1] if limit_start == 1 else
                        np.concatenate(
                            (np.zeros(limit_start),
                             unitTest_temp[:time_fit_len - limit_start, ii - iii -1])
                        )
                    )
                    temp_gov_shiftedValve_2 += factor * shifted_valve_component
    
                gov_shiftedValve += v_coeffifient_limit.loc[i, ii - 1] * temp_gov_shiftedValve_2
    
            Pm_limit_new[:, idx] = alpha[i] * (gov_output - gov_shiftedValve + gov_shiftedUnitstep)
    
    else:
        Pm_limit_new = np.zeros((time_fit_len, 0))
        
    # Final summation across columns
    Pm_limit=Pm_limit+np.sum(Pm_limit_new, axis=1)


# Initialize parameters and variables
test_f0 = 1
test_time_plot = []
test_f = []
test_Pm = []
test_gov = []
test_t_vector1 = []
test_t_vector2 = []
test_test_gov = np.zeros((time_fit_len, mask_index_nolimit.sum()))

# Calculate `test_test_gov` using `v_coeffifient_limit`
v_coeffifient_limit = v_coeffifient_total.loc[mask_index_nolimit, :]

for i, idx in enumerate(mask_index_nolimit[mask_index_nolimit].index):
    unitTest_temp = unittest[:time_fit_len, :, idx]
    test_test_gov[:, i] = unitTest_temp @ v_coeffifient_limit.loc[idx, :]

k_iter = 0
for t in time_fit[time_fit < t_n_star + 1]:
    # Calculate `test_gov` at each time step
    test_gov = test_test_gov[k_iter, :]

    # Calculate `test_Pm` based on presence of limits
    if len(index_limit) == 0:
        test_Pm_val = np.sum(alpha[mask_index_nolimit] * test_gov)
    else:
        test_Pm_val = np.sum(alpha[mask_index_nolimit] * test_gov) + Pm_limit[k_iter]
    test_Pm.append(test_Pm_val)

    # Frequency difference calculation
    test_f_diff = 1 / (2 * Hsys) * (test_Pm_val - P_event)

    # Calculate frequency at time t
    if t == 0:
        test_f_val = test_f0
    else:
        test_f_val = test_f[-1] + test_f_diff * (time_fit[k_iter] - time_fit[k_iter - 1])
    test_f.append(test_f_val)

    # Store time plot values
    test_time_plot.append(t)

    k_iter += 1

# Calculate f_nadir
f_nadir = min(test_f) * 60

# Output the results
print("t_n_star:", t_n_star)
print("f_nadir:", f_nadir)

end_time = time.time()

# Print the elapsed time
print("Execution time:", end_time - start_time, "seconds")
    