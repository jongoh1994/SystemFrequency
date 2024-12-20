import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import unit
import datagen
import readgen
import time
from scipy.linalg import pinv
from scipy.special import factorial

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
event_gen = pd.DataFrame({
    'BusNum' : [7098, 7099],
    'GenID' : [1, 1]
})

# NOTE copy not needed - it is slow 
output_gen_info    = output_gen_info_org
output_gov_gasInfo = output_gov_gasInfo_org

output_gen_info['TSVmax']     = 0.87
output_gov_gasInfo['TSLdref'] = 0.835
gen_data_active, gov_gasInfo  = datagen.DataGenerate(
    gen_data, 
    output_gen_info, 
    output_gov_gasInfo, 
    event_gen
)

numGen  = gen_data_active.shape[0]
P_event = 2558/sum(gen_data_active['GenMVABase'])
Hsys    = sum(gen_data_active['TSH']*100)/sum(gen_data_active['GenMVABase'])

## Coefficient for Polynomial Fitting
# Lambda matrix for integral of ramp polynomials

# Initialize variables
coeffi = 1 / (np.arange(1, order + 2))  # Coefficients for lambda matrix
lamda  = np.diagflat(1-coeffi)

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
default_index = np.arange(time_fit_len)

# Filter the columns in output_unittest['ramp_v'] that match the 'Index' values from gen_data_active
ramp = ramp[:,gen_data_active.index]
ramp_v = ramp_v[:,gen_data_active.index]

rampResponse_fit = ramp[0:time_fit_len, :]
v_rampResponse_fit = ramp_v[0:time_fit_len, :]

# Polynomial fitting for each generator
# Fit polynomials & reverse coefficient order for later evaluation
tempa = np.polyfit(time_fit,rampResponse_fit, order).T[:,::-1] # Flipping the polynomial coefficients
tempb = np.polyfit(time_fit,  v_rampResponse_fit, order).T[:,::-1]
p_gov_total = pd.DataFrame(tempa, index=gen_data_active.index)
p_valve_total = pd.DataFrame(tempb, index=gen_data_active.index)

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

# Calculate nadir time by checking error
Pout_factor = P_event / (2 * Hsys)

# Pre-create Orders
tpo = np.arange(order + 1)
tpo1 = np.arange(1, order + 2)
tpo2 = np.arange(order + 2)
t_powers = np.power(time_fit[:, None], tpo)
t_powers_1 = np.power(time_fit[:, None], tpo1)
t_powers_2 = np.power(time_fit[:, None], tpo2)

# NOTE creating pre-multiplied variable
tpow_lam = t_powers*(1-coeffi)

zeros_column = pd.DataFrame(np.zeros((p_valve_total.shape[0], 1)), index=gen_data_active.index)

# Insert zero-filled columns at the beginning and end
ramp_p_integ = pd.concat([zeros_column, p_valve_total], axis=1)
ramp_p = pd.concat([p_valve_total, zeros_column], axis=1)

ramp_p_integ.columns = range(ramp_p_integ.shape[1])
ramp_p.columns = range(ramp_p.shape[1]) 

# Violation Types
LOAD_LIMIT_VIOLATION = 2
VALVE_LIMIT_VIOLATION = 1

# Precompute limit values for faster access
lim_u = gen_data_active['valve_lim_u']

const_Lref = gov_gasInfo['TSLdref'] / gov_gasInfo['TSKturb'] + gov_gasInfo['TSWfnl']
const_v_0 =  gov_gasInfo['GenMW'] / gov_gasInfo['GenMVABase'] / gov_gasInfo['TSKturb'] + gov_gasInfo['TSWfnl']
inner_limit = 1 - const_v_0
temp = (const_Lref - const_v_0) * gov_gasInfo['TSKpload']

numSteam=len(output_gen_info[output_gen_info['Type'] == 'steam'])

z_coeffi = np.zeros(order+2)
z_coeffi[1:] = coeffi

# NOTE numpy diagonal matrix multiplication is slower than using * multiplication by vector.
temp_matrix1 = ramp_p_integ * z_coeffi
temp_matrix2 = t_powers_1   * coeffi @ p_valve_total.T
temp_matrix3 = t_powers @ p_valve_total.T

# Pre-define this function
factorFunc = lambda k, j:  t_limit ** k * factorial(j - 1) / (factorial(k) * factorial(j - 1 - k))

# Pre-Process Governor Mask
is_ggov1 = gen_data_active['TSGenGovernorName'] == 'GGOV1'
        
while True:
    
    # Frequency nadir calculation
    while True:
        
        p_gov_part = p_gov_total.loc[mask_index_nolimit]
        p_valve_part = p_valve_total.loc[mask_index_nolimit]
                  
        # Multiply each row of `p_gov_part_filtered` by the corresponding element in `alpha_filtered`
        p_gov_sys = (p_gov_part.T * alpha[mask_index_nolimit]).T.sum()

        # Find Frequency Nadir Time by Minimizing Error
        if not index_limit:
            error_save = np.abs(tpow_lam @ p_gov_sys - 2 * Hsys)
        else:
            error_save = np.abs(Pout_factor * tpow_lam @ p_gov_sys + Pm_limit - P_event)

        t_n_star = time_fit[np.argmin(error_save)]


        if t_fit > t_n_star:
            break

        # NOTE this does not get executed with default settings so I am not modifying for now
        else:
            t_fit += t_fit_step
            time_fit = time10[time10 <= t_fit]
            time_fit_len = len(time_fit)
            default_index = np.arange(time_fit_len)

            t_powers = np.power(time_fit[:, None], tpo)
            t_powers_1 = np.power(time_fit[:, None], tpo1)
            t_powers_2 = np.power(time_fit[:, None], tpo2)

            # NOTE creating pre-multiplied variable
            tpow_lam = t_powers*(1-coeffi)
            
            rampResponse_fit = ramp[0:time_fit_len, :]
            v_rampResponse_fit = ramp_v[0:time_fit_len, :]


            p_gov_total = np.empty((numGen, order))
            p_valve_total = np.empty((numGen, order))
            
            # Polynomial fitting for each generator
            for i in range(numGen):
                # Fit polynomials & reverse coefficient order for later evaluation
                p_gov_total[i] = np.polyfit(time_fit,rampResponse_fit[:, i], order)[::-1] # Flipping the polynomial coefficients
                p_valve_total[i] = np.polyfit(time_fit,  v_rampResponse_fit[:, i], order)[::-1] # Flipping the polynomial coefficients
            
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

    # Do not need to make diagonal
    a = t_powers_1 * coeffi

    # Valve coefficient approximation   
    v_coeffifient_total = -P_event / (2 * Hsys) * (temp_matrix1 / t_n_star - ramp_p)
    v_cal = -P_event / (2 * Hsys) * (1 / t_n_star * (temp_matrix2) - temp_matrix3)
    a = -P_event / (2 * Hsys) * (1 / 2.5125 * (temp_matrix2) - temp_matrix3)

    # Load limit response at no limit governors
    # NOTE There is some redundant action here, can simplify ccode greatly by just using binary mask
    # tolist is a very slow action
    nolimit_index_gas = (
        gen_data_active.index[is_ggov1 & mask_index_nolimit]
    ).tolist() 
    
    v_coefficient_limit = v_coeffifient_total.loc[nolimit_index_gas].to_numpy()
    

    # 'Empty' is much more effective
    loadlimiter_results = np.empty((time_fit_len, len(nolimit_index_gas)))
    
    '''
    3 Dimensional Array
    '''

    slice_gas_all = unittest_gas[:time_fit_len, :,:]
    for idx, i in enumerate(nolimit_index_gas):

        # Extract the correct slice from unittest_gas
        slice_gas = slice_gas_all[:,:,i - numSteam]
        
        # Perform matrix multiplication and add temp[i]
        loadlimiter_results[:, idx] = slice_gas @ v_coefficient_limit[idx] + temp[i] 
  
    # NOTE This array/df is commented out in the above forloop, so it can be initiated here
    loadlimiter_output_initial = pd.DataFrame(loadlimiter_results, columns=nolimit_index_gas)
    
    # Initialize the limit check DataFrames with boolean values for faster operations  
    valve_limitCheck = pd.Series(index=gen_data_active.index, dtype=bool)
    load_limitCheck = pd.Series(index=gen_data_active.index, dtype=bool)
    valve_limitCheck[:] = False
    load_limitCheck[:] = False

    # Create boolean masks for `index_nolimit` and `nolimit_index_gas` to apply conditions across all time steps
    mask_nolimit_index_gas1 = pd.Series(gen_data_active.index.isin(nolimit_index_gas), index=gen_data_active.index)
    
    
    column = mask_index_nolimit.index[mask_index_nolimit]
    
    # Creating an empty array with shape (len_time_fit, len_column)
    AAA = np.repeat([lim_u[column].values], time_fit_len, axis=0)
    test_lim_u = pd.DataFrame(AAA, columns=column)
    
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
    if status == LOAD_LIMIT_VIOLATION:


        v_coefficient_limit = v_coeffifient_total.loc[index_limit_new_load,:].to_numpy()
        
        # NOTE empty, not zeros
        loadlimiter_output = np.empty((default_index.shape[0],len(index_limit_new_load)))
        loadlimiter_shiftedValve = np.empty((default_index.shape[0],len(index_limit_new_load)))
        loadlimiter_shiftedUnitstep = np.empty((default_index.shape[0],len(index_limit_new_load)))
        unitTest_temp = np.zeros((time_fit_len,order+2))

        second_loadlimit = np.empty((default_index.shape[0],len(index_limit_new_load))) 

        for pos, i in enumerate(index_limit_new_load):
            # Adjust index to handle the steam generator offset
            unitTest_temp = unittest_gas[:time_fit_len, :, i - len(output_gen_info[output_gen_info['Type'] == 'steam'])]

            loadlimiter_output[:, pos] = np.dot(unitTest_temp[:time_fit_len, :], v_coefficient_limit[pos])
        
            temp_loadlimiter_shiftedValve_2_save = np.zeros((time_fit_len, order + 2))

            shifted_compA = lambda k, j: unitTest_temp[:, j - k]
            shifted_compB = lambda k, j: np.concatenate(
                        [np.zeros(limit_start), unitTest_temp[:time_fit_len - limit_start, j - k -1]]
                    )
        
            # Initialize loadlimiter_shiftedUnitstep DataFrame
            if limit_start == 1:
                loadlimiter_shiftedUnitstep[:, pos] = (temp[i]) * unitTest_temp[:, 0]
                shifted_comp = shifted_compA
            else:
                loadlimiter_shiftedUnitstep[:, pos] = (temp[i]) * np.concatenate([np.zeros(limit_start), unitTest_temp[:time_fit_len - limit_start, 0]])
                shifted_comp = shifted_compB
        
            for j in range(1, order + 3):
                temp_loadlimiter_shiftedValve_2 = np.zeros(time_fit_len)
            
                for k in range(j):
                    temp_loadlimiter_shiftedValve_2 += factorFunc(k, j) * shifted_comp(k, j)
        
                temp_loadlimiter_shiftedValve_2_save[:, j - 1] = temp_loadlimiter_shiftedValve_2
        
            loadlimiter_shiftedValve[:, pos] = np.dot(temp_loadlimiter_shiftedValve_2_save, v_coefficient_limit[pos])
        
            # Polynomial fitting for load limiter
            poly_fit_coeffs=np.polyfit(
                time_fit[limit_start:], 
                loadlimiter_output[limit_start:, pos] - loadlimiter_shiftedValve[limit_start:, pos] + loadlimiter_shiftedUnitstep[limit_start:, pos], 
                order + 1
            )[::-1]
            
            b = np.dot(temp_loadlimiter_shiftedValve_2_save, poly_fit_coeffs)
        
            A = np.concatenate([
                np.zeros((limit_start, order + 2)),
                t_powers_2[:time_fit_len - limit_start, :] - unitTest_temp[:time_fit_len - limit_start, :]
            ])
        
            # Use pseudoinverse to solve for c
            c = np.dot(pinv(A), b)
        
            second_loadlimit[:, pos] = np.dot(
                np.concatenate([np.zeros((limit_start, order + 2)), t_powers_2[:time_fit_len - limit_start, :]]), c
            )
            
        # First part of the code to calculate `first_loadlimit`, `v_cal_load`
        first_loadlimit = loadlimiter_output - loadlimiter_shiftedValve + loadlimiter_shiftedUnitstep
                  
        # Creating an empty array with shape
        v_cal_load = first_loadlimit + second_loadlimit + temp[index_limit_new_load].values#temp_1

        # Calculate limited Pm response considering load limiter
        Pm_limit_new1 = np.empty((time_fit_len, len(index_limit_new_load)))
        
        valve_coeffifient_limit = v_coeffifient_total.loc[index_limit_new_load, :]

        # NOTE we can do this outside the loop :)
        load_coeffifient_limit = np.polyfit(time_fit, v_cal_load, order + 1).T[:,::-1]

        for idx, i in enumerate(index_limit_new_load):

            unitTest_temp = unittest[:time_fit_len, :, i]
            gov_output = np.dot(unitTest_temp, valve_coeffifient_limit.loc[i, :])
        
            gov_shifted_tot = np.zeros(time_fit_len)
        
            for j in range(1, order + 3):
                temp_gov_shiftedValve_2 = np.zeros(time_fit_len)
                temp_gov_shiftedload_2 = np.zeros(time_fit_len)
        
                for k in range(j):
                    factor = factorFunc(k,j)
        
                    shifted_valve_component = unitTest_temp[:, j - k] if limit_start == 1 else np.concatenate(
                        [np.zeros(limit_start), unitTest_temp[:time_fit_len - limit_start, j - k -1]]
                    )
                    shifted_load_component = shifted_valve_component
        
                    temp_gov_shiftedValve_2 += factor * shifted_valve_component
                    temp_gov_shiftedload_2 += factor * shifted_load_component
        
                gov_shifted_tot += valve_coeffifient_limit.loc[i, j - 1] * temp_gov_shiftedValve_2 - load_coeffifient_limit[idx, j - 1] * temp_gov_shiftedload_2
        
            Pm_limit_new1[:, idx] = alpha[i] * (gov_output - gov_shifted_tot)
        
        # NOTE this was not doing anything so I removed
        #valveAfterLoad_limitCheck = pd.Series(data=np.zeros(numGen, dtype=bool), index=gen_data_active.index)
         
        # Creating an empty array with shape (len_time_fit, len_column)
        first_true_row = np.argmax(v_cal_load >= lim_u[index_limit_new_load].values,axis=1)
        non_zero_values_valv = first_true_row[first_true_row != 0]

        if len(non_zero_values_valv)>0:
            status2 = 1
            k_iter = non_zero_values_valv.min()
            # NOTE not in use? #min_index_valv = np.argmin(non_zero_values_valv)#.idxmin()
            
        else:
            status2 = 0
            k_iter=time_fit_len-1
            #min_index_valv=99999
        
        # Calculate additional limited Pm response considering valve limiter after load limiter
        if status2 == 1:
            limit_start_valveAfterLoad = min(k_iter, time_fit_len)
            t_limit_valveAfterLoad = time_fit[limit_start_valveAfterLoad]

            # NOTE make here instead to save memory
            ZRO = np.zeros(limit_start_valveAfterLoad)
        
            # Update indices for limits
            index_limit_old_valveAfterLoad = index_limit_valveAfterLoad
            index_limit_valveAfterLoad = np.where(limitCheck_old_valveAfterLoad) 
            index_limit_new_valveAfterLoad = np.setdiff1d(index_limit_valveAfterLoad, index_limit_old_valveAfterLoad)
            limitCheck_old_valveAfterLoad =  limitCheck_old_valveAfterLoad 

            _, temp_index, _ = np.intersect1d(index_limit_new_load, index_limit_new_valveAfterLoad, return_indices=True)
            load_coeffifient_valveAfterLoad = load_coeffifient_limit[temp_index, :]
        
            v_limit_value = lim_u[index_limit_new_valveAfterLoad]
            Pm_limit_new2 = np.empty((time_fit_len, len(index_limit_new_valveAfterLoad)))

            factorFuncB = lambda k, j: t_limit_valveAfterLoad ** k * factorial(j - 1) / (factorial(k) * factorial(j - 1 - k))

            # Could formulated this outside of loop
            # gov_shiftedUnitStep = v_limi_value * 
            # Empty list - not the cause as of now
            for i, idx in enumerate(index_limit_new_valveAfterLoad):
                unitTest_temp = unittest[:time_fit_len, :, idx]

                gov_shiftedload = np.zeros(time_fit_len)

                # Create Functions here, don't use if inside large loop, very slow
                shifted_compA = lambda k, j: unitTest_temp[:, j - k]
                shifted_compB = lambda k, j: np.concatenate((ZRO, unitTest_temp[:time_fit_len - limit_start_valveAfterLoad, j - k -1]))
        
                if limit_start_valveAfterLoad == 1:
                    gov_shiftedUnitstep = v_limit_value[i] * unitTest_temp[:, 0]
                    shifted_comp = shifted_compA
                else:
                    gov_shiftedUnitstep = v_limit_value[i] * np.concatenate((ZRO, unitTest_temp[:time_fit_len - limit_start_valveAfterLoad, 0]))
                    shifted_comp = shifted_compB 

                for j in range(1, order + 3):
                    temp_gov_shiftedload_2 = np.zeros(time_fit_len)
                    for k in range(j):
                        temp_gov_shiftedload_2 += factorFuncB(k, j) * shifted_comp(k, j)
                    gov_shiftedload += load_coeffifient_valveAfterLoad[i, j - 1] * temp_gov_shiftedload_2
        
                Pm_limit_new2[:, i] = alpha[idx] * (-gov_shiftedload + gov_shiftedUnitstep)
        
            # Combine results
            Pm_limit_new = np.hstack((Pm_limit_new1, Pm_limit_new2))
        
        else:
            Pm_limit_new = Pm_limit_new1

    # Valve Limit Violation
    elif status == VALVE_LIMIT_VIOLATION:

        v_coeffifient_limit = v_coeffifient_total.loc[index_limit_new_valve, :].to_numpy()
    
        Pm_limit_new = np.empty((time_fit_len, len(index_limit_new_valve)))
        v_limit_value = lim_u[index_limit_new_valve]
        
        factorFunc = lambda k, j:  t_limit ** k * factorial(j - 1) / (factorial(k) * factorial(j - 1 - k))
        
        for idx, i in enumerate(index_limit_new_valve):
            
            ''' 3 Dimensional Array '''
            unitTest_temp = unittest[:time_fit_len, :, i]
            gov_output = unitTest_temp @ v_coeffifient_limit[idx]

            # Create Functions here, don't use if inside large loop, very slow
            shifted_compA = lambda k, j: unitTest_temp[:, j - k -1]
            shifted_compB = lambda k, j: np.concatenate((np.zeros(limit_start),unitTest_temp[:time_fit_len - limit_start, j - k -1]))
    
    
            if limit_start == 1:
                gov_shiftedUnitstep = v_limit_value[i] * unitTest_temp[:, 0]
                shifted_comp = shifted_compA
            else:
                gov_shiftedUnitstep = v_limit_value[i] * np.concatenate((np.zeros(limit_start), unitTest_temp[:time_fit_len - limit_start, 0]))
                shifted_comp = shifted_compB

    
            gov_shiftedValve = np.zeros(time_fit_len)
            for j in range(1, order + 3):
                temp_gov_shiftedValve_2 = np.zeros(time_fit_len)
    
                for k in range(j):
                    temp_gov_shiftedValve_2 += factorFunc(k,j) * shifted_comp(k,j)
    
                gov_shiftedValve += v_coeffifient_limit[idx, j - 1] * temp_gov_shiftedValve_2
    
            Pm_limit_new[:, idx] = alpha[i] * (gov_output - gov_shiftedValve + gov_shiftedUnitstep)
    
    else:
        Pm_limit_new = 0 
        
    # Final summation across columns
    Pm_limit += np.sum(Pm_limit_new, axis=1)


# Initialize parameters and variables
test_f0 = 1
test_time_plot = []
test_f = []
test_Pm = []
test_gov = []
test_t_vector1 = []
test_t_vector2 = []
test_test_gov = np.empty((time_fit_len, mask_index_nolimit.sum()))

# Calculate `test_test_gov` using `v_coeffifient_limit`
v_coeffifient_limit = v_coeffifient_total.loc[mask_index_nolimit, :].to_numpy()

for i, idx in enumerate(mask_index_nolimit[mask_index_nolimit].index):
    test_test_gov[:, i] = unittest[:time_fit_len, :, idx] @ v_coeffifient_limit[i]

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
    