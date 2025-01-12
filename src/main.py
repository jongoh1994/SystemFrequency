import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import unit
import datagen
import readgen
import time
from scipy.linalg import pinv
from scipy.special import factorial
from functools import partial
from scipy.special import factorial as fact
from numba import jit

np.set_printoptions(suppress=True)

''' FILE I/O '''
fname_steam, fname_gas, fname_gen = 'data/IEEEG1_data.csv', 'data/GGOV1_data.csv', 'data/GEN_data.csv'
fname_output = "data/OUTPUT"

order, t_sim = 6, 10

rerun = False
if rerun:
    ramp, ramp_v, unittest_gas, unittest  = unit.get_matlab_output(fname_steam, fname_gas, order, t_sim, fname_output)
else:
    ramp, ramp_v, unittest_gas, unittest  = unit.read_matlab_output(fname_output)

# Get Matlab Output and Generator Information Dataframes
output_gen_info_org, output_gov_gasInfo_org, gen_data = readgen.get_gen_params(fname_steam, fname_gas, fname_gen)


def power_series_matrix(order):
    '''
    Description:
        Pre-Process polynomial matrix structure based on order
    '''
    J = np.zeros((order+2, order+2))
    K = 0
    for k in range(order+2):
        one = np.ones(order+2)[:order+2-k]
        K += np.diag(one*k, k)
        J[:k, k] = k

    FACT_MAT = np.divide(fact(J), fact(K)*fact(J-K))

    # Then set lower triangle to zero
    FACT_MAT[np.tril_indices_from(FACT_MAT)] = 0
    FACT_MAT[np.diag_indices_from(FACT_MAT)] = 1

    return FACT_MAT, K

def run_test(order, t_sim, ramp, ramp_v, unittest_gas, unittest, gen_data_active, gov_gasInfo, numSteam):
   
    FACT_MAT, K = power_series_matrix(order)


    gen_data_active_index = gen_data_active.index

    numGen  = gen_data_active.shape[0]
    P_event = 2558 / sum(gen_data_active['GenMVABase'])
    Hsys    = sum(gen_data_active['TSH']*100)/sum(gen_data_active['GenMVABase'])

    # Coefficient for Polynomial Fitting
    coeffi = 1/np.arange(1, order + 2)

    # Proportional coefficient for governor output (based on machine size)
    alpha = gen_data_active['GenMVABase'] / sum(gen_data_active['GenMVABase'])

    # Initialization
    mask_index_nolimit = pd.Series(data=np.ones(numGen, dtype=bool), index=gen_data_active_index)
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
    ramp   = ramp[:,gen_data_active_index]
    ramp_v = ramp_v[:,gen_data_active_index]

    rampResponse_fit   = ramp[0:time_fit_len, :]
    v_rampResponse_fit = ramp_v[0:time_fit_len, :]

    # Polynomial fitting for each generator
    # Fit polynomials & reverse coefficient order for later evaluation
    tempa = np.polyfit(time_fit,  rampResponse_fit  , order).T[:,::-1] # Flipping the polynomial coefficients
    tempb = np.polyfit(time_fit,  v_rampResponse_fit, order).T[:,::-1]
    p_gov_total = pd.DataFrame(tempa, index=gen_data_active_index)
    p_valve_total = pd.DataFrame(tempb, index=gen_data_active_index)

    # Frequency Nadir Calculation - No Limit Violation
    t_fit_step = 1
    Pm_limit = np.zeros(time_fit_len)

    # Initialize arrays with zeros
    limitCheck_old        = pd.Series(data=np.zeros(numGen, dtype=bool), index=gen_data_active_index)
    valve_limitCheck_old  = pd.Series(data=np.zeros(numGen, dtype=bool), index=gen_data_active_index)
    load_limitCheck_old   = pd.Series(data=np.zeros(numGen, dtype=bool), index=gen_data_active_index)
    limitCheck_old_valveAfterLoad = pd.Series(data=np.zeros(numGen, dtype=bool), index=gen_data_active_index)

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

    zeros_column = pd.DataFrame(np.zeros((p_valve_total.shape[0], 1)), index=gen_data_active_index)

    # Insert zero-filled columns at the beginning and end
    ramp_p_integ = pd.concat([zeros_column, p_valve_total], axis=1)
    ramp_p       = pd.concat([p_valve_total, zeros_column], axis=1)

    ramp_p_integ.columns = range(ramp_p_integ.shape[1])
    ramp_p.columns = range(ramp_p.shape[1]) 

    # Violation Types
    LOAD_LIMIT_VIOLATION = 2
    VALVE_LIMIT_VIOLATION = 1

    # Precompute limit values for faster access
    lim_u = gen_data_active['valve_lim_u']

    const_Lref = gov_gasInfo['TSLdref'] / gov_gasInfo['TSKturb'] + gov_gasInfo['TSWfnl']
    const_v_0 =  gov_gasInfo['GenMW'] / gov_gasInfo['GenMVABase'] / gov_gasInfo['TSKturb'] + gov_gasInfo['TSWfnl']
    temp = (const_Lref - const_v_0) * gov_gasInfo['TSKpload']

    

    z_coeffi = np.zeros(order+2)
    z_coeffi[1:] = coeffi

    # NOTE numpy diagonal matrix multiplication is slower than using * multiplication by vector.
    temp_matrix1 = ramp_p_integ * z_coeffi
    temp_matrix2 = t_powers_1   * coeffi @ p_valve_total.T
    temp_matrix3 = t_powers @ p_valve_total.T

   
    # Pre-Process Governor Mask
    is_ggov1 = gen_data_active['TSGenGovernorName'] == 'GGOV1'


            
    while True:
        
        ''' Frequency nadir calculation'''

        while True:
            
            # NOTE this was not referenced #p_valve_part = p_valve_total.loc[mask_index_nolimit]
            p_gov_part   = p_gov_total.loc[mask_index_nolimit]
            

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

            
            # NOTE NOT CALLED
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
                p_gov_total   = pd.DataFrame(p_gov_total  , index=gen_data_active_index)
                p_valve_total = pd.DataFrame(p_valve_total, index=gen_data_active_index)
                
                # NOTE make these *empty* sets not zero.
                zeros_column = pd.DataFrame(np.zeros((p_valve_total.shape[0], 1)), index=gen_data_active_index)
                
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
   

        ''' Coefficients '''

        # Valve coefficient approximation   
        v_coeffifient_total = -P_event / (2 * Hsys) * (temp_matrix1 / t_n_star - ramp_p)
        v_cal               = -P_event / (2 * Hsys) * (1 / t_n_star * (temp_matrix2) - temp_matrix3)

        # Load limit response at no limit governors
        nolimit_index_gas   = gen_data_active_index[is_ggov1 & mask_index_nolimit]
        vcoeff = v_coeffifient_total.loc[nolimit_index_gas].to_numpy()
        loadlimiter_results = np.empty((time_fit_len, len(nolimit_index_gas)))
        
        slice_gas_all = unittest_gas[:time_fit_len, :,:]
        for idx, i in enumerate(nolimit_index_gas):

            # Extract the correct slice from unittest_gas
            slice_gas = slice_gas_all[:,:,i - numSteam]
            
            # Perform matrix multiplication and add temp[i]
            loadlimiter_results[:, idx] = slice_gas @ vcoeff[idx] + temp[i] 
    
        # NOTE This array/df is commented out in the above forloop, so it can be initiated here
        loadlimiter_output_initial = pd.DataFrame(loadlimiter_results, columns=nolimit_index_gas)
        
        # Initialize the limit check DataFrames with boolean values for faster operations  
        valve_limitCheck = pd.Series(index=gen_data_active_index, dtype=bool)
        load_limitCheck = pd.Series(index=gen_data_active_index, dtype=bool)
        valve_limitCheck[:] = False
        load_limitCheck[:]  = False

        # Create boolean masks for `index_nolimit` and `nolimit_index_gas` to apply conditions across all time steps
        mask_nolimit_index_gas1 = pd.Series(gen_data_active_index.isin(nolimit_index_gas), index=gen_data_active_index)
        
        
        # Creating an empty array with shape (len_time_fit, len_column)
        column = mask_index_nolimit.index[mask_index_nolimit]
        test_lim_u = pd.DataFrame(
            np.repeat([lim_u[column].values], time_fit_len, axis=0),
            columns=column
        )
        
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
            
        
        ''' Load Limit Check '''

        column = mask_nolimit_index_gas1.index[mask_nolimit_index_gas1]
        
        # Creating an empty array with shape (len_time_fit, len_column)
        mask_load_limit = (v_cal.loc[:, column] >= loadlimiter_output_initial)
        first_true_row = mask_load_limit.idxmax()
        non_zero_values_load = first_true_row[first_true_row != 0]
        
        if not non_zero_values_load.empty:   
            min_time_load = non_zero_values_load.min()
            min_index_load = non_zero_values_load.idxmin()
        else:
            min_time_load = time_fit_len-1
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
        t_limit     = time_fit[limit_start]

        ''' INDECIES OF VIOLATIONS '''

        
        index_limit_valve_old = index_limit_valve
        index_limit_load_old  = index_limit_load
        index_limit_old       = index_limit

        limitCheck_old       |= valve_limitCheck | load_limitCheck
        valve_limitCheck_old |= valve_limitCheck 
        load_limitCheck_old  |= load_limitCheck
        mask_index_nolimit    = ~limitCheck_old

        index_limit_valve = valve_limitCheck_old.index[valve_limitCheck_old]
        index_limit_load  = load_limitCheck_old.index[load_limitCheck_old]
        index_limit       = limitCheck_old.index[limitCheck_old].tolist()

        index_limit_new_valve = np.setdiff1d(index_limit_valve, index_limit_valve_old)
        index_limit_new_load  = np.setdiff1d(index_limit_load, index_limit_load_old)
        index_limit_new       = np.setdiff1d(index_limit, index_limit_old)
        
        
        

        if not len(index_limit_new) or t_limit > t_n_star:
            break

        ''' Handle Violations '''

        if status == LOAD_LIMIT_VIOLATION:

            # POWER SERIES MATRIX
            PS     = t_limit**K * FACT_MAT
            vcoeff = v_coeffifient_total.loc[index_limit_new_load,:].to_numpy()
            
            LL_OUTPUT   = np.empty((default_index.shape[0],len(index_limit_new_load)))
            LL_SV       = np.zeros((default_index.shape[0],len(index_limit_new_load)))
            LL_SHFT_US  = np.zeros((default_index.shape[0],len(index_limit_new_load)))

            second_loadlimit = np.zeros((default_index.shape[0],len(index_limit_new_load))) 

            # Determine Shift
            SHFT = 0 if limit_start == 1 else limit_start


            for pos, i in enumerate(index_limit_new_load):


                DATA  = unittest_gas[:time_fit_len, :, i - numSteam]

                LL_SV_SAVE        = DATA[:-SHFT]@PS
                LL_OUTPUT[:, pos] = DATA@vcoeff[pos]
                LL_SV[SHFT:, pos] = LL_SV_SAVE@vcoeff[pos]
                
            
                # Polynomial fitting for load limiter
                LL_SHFT_US[SHFT:, pos] = temp[i] * DATA[:-SHFT][:, 0]
                poly_fit_coeffs = np.polyfit(
                    time_fit[limit_start:], 
                    LL_OUTPUT[limit_start:, pos] - LL_SV[limit_start:, pos] + LL_SHFT_US[limit_start:, pos], 
                    order + 1
                )[::-1]
                
                
            
                TP2 = t_powers_2[:time_fit_len - limit_start, :]

                # Removed by removing psuedo inverse zeros
                Ainv = pinv(TP2 - DATA[:-SHFT])
                b    = LL_SV_SAVE@poly_fit_coeffs
                c    = Ainv@b
            

                second_loadlimit[limit_start:, pos] = TP2@c
                
            # First part of the code to calculate `first_loadlimit`, `v_cal_load`
            first_loadlimit = LL_OUTPUT - LL_SV + LL_SHFT_US
                    
            # Creating an empty array with shape
            v_cal_load = first_loadlimit + second_loadlimit + temp[index_limit_new_load].values

            valve_coeffifient_limit = v_coeffifient_total.loc[index_limit_new_load, :]

            # NOTE we can do this outside the loop :)
            load_coeffifient_limit = np.polyfit(time_fit, v_cal_load, order + 1).T[:,::-1]

            
            for idx, i in enumerate(index_limit_new_load):

                # Shifted and UnShifted Data
                DATA = unittest[:time_fit_len, :, i]

                # Coefficients
                vecValve = valve_coeffifient_limit.loc[i, :]
                vecLoad  = load_coeffifient_limit[idx, :]
         
                Pm_limit         += alpha[i] * DATA@vecValve
                Pm_limit[SHFT:]  -= alpha[i] * DATA[:-SHFT]@PS@(vecValve-vecLoad)

            
            # NOTE this was not doing anything so I removed
            #valveAfterLoad_limitCheck = pd.Series(data=np.zeros(numGen, dtype=bool), index=gen_data_active.index)
            
            # Creating an empty array with shape (len_time_fit, len_column)
            first_true_row = np.argmax(v_cal_load >= lim_u[index_limit_new_load].values,axis=1)
            non_zero_values_valv = first_true_row[first_true_row != 0]

            if len(non_zero_values_valv)>0:
                status2 = 1
                k_iter  = non_zero_values_valv.min()
                # NOTE not in use? #min_index_valv = np.argmin(non_zero_values_valv)#.idxmin()
                
            else:
                status2 = 0
                k_iter  = time_fit_len-1
                #min_index_valv=99999
            
            # Calculate additional limited Pm response considering valve limiter after load limiter
            if status2 == 1:

                limit_start_valveAfterLoad = min(k_iter, time_fit_len)
                t_limit_valveAfterLoad     = time_fit[limit_start_valveAfterLoad]

                PS = t_limit_valveAfterLoad**K * FACT_MAT
            
                # Update indices for limits
                index_limit_old_valveAfterLoad  = index_limit_valveAfterLoad
                index_limit_valveAfterLoad      = np.where(limitCheck_old_valveAfterLoad) 
                index_limit_new_valveAfterLoad  = np.setdiff1d(index_limit_valveAfterLoad, index_limit_old_valveAfterLoad)
                limitCheck_old_valveAfterLoad   = limitCheck_old_valveAfterLoad 

                _, temp_index, _ = np.intersect1d(index_limit_new_load, index_limit_new_valveAfterLoad, return_indices=True)
                load_coeffifient_valveAfterLoad = load_coeffifient_limit[temp_index, :]
            
                v_limit_value = lim_u[index_limit_new_valveAfterLoad]

                for i, idx in enumerate(index_limit_new_valveAfterLoad):

                    SHFT = 0 if limit_start_valveAfterLoad == 1 else limit_start_valveAfterLoad 
                    DATA_SHIFTED  = unittest[:-SHFT, :, idx]

                    GOV_SHFT_LOAD = DATA_SHIFTED@PS@load_coeffifient_valveAfterLoad[i,:]
                     
                    GOV_SHFT_UNITSTEP = v_limit_value[i] * DATA_SHIFTED[:, 0]
                    Pm_limit[SHFT:]   += alpha[idx] * (GOV_SHFT_UNITSTEP - GOV_SHFT_LOAD)
            

        elif status == VALVE_LIMIT_VIOLATION:

            vcoeff = v_coeffifient_total.loc[index_limit_new_valve, :].to_numpy()
            v_limit_value       = lim_u[index_limit_new_valve]

            PS = t_limit**K * FACT_MAT

            for idx, i in enumerate(index_limit_new_valve):

                
                ''' 3 Dimensional Array '''
                DATA = unittest[:time_fit_len, :, i]
        
                SHFT = 0 if limit_start == 1 else limit_start

                GOV_SHFT_UNITSTEP = DATA[:-SHFT, 0] * v_limit_value[i]
                GOV_SHFT_VALVE    = DATA[:-SHFT, :]@PS@vcoeff[idx, :]


                Pm_limit        += alpha[i] * (DATA @ vcoeff[idx])
                Pm_limit[SHFT:] += alpha[i] * (GOV_SHFT_UNITSTEP - GOV_SHFT_VALVE)


    ''' END ROUTINE'''

    # Initialize parameters and variables
    test_f0 = 1
    
    test_gov = np.empty((time_fit_len, mask_index_nolimit.sum()))

    # Calculate `test_test_gov` using `v_coeffifient_limit`
    vcoeff = v_coeffifient_total.loc[mask_index_nolimit, :].to_numpy()

    for i, idx in enumerate(mask_index_nolimit[mask_index_nolimit].index):
        test_gov[:, i] = unittest[:time_fit_len, :, idx] @ vcoeff[i]

    test_times = time_fit[time_fit < t_n_star + 1]

    test_f  = np.empty_like(test_times)
    test_Pm = np.empty_like(test_times)

    dt = test_times[1] - test_times[0]
    for i, t in enumerate(test_times):

        # Calculate `test_gov` at each time step
        # and       `test_Pm` based on presence of limits
        test_Pm_val = np.sum(alpha[mask_index_nolimit] * test_gov[i, :])
        if len(index_limit) != 0:
            test_Pm_val += Pm_limit[i]

        # Frequency difference calculation
        test_f_diff = 1 / (2 * Hsys) * (test_Pm_val - P_event)

        # Calculate frequency at time t
        if t == 0: test_f[i] = test_f0
        else:      test_f[i] = test_f[i-1] + test_f_diff * dt
        
        test_Pm[i] = test_Pm_val


    # Calculate f_nadir
    f_nadir = min(test_f) * 60

    # Output the results
    print("t_n_star:", t_n_star)
    print("f_nadir:", f_nadir)

''' Parameters '''
event_gen = pd.DataFrame({
    'BusNum' : [7098, 7099],
    'GenID'  : [1   , 1   ]
})

output_gen_info_org['TSVmax']     = 0.87
output_gov_gasInfo_org['TSLdref'] = 0.835
numSteam = len(output_gen_info_org[output_gen_info_org['Type'] == 'steam'])
gen_data_active, gov_gasInfo  = datagen.DataGenerate(gen_data, output_gen_info_org, output_gov_gasInfo_org, event_gen)

''' Run and Time Program'''

start_time = time.time()
run_test(order, t_sim, ramp, ramp_v, unittest_gas, unittest, gen_data_active, gov_gasInfo, numSteam)
end_time   = time.time()

print("Execution time:", end_time - start_time, "seconds")
    