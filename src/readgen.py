'''
Use this code to read generator parameters from a CSV file.
'''


import pandas as pd

def get_gen_params(fname_steam, fname_gas, fname_gen):


    gov_data_steam = pd.read_csv(fname_steam)
    gov_data_gas = pd.read_csv(fname_gas)
    gen_data = pd.read_csv(fname_gen)

    # Add 'Type' column to differentiate steam and gas governors
    gov_data_steam['Type'] = 'steam'
    gov_data_gas['Type'] = 'gas'

    # Concatenate steam and gas data into one DataFrame
    gen_info = pd.concat([gov_data_steam[['BusNum', 'GenID', 'TSPmax', 'TSPmin', 'Type']], 
                          gov_data_gas[['BusNum', 'GenID', 'TSVmax', 'TSVmin', 'Type']]]).reset_index(drop=True)
    
    return gen_info, gov_data_gas, gen_data

