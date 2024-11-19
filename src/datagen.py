
import numpy as np

def DataGenerate(gen_data_input, gen_info_input, gov_gasInfo_input, event_gen_input):
    # Perform anti-join to filter out rows that have matching 'BusNum' and 'GenID'
    gen_data_input = gen_data_input.merge(event_gen_input, on=['BusNum', 'GenID'], how='left', indicator=True)
    gen_data_input = gen_data_input[gen_data_input['_merge'] == 'left_only']

    # Drop the helper column '_merge'
    gen_data_input = gen_data_input.drop(columns=['_merge']).reset_index(drop=True)

    gen_data_output = gen_data_input[
        (gen_data_input['GenStatus'] == 'Closed') & 
        ((gen_data_input['TSGenGovernorName'] == 'GGOV1') | (gen_data_input['TSGenGovernorName'] == 'IEEEG1'))
    ]

    # Perform an inner merge to find matching rows between output_gen_info and gen_data_active
    matches = gen_info_input.merge(gen_data_output[['BusNum', 'GenID']], on=['BusNum', 'GenID'], how='inner')

    # Get the indices in output_gen_info that match with gen_data_active
    matches['Index'] = gen_info_input.index[gen_info_input[['BusNum', 'GenID']].apply(tuple, axis=1).isin(matches[['BusNum', 'GenID']].apply(tuple, axis=1))]

    # Merge gen_data_active with matches to get the 'TSPmax' values where BusNum and GenID match
    gen_data_output = gen_data_output.merge(matches[['BusNum', 'GenID', 'TSPmax', 'TSVmax','Index']], on=['BusNum', 'GenID'], how='left')

    # Add 'valve_lim_u' column based on TSGenGovernorName, using TSPmax for 'IEEEG1' and TSVmax for 'GGOV1'
    gen_data_output['valve_lim_u'] = np.where(gen_data_output['TSGenGovernorName'] == 'IEEEG1',
                                              gen_data_output['TSPmax'],
                                              np.where(gen_data_output['TSGenGovernorName'] == 'GGOV1',
                                                       gen_data_output['TSVmax'],
                                                       np.nan))  # Default to NaN if no match

    # Optional: Drop the 'TSPmax' column if it's only needed for the new 'valve_lim_u' column
    gen_data_output = gen_data_output.drop(columns=['TSPmax', 'TSVmax'])

    # Add index 
    gov_gasInfo_output = gov_gasInfo_input.merge(gen_data_output[['BusNum', 'GenID', 'Index', 'GenMW', 'GenMVABase']], on=['BusNum', 'GenID'], how='inner')
    gov_gasInfo_output.index=gov_gasInfo_output['Index']



    # For the case where the governor is 'IEEEG1'
    gen_data_output.loc[gen_data_output['TSGenGovernorName'] == 'IEEEG1', 'valve_lim_u'] = \
        gen_data_output['valve_lim_u'] - (gen_data_output['GenMW'] / gen_data_output['GenMVABase'])

    # For the case where the governor is 'GGOV1'
    gen_data_output.loc[gen_data_output['TSGenGovernorName'] == 'GGOV1', 'valve_lim_u'] = \
        gen_data_output.loc[gen_data_output['TSGenGovernorName'] == 'GGOV1', 'valve_lim_u'] - \
        (gen_data_output.loc[gen_data_output['TSGenGovernorName'] == 'GGOV1', 'GenMW'] / 
         gen_data_output.loc[gen_data_output['TSGenGovernorName'] == 'GGOV1', 'GenMVABase']) / \
        gov_gasInfo_output.loc[gov_gasInfo_output['Index'].isin(gen_data_output.loc[gen_data_output['TSGenGovernorName'] == 'GGOV1', 'Index']), 'TSKturb'].values - \
            gov_gasInfo_output.loc[gov_gasInfo_output['Index'].isin(gen_data_output.loc[gen_data_output['TSGenGovernorName'] == 'GGOV1', 'Index']), 'TSWfnl'].values


    gen_data_output.index=gen_data_output['Index']
    
    gen_data_output = gen_data_output.drop(columns=['Index'])
    gov_gasInfo_output = gov_gasInfo_output.drop(columns=['Index'])


    
    return [gen_data_output, gov_gasInfo_output]
    

