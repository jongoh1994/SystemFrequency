

'''
Use this code to write the generator parameters from PowerWorld to CSV so they only need to be read once.
This way we don't have to call power world everytime.
'''

from gridwb.workbench import GridWorkBench

def write_gen_params(CASE, fname_steam, fname_gas, fname_gen):
    '''Run this with Wb in py 3.11'''

    wb = GridWorkBench(CASE) 
    esa = wb.io.esa

    # Not neccessary
    esa.SolvePowerFlow('Flatstart')


    # STEAM
    gen_steam_key_fields = esa.get_key_field_list('Governor_IEEEG1')
    params_steam = gen_steam_key_fields + ['TSK', 'TSK:1', 'TSK:3', 'TSK:5', 'TSK:7', 
                                           'TST:1', 'TST:2', 'TST:3', 'TST:4', 'TST:5', 'TST:6', 'TST:7',
                                           'TSPmax', 'TSPmin']
    gov_data_steam = esa.GetParametersMultipleElement(ObjectType='Governor_IEEEG1', ParamList=params_steam)

    # GAS
    gen_gas_key_fields = esa.get_key_field_list('Governor_GGOV1')
    params_gas = gen_gas_key_fields + ['TSR', 'TSKpgov', 'TSKigov', 'TSTact', 'TSKturb', 
                                       'TSWfnl', 'TSTb', 'TSTfload', 'TSKpload', 'TSKiload', 'TSLdref', 'TSTsa', 'TSTsb', 
                                       'TSVmax', 'TSVmin']
    gov_data_gas = esa.GetParametersMultipleElement(ObjectType='Governor_GGOV1', ParamList=params_gas)

    # GENERATOR INFO
    params =  esa.get_key_field_list('gen') + ['GenMVABase', 'TSH', 'GenStatus', 'TSGenGovernorName','GenMW']
    gen_data = esa.GetParametersMultipleElement(ObjectType='gen', ParamList=params)


    # Save data to CSV files
    gov_data_steam.to_csv(fname_steam, index=False)
    gov_data_gas.to_csv(fname_gas, index=False)
    gen_data.to_csv(fname_gen, index=False)


# Call this file when CSV of generator parameters needs to be updated
write_gen_params(r".\data\case1_Texas2000_test.pwb", 'data/IEEEG1_data.csv', 'data/GGOV1_data.csv', 'data/GEN_data.csv')
