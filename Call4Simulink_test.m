function output = Call4Simulink_test(IEEEG1, GGOV1, order, t_sim)
%% Load data from CSV files
% IEEEG1 and GGOV1 are file paths for CSV files with data in table format
data_IEEEG1 = readtable(IEEEG1, 'VariableNamingRule', 'preserve');
data_GGOV1 = readtable(GGOV1, 'VariableNamingRule', 'preserve');

%% Extract parameters from data tables
% IEEEG1 parameters
index_steam = table2array(data_IEEEG1(:, 'BusNum'))';
K_ieeeg1 = table2array(data_IEEEG1(:, 'TSK'))';
T1_ieeeg1 = table2array(data_IEEEG1(:, 'TST:1'))';
T2_ieeeg1 = table2array(data_IEEEG1(:, 'TST:2'))';
T3_ieeeg1 = table2array(data_IEEEG1(:, 'TST:3'))';
T4_ieeeg1 = table2array(data_IEEEG1(:, 'TST:4'))';
T5_ieeeg1 = table2array(data_IEEEG1(:, 'TST:5'))';
T6_ieeeg1 = table2array(data_IEEEG1(:, 'TST:6'))';
T7_ieeeg1 = table2array(data_IEEEG1(:, 'TST:7'))';
K1_ieeeg1 = table2array(data_IEEEG1(:, 'TSK:1'))';
K3_ieeeg1 = table2array(data_IEEEG1(:, 'TSK:3'))';
K5_ieeeg1 = table2array(data_IEEEG1(:, 'TSK:5'))';
K7_ieeeg1 = table2array(data_IEEEG1(:, 'TSK:7'))';

% GGOV1 parameters
index_gas = table2array(data_GGOV1(:, 'BusNum'))';
R_ggov1 = table2array(data_GGOV1(:, 'TSR'))';
Kp_ggov1 = table2array(data_GGOV1(:, 'TSKpgov'))';
Ki_ggov1 = table2array(data_GGOV1(:, 'TSKigov'))';
Tact_ggov1 = table2array(data_GGOV1(:, 'TSTact'))';
wfnl_ggov1 = table2array(data_GGOV1(:, 'TSWfnl'))';
Kt_ggov1 = table2array(data_GGOV1(:, 'TSKturb'))';
Tb_ggov1 = table2array(data_GGOV1(:, 'TSTb'))';
Kpload_ggov1 = table2array(data_GGOV1(:, 'TSKpload'))';
Kiload_ggov1 = table2array(data_GGOV1(:, 'TSKiload'))';
Tsa_ggov1 = table2array(data_GGOV1(:, 'TSTsa'))';
Tsb_ggov1 = table2array(data_GGOV1(:, 'TSTsb'))';
Tf_ggov1 = table2array(data_GGOV1(:, 'TSTfload'))';
Ldref_ggov1 = table2array(data_GGOV1(:, 'TSLdref'))';

%% Simulation settings
mdl = "IEEEG1_unitStepResponse";
t_sim = double(t_sim);  % Ensure t_sim is a double
time10 = linspace(0, t_sim, 1000); % Time vector for interpolation

% Initialize result arrays
ramp_ieeeg1 = [];
ramp_v_ieeeg1 = [];
unitTest_ieeeg1 = [];

%% Run simulations for each generator
for i = 1:length(index_steam)
    % Prepare simulation input
    simIn = Simulink.SimulationInput(mdl);
    simIn = simIn.setVariable('K', K_ieeeg1(i));
    simIn = simIn.setVariable('T1', T1_ieeeg1(i));
    simIn = simIn.setVariable('T2', T2_ieeeg1(i));
    simIn = simIn.setVariable('T3', T3_ieeeg1(i));
    simIn = simIn.setVariable('T4', T4_ieeeg1(i));
    simIn = simIn.setVariable('T5', T5_ieeeg1(i));
    simIn = simIn.setVariable('T6', T6_ieeeg1(i));
    simIn = simIn.setVariable('T7', T7_ieeeg1(i));
    simIn = simIn.setVariable('K1', K1_ieeeg1(i));
    simIn = simIn.setVariable('K3', K3_ieeeg1(i));
    simIn = simIn.setVariable('K5', K5_ieeeg1(i));
    simIn = simIn.setVariable('K7', K7_ieeeg1(i));

    % Set StopTime in Simulation Input object
    simIn = simIn.setModelParameter('StopTime', num2str(t_sim)); % Correct StopTime setting

    % Run the simulation
    out = sim(simIn);

    % Allocate space for Simulink output
    unitTest_simulink = zeros(length(out.tout), order + 2);

    % Extract outputs and interpolate data
    for j = 1:order+2
        unitTest_simulink(:, j) = out.yout{j+2}.Values.Data;
    end
    ramp_ieeeg1(:, i) = interp1(out.tout, out.yout{1}.Values.Data, time10, 'linear');
    ramp_v_ieeeg1(:, i) = interp1(out.tout, out.yout{2}.Values.Data, time10, 'linear');
    unitTest_ieeeg1(:, :, i) = interp1(out.tout, unitTest_simulink, time10, 'linear');
end


%% Simulation Settings
mdl = "GGOV1_unitStepResponse";

ramp_ggov1=[];
ramp_v_ggov1=[];
unitTest_ggov1=[];
unitTestLoadlimit_ggov1=[];

%% Run Simulink for Unit Step Response of Generators
for i=1:length(index_gas)
    % Prepare simulation input
    unitTest_simulink=[];
    unitTestLoadlimit_simulink=[];

    simIn = Simulink.SimulationInput(mdl);
    simIn = simIn.setVariable('R', R_ggov1(i));
    simIn = simIn.setVariable('Kp', Kp_ggov1(i));
    simIn = simIn.setVariable('Ki', Ki_ggov1(i));
    simIn = simIn.setVariable('Tact', Tact_ggov1(i));
    simIn = simIn.setVariable('wfnl', wfnl_ggov1(i));
    simIn = simIn.setVariable('Kt', Kt_ggov1(i));
    simIn = simIn.setVariable('Tb', Tb_ggov1(i));
    simIn = simIn.setVariable('Kpload', Kpload_ggov1(i));
    simIn = simIn.setVariable('Kiload', Kiload_ggov1(i));
    simIn = simIn.setVariable('Tsa', Tsa_ggov1(i));
    simIn = simIn.setVariable('Tsb', Tsb_ggov1(i));
    simIn = simIn.setVariable('Tf', Tf_ggov1(i));

    % Set StopTime in Simulation Input object
    simIn = simIn.setModelParameter('StopTime', num2str(t_sim)); % Correct StopTime setting

    % Run the simulation
    out = sim(simIn);

    for j=1:order+2
        unitTest_simulink(:,j) = out.yout{j+2}.Values.Data;
    end

    for j=1:order+2
        unitTestLoadlimit_simulink(:,j) = out.yout{j+2+order+2}.Values.Data;
    end

    % Interpolating Simulink output
    ramp_ggov1(:, i) = interp1(out.tout, out.yout{1}.Values.Data, time10, 'linear');
    ramp_v_ggov1(:, i) = interp1(out.tout, out.yout{2}.Values.Data, time10, 'linear');
    unitTest_ggov1(:, :, i) = interp1(out.tout, unitTest_simulink, time10, 'linear');
    unitTestLoadlimit_ggov1(:, :, i) = interp1(out.tout, unitTestLoadlimit_simulink, time10, 'linear');
end

%Collected unit step response
ramp=[ramp_ieeeg1, ramp_ggov1];
ramp_v=[ramp_v_ieeeg1, ramp_v_ggov1];
unitTest = cat(3, unitTest_ieeeg1, unitTest_ggov1);

% Set output to be the final results (modify this as needed)
output = struct('ramp', ramp, 'ramp_v', ramp_v, 'unitTest', unitTest, 'unitTestLoadlimit_ggov1',unitTestLoadlimit_ggov1 );

end
