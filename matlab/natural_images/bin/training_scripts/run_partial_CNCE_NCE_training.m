
%% CNCE
estimator_cnce = run_L1_training('CNCE', 32, 600, 100);
run_L2_training(estimator_cnce, 40);


%% NCE
% NCE often doesn't work with too many neurons, so we use 100 instead of
% 600
estimator_nce = run_L1_training('NCE', 32, 600, 100);
run_L2_training(estimator_nce, 40);
