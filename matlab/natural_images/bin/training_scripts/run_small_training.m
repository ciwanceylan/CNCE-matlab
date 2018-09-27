%% CNCE
estimator_cnce = run_L1_training('CNCE', 25, 160, 100);
run_L2_training(estimator_cnce, 30);

%% NCE
estimator_nce = run_L1_training('NCE', 25, 160, 100);
run_L2_training(estimator_nce, 30);
