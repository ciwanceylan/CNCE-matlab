%% CNCE
estimator_cnce = run_L1_training('CNCE', 32, 600, 600);
run_L2_training(estimator_cnce, 200);
run_L3_training(estimator_cnce, 60)
run_L4_training(estimator_cnce, 30)
