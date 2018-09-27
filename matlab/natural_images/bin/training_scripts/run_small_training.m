%Train an image model with lower data dimensionality and fewer neurons
%% CNCE
estimator_cnce = run_L1_training('CNCE', 25, 160, 160);
run_L2_training(estimator_cnce, 40);

%% NCE
estimator_nce = run_L1_training('NCE', 25, 160, 160);
run_L2_training(estimator_nce, 40);
