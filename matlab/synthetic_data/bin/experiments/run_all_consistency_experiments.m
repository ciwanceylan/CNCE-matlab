%% Run all the consistency experiments

%% Gaussian model
setupGauss = config_consistency('gauss', 5);
resGauss = run_cnce_experiment(setup);

% Results and set up are saved and can be plotted using command below
% plot_consistency_results(setupGauss, resGauss);

%% ICA model
setupICA = config_consistency('ICA', 4);
resICA = run_cnce_experiment(setup);

% plot_consistency_results(setupICA, resICA);

%% Ring (lower-dimensional manifold) model
setupRing = config_consistency('ring', 5);
resRing = run_cnce_experiment(setup);

% plot_consistency_results(setupRing, resRing);

%% Lognormal model
setupLognormal = config_consistency('lognormal', 1);
resLognormal = run_cnce_experiment(setup);

% plot_consistency_results(setupLognormal, resLognormal);

%% Bernoulli model
setupBernoulli = config_consistency('bernoulli', 1);
resBer = run_cnce_experiment(setup);

% plot_consistency_results(setupBer, resBer);
