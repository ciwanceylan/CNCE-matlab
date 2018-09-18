%% Run all the noise-to-data sample ratio experiments

%% Gaussian model
setup = config_noise_to_data_ratio('gauss', 5);
resGauss = run_cnce_experiment(setup);

% Results and set up are saved and can be plotted using command below
% plot_noise_data_ratio_results(setupGauss, resGauss);

%% ICA model
setup = config_noise_to_data_ratio('ICA', 4);
resICA = run_cnce_experiment(setup);

% plot_noise_data_ratio_results(setupICA, resICA);

%% Ring (lower-dimensional manifold) model
setup = config_noise_to_data_ratio('ring', 5);
resRing = run_cnce_experiment(setup);

% plot_noise_data_ratio_results(setupRing, resRing);

%% Lognormal model
setup = config_noise_to_data_ratio('lognormal', 1);
resLognormal = run_cnce_experiment(setup);

% plot_noise_data_ratio_results(setupLognormal, resLognormal);

%% Bernoulli model
setup = config_noise_to_data_ratio('bernoulli', 1);
resBer = run_cnce_experiment(setup);

% plot_noise_data_ratio_results(setupBer, resBer);

