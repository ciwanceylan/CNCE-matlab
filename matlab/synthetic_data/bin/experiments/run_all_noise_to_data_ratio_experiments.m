%% Run all the noise-to-data sample ratio experiments

%% Gaussian model
setupGauss = config_noise_to_data_ratio('gauss', 5);
resGauss = run_cnce_experiment(setupGauss);

% Results and set up are saved and can be plotted using command below
% plot_noise_data_ratio_results(setupGauss, resGauss);

%% ICA model
setupICA = config_noise_to_data_ratio('ICA', 4);
resICA = run_cnce_experiment(setupICA);

% plot_noise_data_ratio_results(setupICA, resICA);

%% Ring (lower-dimensional manifold) model
setupRing = config_noise_to_data_ratio('ring', 5);
resRing = run_cnce_experiment(setupRing);

% plot_noise_data_ratio_results(setupRing, resRing);

%% Lognormal model
setupLognormal = config_noise_to_data_ratio('lognormal', 1);
resLognormal = run_cnce_experiment(setupLognormal);

% plot_noise_data_ratio_results(setupLognormal, resLognormal);

%% Bernoulli model
setupBer = config_noise_to_data_ratio('bernoulli', 1);
resBer = run_cnce_experiment(setupBer);

% plot_noise_data_ratio_results(setupBer, resBer);
