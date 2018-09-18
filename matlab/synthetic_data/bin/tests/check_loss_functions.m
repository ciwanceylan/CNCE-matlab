%% Run gradient checks for both loss functions using Gaussian model
% Warinings will be displayed if the analytical and nummerical gradients
% do not match.
clear; clc;

%% Gaussian model -- CNCE loss
N = 10;
D = 5;
kappa = 2;

theModel = @gaussModel;
noiseBase = gNoiseBase( N, D, kappa);
epsilonBase = 1;
[X, theta_gt, theta_init] = getData(10, 5, 'gauss', 100);

[ U, logPyx_ratio, epsilon] = ...
    gNoise(X, theModel, theta_init, epsilonBase, noiseBase, 0);

[d, ~, ~] = checkgrad_cnce(U, theta_init, theModel, 1e-7, logPyx_ratio);
if d > 1e-6
    warning('Gradients for cnce_loss might be wrong')
    fprintf('Relative graident errors are: ')
    disp(d)
end

[d, dy_cnce, dh_cnce] = checkgrad_cnce(U, theta_gt, theModel, 1e-7, logPyx_ratio);
if d > 1e-6
    warning('Gradients for cnce_loss might be wrong')
    fprintf('Relative graident errors are: ')
    disp(d)
end

fprintf('cnce loss function check done!\n')

%% Gaussian model -- NCE loss
N = 10;
D = 5;
nu = 2;

theModel = @gaussModel;
[X, theta_gt, theta_init] = getData(10, 5, 'gauss', 100);

[ U, logPU ] = gNCEnoise( X, nu );

[d, ~, ~]  = checkgrad_nce(U, theta_init, theModel, 1e-7, logPU, nu);
if d > 1e-6
    warning('Gradients for nce_loss might be wrong')
    fprintf('Relative graident errors are: ')
    disp(d)
end

[d, dy_nce, dh_nce]  = checkgrad_nce(U, theta_gt, theModel, 1e-7, logPU, nu);
if d > 1e-6
    warning('Gradients for nce_loss might be wrong')
    fprintf('Relative graident errors are: ')
    disp(d)
end

fprintf('nce loss function check done!\n')