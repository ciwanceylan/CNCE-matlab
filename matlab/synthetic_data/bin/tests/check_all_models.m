%% Run gradient checks for all models
% Warinings will be displayed if the analytical and nummerical gradients
% do not match.
clear; clc;

%% Gaussian model
theModel = @gaussModel;
[X, theta_gt, theta_init] = getData(10, 5, 'gauss', 100);

for k = 1:size(X,1)
    x = X(k, :);
    [d_t, d_u]  = checkModelGrad(x, theta_gt, theModel, 1e-7);
    if d_t > 1e-6 || d_u > 1e-6
        warning('Gradients for Gaussian model might be wrong')
        fprintf('Relative graident errors are: ')
        disp([d_t, d_u])
    end
end
fprintf('Gaussian model check done!\n')

%% ICA model
theModel = @icaModel;
[X, theta_gt, theta_init] = getData(10, 4, 'ICA', 100);

for k = 1:size(X,1)
    x = X(k, :);
    [d_t, d_u]  = checkModelGrad(x, theta_gt, theModel, 1e-7);
    if d_t > 1e-6 || d_u > 1e-6
        warning('Gradients for ICA model might be wrong')
        fprintf('Relative graident errors are: ')
        disp([d_t, d_u])
    end
end
fprintf('ICA model check done!\n')

%% lognormal model
theModel = @lognormalModel;
[X, theta_gt, theta_init] = getData(10, 1, 'lognormal', 100);

for k = 1:size(X,1)
    x = X(k, :);
    [d_t, d_u]  = checkModelGrad(x, theta_gt, theModel, 1e-7);
    if d_t > 1e-6 || d_u > 1e-6
        warning('Gradients for lognormal model might be wrong')
        fprintf('Relative graident errors are: ')
        disp([d_t, d_u])
    end
end
fprintf('Lognormal model check done!\n')

%% Bernoulli model
theModel = @berModel;
[X, theta_gt, theta_init] = getData(10, 1, 'bernoulli', 100);

for k = 1:size(X,1)
    x = X(k, :);
    [d_t, d_u]  = checkModelGrad(x, theta_gt, theModel, 1e-7);
    if d_t > 1e-6 || d_u > 1e-6
        warning('Gradients for Bernoulli model might be wrong')
        fprintf('Relative graident errors are: ')
        disp([d_t, d_u])
    end
end
fprintf('Bernoulli model check done!\n')


%% Ring model
theModel = @ringModel;
[X, theta_gt, theta_init] = getData(10, 5, 'ring', 100);

for k = 1:size(X,1)
    x = X(k, :);
    [d_t, d_u]  = checkModelGrad(x, theta_gt, theModel, 1e-7, [2]);
    if d_t > 1e-6 || d_u > 1e-6
        warning('Gradients for ting model might be wrong')
        fprintf('Relative graident errors are: ')
        disp([d_t, d_u])
    end
end
fprintf('Ring model check done!\n')