%% Ring_data_visualisation
% Shows the PDF for the lower-dimensional manifold distribution (the "ring
% model") in the 2D case. Also plots histograms of the CNCE and NCE noise
% respectively.
clear; clc; close all;

global CNCE_RESULTS_FOLDER

savedir = fullfile(CNCE_RESULTS_FOLDER, 'figures', 'ring_data_visualisation');
mysave = @(name) cnceSaveFig(savedir, name);

rng(42)  % for full reproducability

% Data setup
N = 8000;
D = 2;
dataType = 'ring';
[ theModel, M] = getModel(dataType, D);
theModelNCE = @ringModelNCE;
R = [randn(N,1), rand(N, 1)];
muTrue = 6;
precTrue = 1;
R = bsxfun(@times, R, [1/sqrt(precTrue), 2*pi]);
R = bsxfun(@plus, R, [muTrue, 0]);
X = [R(:, 1) .* cos(R(:,2)),  R(:, 1) .* sin(R(:,2))];

% Noise setup
nu = 2;
kappa = nu;
thetaInit = [2, 0.6];
thetaInitNCE = [thetaInit, -7.5];
useaCNCE = 0;

[ Unce, logPnce ] = gNCEnoise( X, nu ); %NCE
Ynce = Unce(N+1:end, :);
% phiSample = exp(theModel(Ynce(1:30), [muTrue, precTrue]));
noiseBase = randn(N, D, kappa);
[ Ucnce, logPcnce, epsilon ] = ...
    gNoise(X, theModel, thetaInit, 1, noiseBase, useaCNCE); %CNCE
Ycnce1 = Ucnce(:, 1, 2:end);
Ycnce2 = Ucnce(:, 2, 2:end);
Ycnce = [Ycnce1(:), Ycnce2(:)];
clearvars Ycnce1 Ycnce2


%% Ploting

addColors % add some nice colors
histAxes = 10 *[-1, 1, -1, 1];
colormap jet;
% Surface
binW = 0.5;
[Xgrid, Ygrid] = meshgrid(-10:binW:10, -10:binW:10);
phi = exp(theModel([Xgrid(:), Ygrid(:)], [muTrue, precTrue])) .* (sqrt(precTrue/(2*pi)));
phi = reshape(phi, size(Xgrid));

% Data distr
surf(Xgrid, Ygrid, phi)
axis(histAxes)
view(-30, 75)
mysave('ring_distr_volume')
axis image
view(2)
mysave('ring_distr_top_down')

figure
hNce = histogram2(Ynce(1:N, 1), Ynce(1:N, 2), [25, 25], 'Normalization', 'Probability');
hNce.FaceColor = 'flat';
hNce.BinWidth = [binW, binW];
axis(histAxes)
view(-30, 75)
mysave('ring_nce_volume')
axis image
axis(histAxes)
view(2)
mysave('ring_nce_top_down')

figure
hCnce = histogram2(Ycnce(1:N, 1), Ycnce(1:N, 2), [25, 25], 'Normalization', 'Probability');
hCnce.FaceColor = 'flat';
hCnce.BinWidth = [binW, binW];
axis(histAxes)
view(-30, 75)
mysave('ring_cnce_volume')
axis image
view(2)
axis(histAxes)
mysave('ring_cnce_top_down')


% Contour
binW = 0.1;
[Xgrid, Ygrid] = meshgrid(-10:binW:10, -10:binW:10);
phi = exp(theModel([Xgrid(:), Ygrid(:)], [muTrue, precTrue])) .* (sqrt(precTrue/(2*pi)));
phi = reshape(phi, size(Xgrid));
figure
contour(Xgrid, Ygrid, phi, 15)
axis image
axis(histAxes)
mysave('ring_distr_contour')

% figure
nSamples = 100;
hold on
hnce = scatter(Ynce(1:nSamples, 1), Ynce(1:nSamples, 2), 22, c.kthRed, 's', 'filled');
hcnce = scatter(Ycnce(1:nSamples, 1), Ycnce(1:nSamples, 2), 22, c.kthBlue, 'filled');
axis image
axis(histAxes)
legend([hcnce, hnce], {'CNCE', 'NCE'})
mysave('ring_distr_contour_with_noise')
