function estimator = run_L1_training(method, winsize, pca_dims, nNeurons)
%run_L1_training Runs the training of the 1nd layer for the neural image
%model. 
% Input:
%   method: either 'CNCE' or 'NCE'
%   pca_dims: The number of image PCA dimensions. Needs to be in interval
%   [1, 32^2]
%   nNeurons: the number of neurons in the second layer.

if ~(pca_dims > 0 && pca_dims < 32^2 + 1)
    error('pca_dims must lie in interval [%d, %d].', 1, 32^2);
end

% Data setup
dataOpt.dataset = 'VanHateren';
dataOpt.rDim = pca_dims;
dataOpt.winsize = winsize;
dataOpt.N = 20e4;
dataOpt.rngseed = 8148;

% Model setup
modelOpt.L 	= 1;
modelOpt.activators = {'LinearLog'};
modelOpt.dimensions = {[dataOpt.rDim, nNeurons]};
modelOpt.indices = {':'};
modelOpt.types = {'SingleNeuronLayer'};

% Estimation setup
cnceOpt.nmIter = 40;
cnceOpt.kappa = 6;
cnceOpt.batchSize = 10e4;
cnceOpt.aCNCE = 0;
cnceOpt.epsBase = 0.5;
cnceOpt.id = ['L1_', 'n', num2str(nNeurons)];
cnceOpt.noiseType = 'Extra'; 

optimizerOpt.alg = 'minimize';

% Set seed for reproducability
rng(dataOpt.rngseed)

if strcmp(method, 'NCE')
    estimator = NCEstimator({dataOpt, cnceOpt, modelOpt, optimizerOpt});
    estimator.saveResult([estimator.savename, '0_T', datestr(now, 'HH_MM_SS')]) % save the initial parameters
    estimator.estimate(1);

elseif strcmp(method, 'CNCE')
    estimator = CNCEstimator({dataOpt, cnceOpt, modelOpt, []});
    estimator.saveResult([estimator.savename, '0_T', datestr(now, 'HH_MM_SS')]) % save the initial parameters
    estimator.estimate(1);
end

% Remove intermediate trash variables.
estimator.clearHidden();

end