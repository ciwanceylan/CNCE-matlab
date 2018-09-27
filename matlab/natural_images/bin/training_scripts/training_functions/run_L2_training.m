function estimator = run_L2_training(estimator, nNeurons)
%run_L2_training Runs the training of the 2nd layer for the neural image
%model. 
% Input:
%   estimator can be either a CNCEstimator or at NCEstimator with L1 trained.
%   nNeurons is the number of neurons in the second layer.


% Setup for second layer.
newIndices = {[], ':'};
newLayerDim = nNeurons;
newActivator = 'LinearQuadraticLog'; % choice of activation function

cnceOpt.nmIter = 20;
cnceOpt.kappa = 6;
cnceOpt.batchSize = 10e4;
cnceOpt.N = 20e4;
cnceOpt.aCNCE = 0;
cnceOpt.epsBase = 0.5;
cnceOpt.id = ['L2_', 'n', num2str(nNeurons)];
cnceOpt.noiseType = 'Extra';
cnceOpt.epochSize = 21;     % frequency to resample patches. 
                            % If epochSize > nmIter patches will only be sampled once

estimator.setCnceOpt(cnceOpt);

% Upgrade 1st later to a double layer with both feature extraction and
% pooling
estimator.turnSingleToDouble(1, newLayerDim, newIndices, newActivator);

% Run estimation
estimator.estimate(1);

% Remove intermediate trash variables.
estimator.clearHidden();

end