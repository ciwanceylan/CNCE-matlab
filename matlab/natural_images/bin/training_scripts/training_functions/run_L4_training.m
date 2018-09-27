function estimator = run_L4_training(estimator, nNeurons)
%run_L4_training Runs the training of the 4th layer for the neural image
%model. 
% Input:
%   estimator can be either a CNCEstimator or at NCEstimator with L3 trained.
%   nNeurons is the number of neurons in the second layer.


% Setup for fourth layer.
newIndices = {[], ':'};
newLayerDim = nNeurons;
newActivator = 'LinearLinear';

cnceOpt.nmIter = 20;
cnceOpt.kappa = 6;
cnceOpt.batchSize = 10e4;
cnceOpt.N = 10e4;
cnceOpt.aCNCE = 0;
cnceOpt.epsBase = 0.5;
cnceOpt.id = ['L4_', 'n', num2str(nNeurons)];
cnceOpt.epsHardCap = 8;
cnceOpt.epochSize = 8;  % frequency to resample patches. 
                        % If epochSize > nmIter patches will only be sampled once
cnceOpt.noiseType = 'Extra';   

estimator.setCnceOpt(cnceOpt);

% Upgrade 3rd later to a double layer with both feature extraction and
% pooling
estimator.turnSingleToDouble(2, newLayerDim, newIndices, newActivator);

% Run estimation
estimator.estimate(2);

% Remove intermediate trash variables.
estimator.clearHidden();

end

