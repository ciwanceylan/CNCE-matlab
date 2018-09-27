function estimator = run_L3_training(estimator, nNeurons)
%run_L3_training Runs the training of the 2nd layer for the neural image
%model. 
% Input:
%   estimator can be either a CNCEstimator or at NCEstimator with L2 trained.
%   nNeurons is the number of neurons in the second layer.


% L3 - add dataProcessors{2} for PCA whitening between layer 2 and 3 (gain
% control)
rDim = 196; % replace with [] to plot curve and choose manually 
estimator.addInterLayer(2, rDim)

init = {[estimator.dataProcessors{2}.rDim, nNeurons], ':', 'Max'};
snl3 = SingleNeuronLayer(init);
estimator.addNeuronLayer(snl3);
estimator.show();

cnceOpt.nmIter = 20;
cnceOpt.kappa = 6;
cnceOpt.N = 15e4;
cnceOpt.batchSize = 15e4;
cnceOpt.aCNCE = 0;
cnceOpt.epsBase = 0.5;
cnceOpt.id = 'L3';
cnceOpt.epochSize = 8;
cnceOpt.noiseType = 'Extra';

estimator.setCnceOpt(cnceOpt);

estimator.estimate(2)

% Remove intermediate trash variables.
estimator.clearHidden();
end

