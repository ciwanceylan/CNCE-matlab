classdef CNCEstimator < handle
    %CNCESTIMATOR Estimator using CNCE for a multilayer image model.
    % CNCEstimator inherits from handle so that it can be passed by reference

    properties
		neuronLayers 			% {NeuronLayer1, NeuronLayer2, ...} Each layer can be a 
								%	SingleNeuronLayer or a DoubleNeuronLayer
		dataProcessors 			% For PCA transforms and sample data
		dataHandle 				% The current data used for training
		noiseHandle 			% The current noise used for training
		noiseProcessor			% For sampling noise
		cnceOpt					% Options for estimation (batchSize, noise options etc)
		dataOpt					% Options for sampling the data (dataset and total data amount etc)
		modelOpt				% Options for the image model
		optimizerOpt			% Options for the gradient descent optimiser
		stats					% For storing estimation statistics
		savename				% File name base for savning results
    end

	properties (Hidden = true)
		cIt						% The current iteration (for restarting estimation)
		epoch					% The current epoch (each epoch the data is resampled)
		theta					% The current model parameters as a vector
	end

    methods
        %------------------------------------------------------------
		% ============= Estimation and model evaluation =============
		%____________________________________________________________
		function estimate(obj, layer)
			% estimate Run estimator for given layer
			% 	Repeat until done:
			%		1. Get data
			%		2. Update noise
			%		3. Run optimisation
			startIter = obj.cIt;
			endIter = obj.cIt + obj.cnceOpt.nmIter - 1;
			% Set when to get new data batch
			if obj.dataProcessors{layer}.N == obj.cnceOpt.batchSize
				nDataResample = obj.cnceOpt.epochSize;
			else
				nDataResample = 1;
			end
			obj.addDataFun(layer); % Make sure to have a function for sampling
			for it = startIter : endIter
				fprintf('Starting meta-iteration %d/%d\n', it, endIter);
				% Get the model parameters as a vector
				obj.theta = obj.neuronLayers{layer}.getTheta('variable');

				% Determine if new Epoch or new batch
				newEpoch = mod(it, obj.cnceOpt.epochSize) == 0 || it == startIter;
				newDataBatch = mod(it, nDataResample) == 0 || it == startIter;

				% Sample data and noise as required and return function handle
				objfun = obj.objFunSetup(layer, newDataBatch, newEpoch);

				% Run optimisation of objective funktion
				[obj.theta, aux] = estimateMain(objfun, obj.theta, obj.optimizerOpt);
				% update the model with the new theta
				obj.neuronLayers{layer}.updateParameters(obj.theta);

				% Update stats
				obj.stats.lossOpt = [obj.stats.lossOpt; aux.loss];
				obj.stats.loss = [obj.stats.loss; aux.loss(1), aux.loss(end)];

				obj.cIt = obj.cIt + 1; % Next iteration
				% Save intermediate results every iteration
				obj.saveResult([obj.savename, 'checkpoint.mat'], 1:length(obj.dataProcessors));

				% Save some more resutls every 20th iteration
				if it == 1 || mod(it, 20) == 0 || (it < 20 && layer == 1)
					ss = [obj.savename, num2str(it), '_T', datestr(now, 'HH_MM_SS')];
					obj.saveResult(ss, layer);
				end
			end
		end

		function [Z, gradU, vars] = evalModel(obj, X, L, inxL)
			% evalModel Forward prop through neural network until layer L 
            %  and backprops the state-space (image) gradient.
			% 	Input:
			% 		X - [N x D] image data in PCA space
			%		L - progagate until layer L-1, default is length of network
			% 		inxL - use output of unit "inxL" in layer L as evaluation
			%			Can be vector of values
			%			':' for all units
			%			[] for model pdf value <- default
			% 	Output:
			%		Z - [N x length(inxL)]
			% 		gradU - [N x D]
			%		vars - [1 x L] cell of intermediate results for each layer
			if nargin < 3, L = length(obj.neuronLayers); end

			Z = X;
			vars = cell(L, 1);
			% Forward progagation
			for l = 1:L-1
				Z = obj.neuronLayers{l}.forwardPass(Z);
				vars{l} = obj.neuronLayers{l}.hidden;
				vars{l}.V3 = Z;
% 				rDim = obj.neuronLayers{l+1}.layerDimensions(1);
				Z = obj.dataProcessors{l+1}.toPCA(Z);
			end

			% Determine what to evaluate after final layer
			if nargin < 4 || isempty(inxL)
				[Z, V3] = obj.neuronLayers{L}.pdf(Z);
				vars{L} = obj.neuronLayers{L}.hidden;
				if nargout > 1
					gradU = obj.neuronLayers{L}.gradV3(V3); % gradient for pdf
				end
			else
				Z = obj.neuronLayers{L}.forwardPass(Z);
				vars{L} = obj.neuronLayers{L}.hidden;
				Z = Z(:, inxL);
				if nargout > 1
					gradU = obj.neuronLayers{L}.gradV3(1, inxL); % gradient for unit inxL
				end
			end

			% Back progagation
			if nargout > 1
				for l = L-1:-1:1
					obj.neuronLayers{l}.hidden = vars{l};
					rDim = obj.neuronLayers{l+1}.layerDimensions(1);
					D = obj.neuronLayers{l}.layerDimensions(end);
					gradU = gradU * (obj.dataProcessors{l + 1}.getWh(rDim))'; % reverse PCA
					gradU = gradU * (eye(D) - (1/D)*ones(D));
					gradU = obj.neuronLayers{l}.gradV3(gradU, ':');
				end
			end
			obj.clearHiddenNL(); % Clear the intermediate variables to avoid unexpected behaviour
		end

		function [Z, vars] = evalModelVars(obj, X, L, inxL)
			% evalModelVars Forward prop through neural network until layer L
			% 	Input:
			% 		X - [N x D] image data in PCA space
			%		L - progagate until layer L-1, default is length of network
			% 		inxL - use output of unit "inxL" in layer L as evaluation
			%			Can be vector of values
			%			':' for all units
			%			[] for model pdf value <- default
			% 	Output:
			%		Z - [N x length(inxL)]
			%		vars - [1 x L] cell of intermediate results for each layer
			if nargin < 3, L = length(obj.neuronLayers); end

			Z = X;
			vars = cell(L, 1);
			% Forward progagation
			for l = 1:L-1
				Z = obj.neuronLayers{l}.forwardPass(Z);
				vars{l} = obj.neuronLayers{l}.hidden;
				vars{l}.V3 = Z;
% 				rDim = obj.neuronLayers{l+1}.layerDimensions(1);
				Z = obj.dataProcessors{l + 1}.toPCA(Z);
			end

			% Determine what to evaluate after final layer
			if nargin < 4 || isempty(inxL)
				[Z, ~] = obj.neuronLayers{L}.pdf(Z);
				vars{L} = obj.neuronLayers{L}.hidden;
			else
				Z = obj.neuronLayers{L}.forwardPass(Z);
				vars{L} = obj.neuronLayers{L}.hidden;
				Z = Z(:, inxL);
			end
			obj.clearHiddenNL(); % Clear the intermediate variables to avoid unexpected behaviour
        end

        %------------------------------------------------------------
		% ============= Loss function helper functions ==============
		%____________________________________________________________

		function objfun = getObjFun(obj, layer, dh, nh)
			% getObjFun Get the objective function for a layer with theta as input given data and noise
			%	Input:
			%		layer - layer of model to evaluate
			%		dh - DataHandle containing data samples
			%		nh - NoiseHandle containing noise samples
			%	Output:
			%		objfun - function handle with theta as input
			if nargin < 3 || isempty(dh), dh = obj.dataHandle; end
			if nargin < 4 || isempty(nh), nh = obj.noiseHandle; end
			if isa(obj.neuronLayers{layer}, 'DoubleNeuronLayer')
				objfun = @(theta) lossFunDNL(theta, obj.neuronLayers{layer}, dh, nh, obj.cnceOpt);
			elseif isa(obj.neuronLayers{layer}, 'SingleNeuronLayer')
				objfun = @(theta) lossFunSNL(theta, obj.neuronLayers{layer}, dh, nh, obj.cnceOpt);
			else
				error('Unrecognized NeuronLayer.')
			end
		end

		function objfun = getObjFunNoise(obj, layer)
			% getObjFunNoise Get the objective function with datahandle and noisehandle as input
			%	Input:
			%		layer - layer of model to evaluate
			%	Output:
			%		objfun - function handle with datahandle and noisehandle as input
			if isa(obj.neuronLayers{layer}, 'DoubleNeuronLayer')
				objfun = @(dh, nh) lossFunDNL(obj.theta, obj.neuronLayers{layer}, dh, nh, obj.cnceOpt);
			elseif isa(obj.neuronLayers{layer}, 'SingleNeuronLayer')
				objfun = @(dh, nh) lossFunSNL(obj.theta, obj.neuronLayers{layer}, dh, nh, obj.cnceOpt);
			else
				error('Unrecognized NeuronLayer.')
			end
		end

		function objfun = objFunSetup(obj, layer, getNewDataBatch, newEpoch)
			% objFunSetup Samples data and noise and get objective function
			%	Input:
			%		layer - layer of model to evaluate
			%		getNewDataBatch - bool, get new minibatch from dataProcessors if true
			%		newEpoch - bool, sample new larger batch in dataProcessors
			%	Output:
			%		objfun - function handle with theta as input

			% Sample new larger batch
			if newEpoch
				fprintf('Starting epoch %d. Generating new patch data...\n', obj.epoch)
				if strcmp(obj.cnceOpt.noiseType, 'Inter')
					obj.generateDataBatches(layer, obj.epoch);
				elseif strcmp(obj.cnceOpt.noiseType, 'Extra')
					obj.generateDataBatches(1, obj.epoch);
				else
					error('Unrecognized noiseType: %s', obj.cnceOpt.noiseType)
				end
				obj.epoch = obj.epoch + 1;
				fprintf('done!\n')
			end

			% Get new mini-batch
			if getNewDataBatch
				fprintf('Geting PCA databatch ...')
				if strcmp(obj.cnceOpt.noiseType, 'Inter')
					obj.getDataBatch(layer, obj.cnceOpt.batchSize);
				elseif strcmp(obj.cnceOpt.noiseType, 'Extra')
					obj.getDataBatch(1, obj.cnceOpt.batchSize);
				else
					error('Unrecognized noiseType: %s', obj.cnceOpt.noiseType)
				end
				fprintf('done!\n')
			end

			% Get new noise batch
			[dh, nh] = obj.getNoiseBatch(layer);   % for Inter noise type dh is empty and 
			objfun = obj.getObjFun(layer, dh, nh); % obj.dataHandle is used instead
		end

		%------------------------------------------------------------
		% ================== Saving Model ===========================
		%____________________________________________________________
		function saveResult(cest, filename, layers)
			% saveResult Saves the current CNCEstimator while avoiding to save large matricies
			%	Input:
			%		layers - vector of which layers for which to retain 
			%			data in dataProcessors after savning
			if nargin < 3, layers = []; end
			fprintf('saving... ')
			% Make sure to not save any data matricies
			% but save them temporarily
			[dataHandletmp, Zcell] = cest.clearHidden(layers);
			save(filename, 'cest')
			% Restore the data matricies after saving
			cest.dataHandle = dataHandletmp;
			for l = 1:length(cest.dataProcessors)
				if ismember(l, layers)
					cest.dataProcessors{l}.Z = Zcell{l};
				end
			end
			fprintf('done!\n')
		end

		function [dataHandle, Zcell] = clearHidden(obj, layers)
			% clearHidden Clear large matrices.
			%	Input:
			%		layers - vector of which layers for which to retain 
			%			data in dataProcessors after savning	
			if nargin < 2, layers = []; end
			dataHandle = obj.dataHandle;
			obj.dataHandle = [];
			obj.noiseHandle = [];
			Zcell = cell(length(obj.dataProcessors), 1);
			% (temporarily) clear data matrices 
			for l = 1:length(obj.dataProcessors)
				if ismember(l, layers)
					Zcell{l} = obj.dataProcessors{l}.clearZ();
				else
					obj.dataProcessors{l}.clearZ();
				end
			end
			obj.clearHiddenNL(); % clear hidden variables in neuronLayers
			obj.noiseProcessor.clearNoise(); % clear noise variables
			obj.theta = [];
		end

		function clearHiddenNL(obj)
			% clearHiddenNl clears hidden variables in neuronLayers
			if ~isempty(obj.neuronLayers)
				for l = 1:length(obj.neuronLayers)
					obj.neuronLayers{l}.clearHidden()
				end
			end
		end

		%------------------------------------------------------------
		% ================== Data and Noise =========================
		%____________________________________________________________

        function [Z, X, vars] = sampleFromX(obj, batchSize, layer, inxL, seed)
			% sampleFromX Sample image patches and evaluate model for these
				%	Input:
				%		batchSize - number of samples
				%		layer - evaluate until "layer"
				%		inxL - evaluate unit "inxL" in layer "layer"
				% 		seed - for reproducibility, need to set rng first
				% 	Output:
				%		Z - [N x length(inxL)] model output
				%		X - [N x D] data in PCA space
				%		vars - [1 x L] cell of intermediate results for each layer
			if nargin < 5, seed = 42; end
			X = getRawData(obj.dataOpt.dataset, obj.dataOpt.winsize, batchSize, seed);
			X = obj.dataProcessors{1}.toPCA(X);
			[Z, vars] = obj.evalModelVars(X, layer, inxL);
		end

		function [ZX, ZY, X, Y, varsX, varsY] = sampleFromXandY(obj, batchSize, layer, inxL, seed)
			% sampleFromXandY Sample image patches and noise and evaluate model for these
			%	Input:
			%		batchSize - number of samples
			%		layer - evaluate until "layer"
			%		inxL - evaluate unit "inxL" in layer "layer"
			% 		seed - for reproducibility, need to set rng first
			% 	Output:
			%		ZX - [N x length(inxL)] model output for data
			%		ZY - [N*kappa x length(inxL)] model output for noise
			%		X - [N x D] data in PCA space
			%		Y - [N*kappa x D] noise in PCA space
			%		varsX - [1 x L] cell of intermediate results for each layer for data
			%		varsY - [1 x L] cell of intermediate results for each layer for noise
			if nargin < 5, seed = 42; end

			% Sample data
			X = getRawData(obj.dataOpt.dataset, obj.dataOpt.winsize, batchSize, seed);
			X = obj.dataProcessors{1}.toPCA(X);
			[ZX, varsX] = obj.evalModelVars(X, layer, inxL);
			dh = DataHandle();
			dh.X = X;

			% Sample noise using data
			if isempty(obj.noiseProcessor)
				obj.noiseProcessor = NoiseProcessor(obj.cnceOpt);
			end
			[N, D] = size(X);
			obj.noiseProcessor.generateNoiseBase(N, D);
			noiseHandletmp = obj.noiseProcessor.computeNoise(dh, obj.neuronLayers{1});
			Y = noiseHandletmp.Y;
			[ZY, varsY] = obj.evalModelVars(Y, layer, inxL);
			obj.noiseProcessor.clearNoise(); % Clear intermediate noise variables
        end

		function dh = getDataBatch(obj, layer, batchSize)
			% getDataBatch Gets a data batch from dataProcessors{layer}
			obj.dataHandle = obj.dataProcessors{layer}.getPCAdataBatch(batchSize);
			if nargout > 0,	dh = obj.dataHandle; end
		end

		function generateDataBatches(obj, layer, epoch)
			% generateDataBatches Generates PCA data for dataProcessors{layer}
			if isempty(obj.dataProcessors{layer}.dataFun)
				obj.addDataFun(layer); % If datafun not yet set or removed when savning
			end
			obj.dataProcessors{layer}.generatePCAdata(epoch);
		end

		function [dh, nh] = getNoiseBatch(obj, layer)
			% getNewDataBatch Generates a noise batch, and for noiseType 'Extra' also a data batch
			if isempty(obj.noiseProcessor)
				obj.noiseProcessor = NoiseProcessor(obj.cnceOpt);
			end
			% Either 'Inter' noise or 'Extra' noise
			if (~isfield(obj.cnceOpt, 'noiseType') || isempty(obj.cnceOpt.noiseType) || ...
					strcmp(obj.cnceOpt.noiseType, 'Inter') )
				objFunNoise = obj.getObjFunNoise(layer);
				[nh, tmpstats] = obj.noiseProcessor.generateNoiseInter(obj.dataHandle, ...
													obj.neuronLayers{layer}, objFunNoise);
				dh = [];
			elseif strcmp(obj.cnceOpt.noiseType, 'Extra')
				[dh, nh, tmpstats] = obj.noiseProcessor.generateNoiseExtra(obj, layer);
			else
				error('Unrecognized noiseType: %s', obj.cnceOpt.noiseType)
			end
			obj.noiseHandle = nh;
			obj.setNoiseStats(tmpstats); % update statistics
		end

		function setNoiseStats(obj, tmpstats)
			% setNoiseStats updates the statistics after generating noise
			% loss using epsilon base
			obj.stats.lossEpsBase = [obj.stats.lossEpsBase; tmpstats.lossEpsBase];
			% loss using selected epsilon
			obj.stats.lossEps = [obj.stats.lossEps; tmpstats.lossEps];
			% the epsilon factor (which epsBase is multiplied with)
			obj.stats.epsf = [obj.stats.epsf; tmpstats.epsf];
			% Number of noise generation iterations
			obj.stats.noiseIters = [obj.stats.noiseIters; tmpstats.noiseIters];
			% the current values of epsilon
			obj.stats.epsilon = [obj.stats.epsilon; obj.noiseProcessor.epsilon];
		end

		function addDataFun(obj, l)
			% addDataFun add the data sampling function to the DataProcessor at layer "l"
			if l == 1
				% Raw PCA image data generated for first layer
				obj.dataProcessors{l}.dataFun = @(bSize, bNr) getRawData(obj.dataOpt.dataset, ...
													obj.dataOpt.winsize, bSize, bNr);
			else
				% Data is passed through model for following layers (though not when using 
				% noise type 'Extra')
				obj.dataProcessors{l}.dataFun = @(bSize, bNr) obj.sampleFromX(bSize, l-1, ':', bNr);
			end
		end

		%------------------------------------------------------------
		% ================== Layer Management =======================
		%____________________________________________________________

		function addNeuronLayer(obj, nl)
			% addNeuronLayer add a neuron layer nl to the end of the network
			L = length(obj.neuronLayers) + 1;
			% set all units in previous layers to inactive
			for l = 1:L-1
				if isa(obj.neuronLayers{l}, 'DoubleNeuronLayer')
					obj.neuronLayers{l}.indices = {[], []};
				elseif isa(obj.neuronLayers{l}, 'SingleNeuronLayer')
					obj.neuronLayers{l}.indices = [];
				else
					warning(['Unrecognized NeuronlLayer at l = ', num2str(l)])
				end
			end
			obj.setLayer(nl, L); % add the layer and sync modelOpt
		end

		function removeLayer(obj)
			% removeLayer removes the last layer of the model
			obj.neuronLayers = obj.neuronLayers(1:end-1);
			obj.syncModelOpt(); % sync values in modelOpt
		end

		function setLayer(obj, nl, layer)
			% setLayer sets the neuronLayer at layer
			obj.neuronLayers{layer} = nl;
			obj.syncModelOpt(); % sync values in modelOpt
		end

		function addInterLayer(obj, l, rDim, dirName)
			% addInterLayer adds a DataProcessor at between layer l-1 and l
			if nargin < 4 || isempty(dirName), dirName = []; end
			D = obj.neuronLayers{l-1}.layerDimensions(end);
			% create the DataProcessor
			obj.dataProcessors{l} = DataProcessor({obj.dataOpt.dataset, ...
				D, obj.cnceOpt.N, l, rDim, obj.dataOpt.rngseed, dirName});
			obj.addDataFun(l); % Add the data sampling function
            if isempty(rDim) || ~isfile(obj.dataProcessors{l}.pcaFileName)
                batchSize = min(50000, obj.dataProcessors{l}.N);
				nBatches = ceil(obj.dataProcessors{l}.N / batchSize);
                obj.dataProcessors{l}.generatePCAtransform(batchSize, 2*nBatches);
            end
                
		end

		function generateInterLayerTransform(obj, layer)
			% generateInterLayerTransform generates the whitening and dewhitening matrices for
			% inter layer "layer"
			batchSize = min(50000, obj.dataOpt.N);
			nBatches = 2 * ceil(obj.dataOpt.N / batchSize );
			obj.dataProcessors{layer}.generatePCAtransform(batchSize, nBatches);
		end

		function turnSingleToDouble(obj, layer, newLayerDim, newIndices, newActivator)
			% turnSingleToDouble turns a SingleNeuronLayer at "layer" into a DoubleNeuronLayer
			% 	newLayerDim, newIndices, newActivator are new setup for the DoubleNeuronLayer
			
			% Make sure the new layer dimensions are compatible with current network
			if (layer < length(obj.neuronLayers) && ...
				newLayerDim < obj.neuronLayers{layer + 1}.layerDimensions(1))
				warning('Layer dimensions mismatch')
			end
			nl = obj.neuronLayers{layer}; % The current SingleNeuronLayer
			
			% Default upgrade if not newActivator is provided
			if nargin < 5
				if strcmp(nl.activator, 'LinearLog')
					newActivator = 'LinearQuadraticLog';
				elseif strcmp(nl.activator, 'Max')
					newActivator = 'MaxQuadraticLog';
				else
					error('The SingleNeuronLayer must have either LinearLog or Max activator')
				end
			end
			fprintf('Upgrading to DoubleLayer with activator: %s...', newActivator)
			% create the new DoubleLayer
			dnl = DoubleNeuronLayer({[nl.layerDimensions, newLayerDim], newIndices, newActivator});
			dnl.W = nl.W; % Set the W matrices to the current one
			obj.neuronLayers{layer} = dnl;
			obj.syncModelOpt();
			% Change the input dimensions of the following dataProcessor
			if layer < length(obj.neuronLayers) && ~isempty(obj.dataProcessors)
				obj.dataProcessors{layer + 1}.dim = newLayerDim;
			end
			fprintf('done!\n')
		end
		
		function extendNeuronLayerW(obj, layer, size)
			%  extendNeuronLayerW widens W matrix in the image model at layer "layer" by "size" units
			fprintf('Extending W in layer %d by %d units... ', layer, size)
			dims = obj.neuronLayers{layer}.layerDimensions;
			if isa(obj.neuronLayers{layer}, 'DoubleNeuronLayer')
				dims(2) = size; % the added width
				newSnl = SingleNeuronLayer({dims, ':', obj.neuronLayers{layer}.activator});
				obj.neuronLayers{layer}.indices{1} = []; % deactivate old units
			elseif isa(obj.neuronLayers{layer}, 'SingleNeuronLayer')
				dims(2) = size; % the added width
				newSnl = SingleNeuronLayer({dims, ':', obj.neuronLayers{layer}.activator});
				obj.neuronLayers{layer}.indices = []; % deactivate old units
			end
			obj.neuronLayers{layer}.extendW(newSnl);
			fprintf('done!\n ')
			obj.syncModelOpt();
			obj.show();
		end
		
		function extendNeuronLayerR(obj, layer, size)
			%  extendNeuronLayerR widens R matrix in the image model at layer "layer" by "size" units
			fprintf('Extending R in layer %d by %d units... ', layer, size)
			if isa(obj.neuronLayers{layer}, 'SingleNeuronLayer')
				error('Cannot extend R for a SingleNeuronLayer.')
			elseif isa(obj.neuronLayers{layer}, 'DoubleNeuronLayer')
				dims = obj.neuronLayers{layer}.layerDimensions;
				dims(3) = size; % the added width
				newDnl = DoubleNeuronLayer({dims, ':', obj.neuronLayers{layer}.activator});
				obj.neuronLayers{layer}.indices{2} = []; % deactivate old units
				obj.neuronLayers{layer}.extendR(newDnl);
			end
			fprintf('done!\n ')
			obj.syncModelOpt();
			obj.show();
		end

		function syncModelOpt(obj)
			% syncModelOpt make sure values in modelOpt agree with current status of network.
			obj.modelOpt.L = length(obj.neuronLayers);
			obj.modelOpt.indices = cell(1, obj.modelOpt.L);
			obj.modelOpt.dimensions = cell(1, obj.modelOpt.L);
			obj.modelOpt.activators = cell(1, obj.modelOpt.L);
			obj.modelOpt.types = cell(1, obj.modelOpt.L);
			for l = 1:obj.modelOpt.L
				obj.modelOpt.indices{l} = obj.neuronLayers{l}.indices;
				obj.modelOpt.dimensions{l} = obj.neuronLayers{l}.layerDimensions;
				obj.modelOpt.activators{l} = obj.neuronLayers{l}.activator;
				if isa(obj.neuronLayers{l}, 'DoubleNeuronLayer')
					obj.modelOpt.types{l} = 'DoubleNeuronLayer';
				elseif isa(obj.neuronLayers{l}, 'SingleNeuronLayer')
					obj.modelOpt.types{l} = 'SingleNeuronLayer';
				else
					fprintf('At layer %d\n', l);
					error('Unrecognized NeuronLayer')
				end
			end
		end

		function str = show(obj)
			% show returns and displays a string representation of the neural image model
			str = '';
			obj.syncModelOpt(); % make sure modelOpt agrees with current status
			L = length(obj.neuronLayers);
			% create the string layer by layer
			for l = 1:L
				dims = obj.neuronLayers{l}.layerDimensions;
				activ = obj.neuronLayers{l}.activator;
				nNeurons = obj.neuronLayers{l}.getnNeurons('variable');

				obj.dataProcessors{l}.syncDwinsize(); % For compatability with old version
				D = obj.dataProcessors{l}.dim;
				rDim = obj.dataProcessors{l}.rDim;
				if isempty(obj.dataProcessors{l}.isDisabled) || ~obj.dataProcessors{l}.isDisabled
					s = sprintf('(%d -> %d) ', D, rDim);
				else
					s = sprintf('(%d -I-> %d) ', D, rDim);
				end

				if isa(obj.neuronLayers{l}, 'DoubleNeuronLayer')
					s2 = sprintf('[%d-%s-%d(%d)-%d(%d)] ', ...
						dims(1), activ, dims(2), nNeurons(1), dims(3), nNeurons(2));
				elseif isa(obj.neuronLayers{l}, 'SingleNeuronLayer')
					s2 = sprintf('[%d-%s-%d(%d)] ', dims(1), activ, dims(2), nNeurons(1));
				else
					fprintf('At layer %d\n', l);
					error('Unrecognized NeuronLayer')
				end
				str = [str, s, s2];
			end
			disp(str)
		end

		%------------------------------------------------------------
		% ================== Initialization =========================
		%____________________________________________________________
		function obj = CNCEstimator(init)
			% CNCESTIMATOR init = {dataOpt, cnceOpt, modelOpt, optimizerOpt}
			% 	set dataOpt with setDataOpt(dataOpt)
			% 	set cnceOpt with setCnceOpt(cnceOpt)
			% 	set modelOpt with initializeNeuronLayers(modelOpt)
			% 	set optimizerOpt with setOptimizerOpt(optimizerOpt)
			if nargin == 0
			;
			elseif isa(init, 'cell')
				fprintf([repelem('_', 40), '\n'])
				obj.setDataOpt(init{1});
				obj.setCnceOpt(init{2});
				obj.initializeNeuronLayers(init{3});
				obj.setOptimizerOpt(init{4});
				obj.initializeStats();
			end
		end

		function setDataOpt(obj, dataOpt)
			if nargin > 1, obj.dataOpt = dataOpt; end
			if ~isfield(obj.dataOpt, 'dataset')
				error('No dataset field for dataOpt')
			end
			fprintf('dataOpt.dataset:%24s\n', obj.dataOpt.dataset)
			if ~isfield(obj.dataOpt, 'winsize')
				obj.dataOpt.winsize = 32;
				fprintf('dataOpt.winsize (default):%14d\n', obj.dataOpt.winsize)
			else
				fprintf('dataOpt.winsize:%24d\n', obj.dataOpt.winsize)
			end
			if ~isfield(obj.dataOpt, 'N')
				obj.dataOpt.N = 50000;
				fprintf('dataOpt.N (default):%20d\n', obj.dataOpt.N)
			else
				fprintf('dataOpt.N:%30d\n', obj.dataOpt.N)
			end
			if ~isfield(obj.dataOpt, 'modelLayer')
				obj.dataOpt.modelLayer = 1;
				fprintf('dataOpt.modelLayer (default):%11d\n', obj.dataOpt.modelLayer)
			else
				fprintf('dataOpt.modelLayer:%21d\n', obj.dataOpt.modelLayer)
			end
			if ~isfield(obj.dataOpt, 'rDim')
				obj.dataOpt.rDim = [];
				error('dataOpt.rDim (default):%17s\n', '[]');
			else
				fprintf('dataOpt.rDim:%27d\n', obj.dataOpt.rDim)
			end
			if ~isfield(obj.dataOpt, 'rngseed')
				obj.dataOpt.rngseed = randi(10000);
				fprintf('dataOpt.rngseed (default):%14d\n', obj.dataOpt.rngseed)
			else
				fprintf('dataOpt.rngseed:%24d\n', obj.dataOpt.rngseed)
			end
			if ~isfield(obj.dataOpt, 'pcaDirname')
				obj.dataOpt.pcaDirname = [];
				fprintf('dataOpt.pcaDirname (default):%11s\n', '[]')
			else
				fprintf('dataOpt.pcaDirname:%21s\n', obj.dataOpt.pcaDirname)
			end
			imDaInit = {obj.dataOpt.dataset, obj.dataOpt.winsize^2, obj.dataOpt.N, ...
				obj.dataOpt.modelLayer, obj.dataOpt.rDim, obj.dataOpt.rngseed, obj.dataOpt.pcaDirname};
			obj.dataProcessors{1} = DataProcessor(imDaInit);
			fprintf([repelem('-', 40), '\n'])
		end

		function setCnceOpt(obj, cnceOpt)
			if nargin > 1, obj.cnceOpt = cnceOpt; end

			if ~isfield(obj.cnceOpt, 'aCNCE')
				obj.cnceOpt.aCNCE = 0;
				fprintf('cnceOpt.aCNCE (default):%16d\n', obj.cnceOpt.aCNCE)
			else
				fprintf('cnceOpt.aCNCE:%26d\n', obj.cnceOpt.aCNCE)
			end
			if ~isfield(obj.cnceOpt, 'nmIter')
				obj.cnceOpt.nmIter = 20;
				fprintf('cnceOpt.nmIter (default):%15d\n', obj.cnceOpt.nmIter)
			else
				fprintf('cnceOpt.nmIter:%25d\n', obj.cnceOpt.nmIter)
			end
			if ~isfield(obj.cnceOpt, 'kappa')
				obj.cnceOpt.kappa = 6;
				fprintf('cnceOpt.kappa (default):%16d\n', obj.cnceOpt.kappa)
			else
				fprintf('cnceOpt.kappa:%26d\n', obj.cnceOpt.kappa)
			end
			if ~isfield(obj.cnceOpt, 'batchSize')
				obj.cnceOpt.batchSize = 50000;
				fprintf('cnceOpt.batchSize (default):%12d\n', obj.cnceOpt.batchSize)
			else
				fprintf('cnceOpt.batchSize:%22d\n', obj.cnceOpt.batchSize)
			end
			if ~isfield(obj.cnceOpt, 'cIt')
				obj.cnceOpt.cIt = 1;
				fprintf('cnceOpt.cIt (default):%18d\n', obj.cnceOpt.cIt)
			else
				fprintf('cnceOpt.cIt:%28d\n', obj.cnceOpt.cIt)
			end
			obj.cIt = obj.cnceOpt.cIt;
			if ~isfield(obj.cnceOpt, 'id')
				obj.cnceOpt.id = 'default';
				fprintf('cnceOpt.id (default):%19s\n', obj.cnceOpt.id)
			else
				fprintf('cnceOpt.id:%29s\n', obj.cnceOpt.id)
			end
			if ~isfield(obj.cnceOpt, 'epsBase')
				obj.cnceOpt.epsBase = 1;
				fprintf('cnceOpt.epsBase (default):%14d\n', obj.cnceOpt.epsBase)
			else
				fprintf('cnceOpt.epsBase:%24d\n', obj.cnceOpt.epsBase)
			end
			if ~isfield(obj.cnceOpt, 'N')
				obj.cnceOpt.N = 20e4;
				fprintf('cnceOpt.N (default):%20e\n', obj.cnceOpt.N)
			else
				fprintf('cnceOpt.N:%30e\n', obj.cnceOpt.N)
			end
			if ~isfield(obj.cnceOpt, 'epochSize')
				obj.cnceOpt.epochSize = 10 * floor(obj.cnceOpt.N/obj.cnceOpt.batchSize);
				fprintf('cnceOpt.epochSize (default):%12d\n', obj.cnceOpt.epochSize)
			else
				fprintf('cnceOpt.epochSize:%22d\n', obj.cnceOpt.epochSize)
			end
			if ~isfield(obj.cnceOpt, 'noiseType')
				obj.cnceOpt.noiseType = 'Inter';
				fprintf('cnceOpt.noiseType (default):%12s\n', obj.cnceOpt.noiseType)
			else
				fprintf('cnceOpt.noiseType:%22s\n', obj.cnceOpt.noiseType)
			end
			% Fix to make sure total batch size is set via cnceOpt
			for l = 1:length(obj.dataProcessors)
				obj.dataProcessors{l}.N = obj.cnceOpt.N;
			end
			obj.noiseProcessor = NoiseProcessor(obj.cnceOpt);
			obj.dataHandle = [];
			obj.noiseHandle = [];
			obj.epoch = 1;
			fprintf([repelem('-', 40), '\n'])
			obj.setSavename()
			fprintf([repelem('-', 40), '\n'])
		end

		function setSavename(obj)
            global CNCE_IMAGE_RESULTS_FOLDER;
            
			savefolder = sprintf('%s/%s/rDim_%d/%s', ...
                CNCE_IMAGE_RESULTS_FOLDER, ...
				obj.dataOpt.dataset, obj.dataOpt.rDim, datestr(now, 29));
			if ~isfolder(savefolder), mkdir(savefolder); end
			obj.savename = sprintf('%s/%s_', savefolder, obj.cnceOpt.id);

			fprintf('savename:\t%s\n', obj.savename)
		end

		function initializeNeuronLayers(obj, modelOpt)
			if nargin > 1, obj.modelOpt = modelOpt; end
			for l = 1:obj.modelOpt.L
				fprintf('. ')
				init = {obj.modelOpt.dimensions{l}, obj.modelOpt.indices{l},...
						obj.modelOpt.activators{l}};
				if strcmp(obj.modelOpt.types{l}, 'DoubleNeuronLayer')
					obj.neuronLayers{l} = DoubleNeuronLayer(init);
				elseif strcmp(obj.modelOpt.types{l}, 'SingleNeuronLayer')
					obj.neuronLayers{l} = SingleNeuronLayer(init);
				else
					error('Unrecognized NeuronLayer type.');
				end
			end
			fprintf('\n')
		end

		function setOptimizerOpt(obj, optimizerOpt)
			if nargin > 1, obj.optimizerOpt = optimizerOpt; end
			if ~isfield(obj.optimizerOpt, 'alg')
				obj.optimizerOpt.alg = 'minimize';
				fprintf('optimizerOpt.alg (default):%13s\n', obj.optimizerOpt.alg)
			else
				fprintf('optimizerOpt.alg:%23s\n', obj.optimizerOpt.alg)
			end
			if ~isfield(obj.optimizerOpt, 'maxIter')
				obj.optimizerOpt.maxIter = 10;
				fprintf('optimizerOpt.maxIter (default):%9d\n', obj.optimizerOpt.maxIter)
			else
				fprintf('optimizerOpt.maxIter:%19d\n', obj.optimizerOpt.maxIter)
			end
			if ~isfield(obj.optimizerOpt, 'verbose')
				obj.optimizerOpt.verbose = true;
				fprintf('optimizerOpt.verbose (default):%9d\n', obj.optimizerOpt.verbose)
			else
				fprintf('optimizerOpt.verbose:%19d\n', obj.optimizerOpt.verbose)
			end
			fprintf([repelem('-', 40), '\n'])
		end

		function initializeStats(obj)
			obj.stats.loss = [];
			obj.stats.lossOpt = [];
			obj.stats.epsilon = [];
			obj.stats.lossEps = [];
			obj.stats.noiseIters = [];
			obj.stats.epsf = [];
			obj.stats.lossEpsBase = [];
		end
    end

end

%------------------------------------------------------------
% ================== CNCE loss functions ====================
%____________________________________________________________
function [loss, grad_t] = lossFunDNL(theta, dnl, dh, nh, cnceOpt)
	% lossFunDNL Loss function for DoubleNeuronLayers
	%	Input:
	%		dnl - DoubleNeuronLayer
	%		dh - DataHandle containing the data
	% 		nh - NoiseHandle containing the noise
	%		cnceOpt - struct containing options (kappa and approxFactor)
	%	Output:
	%		loss - the loss value
	%		grad_t - the gradient of the loss function

	if isfield(cnceOpt, 'approxFactor') && ~isempty(cnceOpt.approxFactor)
		approxFactor = cnceOpt.approxFactor;
	else
		approxFactor = 30;
	end
	N = size(dh.X, 1);
	kappa = cnceOpt.kappa;

    % Set model weights to given parameter value
	if ~isempty(theta)
		dnl.updateParameters(theta);
	end
	if ~dh.isOnSphere, dh.moveToSphere(); end
	if ~nh.isOnSphere, nh.moveToSphere(); end

	% Forward pass for different activation functions
	Q = dnl.R.^2;
	if strcmp(dnl.activator, 'LinearQuadraticLog')
		V1x = dh.X * dnl.W'; 							% [N x nNeurons1]
		V1y = nh.Y * dnl.W'; 							% [kappa * N x nNeurons1]
		V2x = (V1x.^2) * Q'; 							% [N x nNeurons2]
		V2y = (V1y.^2) * Q'; 							% [kappa * N x nNeurons2]
		V3x = bsxfun(@plus, log(V2x + 1), dnl.b'); 		% [N x nNeurons2]
		V3y = bsxfun(@plus, log(V2y + 1), dnl.b'); 		% [kappa * N x nNeurons2]
	elseif strcmp(dnl.activator, 'MaxQuadraticLog')
		V1x = max(dh.X * dnl.W', 0); 					% [N x nNeurons1]
		V1y = max(nh.Y * dnl.W', 0); 					% [N x nNeurons1]
		V2x = (V1x.^2) * Q'; 							% [N x nNeurons2]
		V2y = (V1y.^2) * Q'; 							% [N x nNeurons2]
		V3x = bsxfun(@plus, log(V2x + 1), dnl.b'); 		% [N x nNeurons2]
		V3y = bsxfun(@plus, log(V2y + 1), dnl.b'); 		% [kappa * N x nNeurons2]
	elseif strcmp(dnl.activator, 'LinearLinearLog')
		V1x = dh.X * dnl.W'; 							% [N x nNeurons1]
		V1y = nh.Y * dnl.W'; 							% [kappa * N x nNeurons1]
		V2x = V1x * Q'; 								% [N x nNeurons2]
		V2y = V1y * Q'; 								% [kappa * N x nNeurons2]
		V3x = bsxfun(@plus, log(V2x.^2 + 1), dnl.b'); 	% [N x nNeurons2]
		V3y = bsxfun(@plus, log(V2y.^2 + 1), dnl.b'); 	% [kappa * N x nNeurons2]
	elseif strcmp(dnl.activator, 'LinearLinearMaxLog')
		V1x = dh.X * dnl.W'; 							% [N x nNeurons1]
		V1y = nh.Y * dnl.W'; 							% [kappa * N x nNeurons1]
		V2x = V1x * Q'; 								% [N x nNeurons2]
		V2y = V1y * Q'; 								% [kappa * N x nNeurons2]
		V3x = bsxfun(@plus, log(abs(V2x) + 1), dnl.b'); % [N x nNeurons2]
		V3y = bsxfun(@plus, log(abs(V2y) + 1), dnl.b'); % [kappa * N x nNeurons2]
	elseif strcmp(dnl.activator, 'LinearLinearMax')
		V1x = dh.X * dnl.W'; 							% [N x nNeurons1]
		V1y = nh.Y * dnl.W'; 							% [kappa * N x nNeurons1]
		V2x = V1x * Q'; 								% [N x nNeurons2]
		V2y = V1y * Q'; 								% [kappa * N x nNeurons2]
		V3x = bsxfun(@plus, max(V2x, 0), dnl.b'); 		% [N x nNeurons2]
		V3y = bsxfun(@plus, max(V2y, 0), dnl.b'); 		% [kappa * N x nNeurons2]
	elseif strcmp(dnl.activator, 'LinearLinear')
		V1x = dh.X * dnl.W'; 							% [N x nNeurons1]
		V1y = nh.Y * dnl.W'; 							% [kappa * N x nNeurons1]
		V2x = V1x * Q'; 								% [N x nNeurons2]
		V2y = V1y * Q'; 								% [kappa * N x nNeurons2]
		V3x = bsxfun(@plus, V2x, dnl.b'); 				% [N x nNeurons2]
		V3y = bsxfun(@plus, V2y, dnl.b'); 				% [kappa * N x nNeurons2]
	else
		error(['Unrecognized DoubleNeuronLayer activator: ', dnl.activator])
	end

	logPhix = sum(rectif(V3x), 2); 						% [N x 1]
	logPhiy = sum(rectif(V3y), 2); 						% [kappa * N x 1]

	Gxy = repmat(logPhix, kappa, 1) - logPhiy;
	Gxy = Gxy - nh.logP; % Gxy for all sample pairs, see paper
	lnegG 	= Gxy < -1 * approxFactor;
	lposG 	= Gxy > approxFactor;
	midG 	= ~lnegG  & ~lposG;
	h = 1./ ( 1 +  exp(Gxy(midG)));
	loss = -1 * sum(log(1 - h));
	loss = loss - sum(Gxy(lnegG));
	loss = loss / (0.5 * kappa *N);

	% Backward pass
	if nargout > 1

        dJ_dG = zeros(N*kappa, 1);
        dJ_dG(midG) = h;
        dJ_dG(lposG) = exp(-1 * Gxy(lposG) );
        dJ_dG(lnegG) = 1;

        % Backward -- Noise term
        dJ_dfV	 	= bsxfun(@times, dJ_dG, rectifDot(V3y)); 		%[N*kappa x nNeurons2]
        if (strcmp(dnl.activator, 'LinearQuadraticLog') || ...
            strcmp(dnl.activator, 'MaxQuadraticLog'))
            dJ_dV2 	= dJ_dfV./(V2y + 1);							%[N*kappa x nNeurons2]
            dJ_dR 	= (dJ_dV2(:, dnl.indices{2})' * (V1y.^2)) ...
                        .* dnl.R(dnl.indices{2}, :);				%[nNeurons2Var x nNeurons1]
            dJ_dV1	= (dJ_dV2 * Q(:, dnl.indices{1})) ...
                        .* V1y(:, dnl.indices{1}); 					%[N*kappa x nNeurons1Var]
        elseif strcmp(dnl.activator, 'LinearLinearLog')
            dJ_dV2 	= dJ_dfV .* (V2y ./ (V2y.^2 + 1));				%[N*kappa x nNeurons2]
            dJ_dR 	= 2 * (dJ_dV2(:, dnl.indices{2})' * V1y) ...
                            .* dnl.R(dnl.indices{2}, :);			%[nNeurons2Var x nNeurons1]
            dJ_dV1	= dJ_dV2 * Q(:, dnl.indices{1}); 				%[N*kappa x nNeurons1Var]
        elseif strcmp(dnl.activator, 'LinearLinearMaxLog')
            dJ_dV2 	= dJ_dfV .* (sign(V2y) ./ (abs(V2y) + 1));		%[N*kappa x nNeurons2]
            dJ_dR 	= (dJ_dV2(:, dnl.indices{2})' * V1y) ...
                            .* dnl.R(dnl.indices{2}, :);			%[nNeurons2Var x nNeurons1]
            dJ_dV1	= 0.5 * dJ_dV2 * Q(:, dnl.indices{1}); 			%[N*kappa x nNeurons1Var]
        elseif strcmp(dnl.activator, 'LinearLinearMax')
            dJ_dV2 	= dJ_dfV .* (V2y > 0);							%[N*kappa x nNeurons2]
            dJ_dR 	= (dJ_dV2(:, dnl.indices{2})' * V1y) ...
                            .* dnl.R(dnl.indices{2}, :);			%[nNeurons2Var x nNeurons1]
            dJ_dV1	= 0.5 * dJ_dV2 * Q(:, dnl.indices{1}); 			%[N*kappa x nNeurons1Var]
        elseif strcmp(dnl.activator, 'LinearLinear')
            dJ_dV2 	= dJ_dfV;										%[N*kappa x nNeurons2]
            dJ_dR 	= (dJ_dV2(:, dnl.indices{2})' * V1y) ...
                            .* dnl.R(dnl.indices{2}, :);			%[nNeurons2Var x nNeurons1]
            dJ_dV1	= 0.5 * dJ_dV2 * Q(:, dnl.indices{1}); 			%[N*kappa x nNeurons1Var]
        else
            error('Unrecognized activator') % sanity check
        end

        grad_b_y 	= -2 * sum(dJ_dfV(:, dnl.indices{2}), 1) / (N * kappa);	%[1 x nNeurons2Var]
        grad_W_y 	= -2 * 2 * (dJ_dV1' * nh.Y ) / (N * kappa); 			%[nNeurons1Var x D]
        grad_R_y	= -2 * 2 * dJ_dR / (N * kappa);							%[nNeurons2Var x nNeurons1]

        % Backward -- data term	
        dJ_dG_x 	= sum(reshape(dJ_dG, N, kappa), 2);						% [N x 1]
        dJ_dfV		= bsxfun(@times, dJ_dG_x, rectifDot(V3x));				%[N x nNeurons2]
        if (strcmp(dnl.activator, 'LinearQuadraticLog') || ...
            strcmp(dnl.activator, 'MaxQuadraticLog'))
            dJ_dV2 		= dJ_dfV./( V2x + 1);								%[N x nNeurons2]
            dJ_dR 		= (dJ_dV2(:, dnl.indices{2})' * (V1x.^2)) ...
                            .* dnl.R(dnl.indices{2}, :);					% [nNeurons2Var x nNeurons1]
            dJ_dV1		= (dJ_dV2 * Q(:, dnl.indices{1})) ...
                            .* V1x(:, dnl.indices{1});  					%[N x nNeurons1Var]
        elseif strcmp(dnl.activator, 'LinearLinearLog')
            dJ_dV2 	= dJ_dfV .* (V2x ./ (V2x.^2 + 1));						%[N*kappa x nNeurons2]
            dJ_dR 	= 2 * (dJ_dV2(:, dnl.indices{2})' * V1x) ...
                            .* dnl.R(dnl.indices{2}, :);					%[nNeurons2Var x nNeurons1]
            dJ_dV1	= dJ_dV2 * Q(:, dnl.indices{1}); 						%[N*kappa x nNeurons1Var]
        elseif strcmp(dnl.activator, 'LinearLinearMaxLog')
            dJ_dV2 	= dJ_dfV .* (sign(V2x) ./ (abs(V2x) + 1));				%[N*kappa x nNeurons2]
            dJ_dR 	= (dJ_dV2(:, dnl.indices{2})' * V1x) ...
                            .* dnl.R(dnl.indices{2}, :);					%[nNeurons2Var x nNeurons1]
            dJ_dV1	= 0.5 * dJ_dV2 * Q(:, dnl.indices{1}); 					%[N*kappa x nNeurons1Var]
        elseif strcmp(dnl.activator, 'LinearLinearMax')
            dJ_dV2 	= dJ_dfV .* (V2x > 0);									%[N*kappa x nNeurons2]
            dJ_dR 	= (dJ_dV2(:, dnl.indices{2})' * V1x) ...
                            .* dnl.R(dnl.indices{2}, :);					%[nNeurons2Var x nNeurons1]
            dJ_dV1	= 0.5 * dJ_dV2 * Q(:, dnl.indices{1}); 					%[N*kappa x nNeurons1Var]
        elseif strcmp(dnl.activator, 'LinearLinear')
            dJ_dV2 	= dJ_dfV;												%[N*kappa x nNeurons2]
            dJ_dR 	= (dJ_dV2(:, dnl.indices{2})' * V1x) ...
                            .* dnl.R(dnl.indices{2}, :);					%[nNeurons2Var x nNeurons1]
            dJ_dV1	= 0.5 * dJ_dV2 * Q(:, dnl.indices{1}); 					%[N*kappa x nNeurons1Var]
        else
            error('Unrecognized activator') % sanity check
        end

        grad_b_x 	= -2 * sum(dJ_dfV(:, dnl.indices{2}), 1) / (N * kappa); %[1 x nNeurons2Var]
        grad_W_x 	= -2 * 2 * (dJ_dV1' * dh.X ) / (N * kappa);             %[nNeurons1Var x D]
        grad_R_x	= -2 * 2 * dJ_dR / (N * kappa);							%[nNeurons2Var x nNeurons1]

        grad_W = grad_W_x - grad_W_y;
        grad_b = grad_b_x - grad_b_y;
        grad_R = grad_R_x - grad_R_y;

        grad_t = [ grad_W(:); grad_R(:); grad_b'];
	end
end

function [loss, grad_t] = lossFunSNL(theta, snl, dh, nh, cnceOpt)
	% lossFunSNL Loss function for SingleNeuronLayers
	%	Input:
	%		snl - SingleNeuronLayer
	%		dh - DataHandle containing the data
	% 		nh - NoiseHandle containing the noise
	%		cnceOpt - struct containing options (kappa and approxFactor)
	%	Output:
	%		loss - the loss value
	%		grad_t - the gradient of the loss function
	if isfield(cnceOpt, 'approxFactor') && ~isempty(cnceOpt.approxFactor)
		approxFactor = cnceOpt.approxFactor;
	else
		approxFactor = 30;
	end
	N = size(dh.X, 1);
	kappa = cnceOpt.kappa;

     % Set model weights to given parameter value
	if ~isempty(theta)
		snl.updateParameters(theta);
	end
	if ~dh.isOnSphere, dh.moveToSphere(); end
	if ~nh.isOnSphere, nh.moveToSphere(); end

	% Forward pass for different activation functions
	if strcmp(snl.activator, 'LinearLog')
		V1x = dh.X * snl.W'; 							% [N x nNeurons1]
		V1y = nh.Y * snl.W';							% [kappa * N x nNeurons1]
		V2x = V1x.^2; 									% [N x nNeurons1]
		V2y = V1y.^2; 									% [kappa * N x nNeurons1]
		V3x = bsxfun(@plus, log(V2x + 1), snl.b'); 		% [N x nNeurons1]
		V3y = bsxfun(@plus, log(V2y + 1), snl.b'); 		% [kappa * N x nNeurons1]
	elseif strcmp(snl.activator, 'QuadraticLog')
		V1x = dh.X.^2;
		V1y = nh.Y.^2;
		V2x = V1x * (snl.W.^2)'; 						% [N x nNeurons1]
		V2y = V1y * (snl.W.^2)'; 						% [kappa * N x nNeurons1]
		V3x = bsxfun(@plus, log(V2x + 1), snl.b'); 		% [N x nNeurons1]
		V3y = bsxfun(@plus, log(V2y + 1), snl.b'); 		% [kappa * N x nNeurons1]
	elseif strcmp(snl.activator, 'Max')
		V1x = max(dh.X * snl.W', 0); 					% [N x nNeurons1]
		V1y = max(nh.Y * snl.W', 0);                    % [kappa * N x nNeurons1]
		V3x = bsxfun(@plus, V1x, snl.b'); 				% [N x nNeurons1]
		V3y = bsxfun(@plus, V1y, snl.b'); 				% [kappa * N x nNeurons1]
	else
		error('Unrecognized SingleNeuronLayer activator')
	end

	% Sum to "probability"
	logPhix = sum(rectif(V3x), 2); 				% [N x 1]
	logPhiy = sum(rectif(V3y), 2); 				% [kappa * N x 1]
	% CNCE loss function
	Gxy = repmat(logPhix, kappa, 1) - logPhiy;
	Gxy = Gxy - nh.logP;
	lnegG 	= Gxy < -1 * approxFactor; 			% large negative values
	lposG 	= Gxy > approxFactor; 				% very large positive values
	midG 	= ~lnegG  & ~lposG; 				% neither small nor large
	h = 1./ ( 1 +  exp(Gxy(midG)));
	loss = -1 * sum(log(1 - h));
	loss = loss - sum(Gxy(lnegG)); % large negative values are linear, large positive values 0
	loss = loss / (0.5 * kappa *N);

	% Backwards
	if nargout > 1
		if isempty(snl.indices)
			grad_t = [];
			return
		end

		dJ_dG = zeros(N*kappa, 1);
		dJ_dG(midG) = h;
		dJ_dG(lposG) = exp(-1 * Gxy(lposG) ); % Close to zero
		dJ_dG(lnegG) = 1;
		
		% Backward -- Noise term
		dJ_dfV	 = bsxfun(@times, dJ_dG, rectifDot(V3y(:, snl.indices))); 	%[N*kappa x nNeurons1Var]
		if strcmp(snl.activator, 'LinearLog')
			dJ_dV2 		= dJ_dfV./(V2y(:, snl.indices) + 1); 				%[N*kappa x nNeurons1Var]
			dJ_dV1		= dJ_dV2 .* V1y(:, snl.indices); 					%[N*kappa x nNeurons1Var]
			grad_W_y 	= -2 * 2 * (dJ_dV1' * nh.Y) / (N * kappa);          %[nNeurons1Var x D]
		elseif strcmp(snl.activator, 'QuadraticLog')
			dJ_dV2 		= dJ_dfV./(V2y(:, snl.indices) + 1); 				%[N*kappa x nNeurons1Var]
			grad_W_y 	= -2 * 2 * (dJ_dV2' * V1y) .* ...
						snl.W(snl.indices, :) / (N * kappa); 				%[nNeurons1Var x D]
		elseif strcmp(snl.activator, 'Max')
			grad_W_y 	= -2 * (dJ_dfV .* (V1y(:, snl.indices) > 0))' ...
							* nh.Y / (N * kappa);
		else
			error('Unrecognized SingleNeuronLayer activator')
		end
		grad_b_y 	= -2 * sum(dJ_dfV, 1) / (N * kappa);						%[1 x nNeurons1Var]
		
		% Backward -- Data term
		dJ_dG_x 	= sum(reshape(dJ_dG, N, kappa), 2);							% [N x 1]
		dJ_dfV		= bsxfun(@times, dJ_dG_x, rectifDot(V3x(:, snl.indices)));	%[N x nNeurons1Var]
		if strcmp(snl.activator, 'LinearLog')
			dJ_dV2 		= dJ_dfV./(V2x(:,  snl.indices) + 1); 					%[N x nNeurons1Var]
			dJ_dV1		= dJ_dV2 .* V1x(:, snl.indices);						%[N x nNeurons1Var]
			grad_W_x 	= -2 * 2 * (dJ_dV1' * dh.X) / (N * kappa);              %[nNeurons1Var x D]
		elseif strcmp(snl.activator, 'QuadraticLog')
			dJ_dV2 		= dJ_dfV./(V2x(:, snl.indices) + 1);                    %[N*kappa x nNeurons1Var]
			grad_W_x 	= -2 * 2 * (dJ_dV2' * V1x) .* ...
						snl.W(snl.indices, :) / (N * kappa); 					%[nNeurons1Var x D]
		elseif strcmp(snl.activator, 'Max')
			grad_W_x 	= -2 * (dJ_dfV .* (V1x(:, snl.indices) > 0))' * dh.X / (N * kappa);
		else
			error('Unrecognized SingleNeuronLayer activator')
		end
		grad_b_x 	= -2 * sum(dJ_dfV, 1) / (N * kappa); 						%[1 x nNeurons2Var]

		% Combining data and noise term
		grad_W = grad_W_x - grad_W_y;
		grad_b = grad_b_x - grad_b_y;
		grad_t = [ grad_W(:); grad_b'];
	end
end


