classdef NoiseProcessor < handle
    %NOISEPROCESSOR Methods to sample noise batches.
    properties
		base
		kappa
		gradUx
		epsBase
		aCNCE
		epsilon
		opt
	end
	
	methods
		function obj = NoiseProcessor(cnceOpt)
			if nargin == 0
			;
			else
				obj.kappa = cnceOpt.kappa;
				obj.epsBase = cnceOpt.epsBase;
				obj.aCNCE = cnceOpt.aCNCE;
				obj.epsilon = obj.epsBase;
				obj.opt = obj.getNoiseOpt(cnceOpt);
			end
			fprintf('NoiseProcessor initialized.\n')
		end
		
		function noiseOpt = getNoiseOpt(obj, cnceOpt)
			noiseOpt = cnceOpt;
			if ~isfield(noiseOpt, 'loss_inf'), noiseOpt.loss_inf = 2*log(2); end
			if ~isfield(noiseOpt, 'lossinf'), noiseOpt.lossinf = 0; end
			if ~isfield(noiseOpt, 'thrsUpper'), noiseOpt.thrsUpper = 0.2 * noiseOpt.loss_inf; end
			if ~isfield(noiseOpt, 'thrsLower'), noiseOpt.thrsLower = 0.05 * noiseOpt.loss_inf;end
			if ~isfield(noiseOpt, 'incRate'), noiseOpt.incRate = 0.15; end
			if ~isfield(noiseOpt, 'decRate'), noiseOpt.decRate = 0.2; end
			if ~isfield(noiseOpt, 'maxNoiseIter'), noiseOpt.maxNoiseIter = 20; end
			if ~isfield(noiseOpt, 'epsHardCap'), noiseOpt.epsHardCap = 4; end
		end
		
		function clearNoise(obj)
			obj.base = []; obj.gradUx = [];
		end

		function generateNoiseBase(obj, N, D)
			obj.base = randn(N * obj.kappa, D);
        end
        
        function [dh, nh, stats] = generateNoiseExtra(obj, cest, layer)
            %generateNoiseExtra Generates noise in data space (eg the
            %image PCA space) and passes the noise through the network
            %up until the layer currently being trained.

			fprintf('Generating CNCE noise (Extra) \n')
			obj.epsilon = obj.epsBase;
			objFunNoise = cest.getObjFunNoise(layer);
			[dh, nh] = obj.computeNoiseCEST(cest, layer, cest.dataHandle);
			loss = objFunNoise(dh, nh);
			epsf = 1; k = 1;
            
			% Three part condition: exceeded max iterations OR in correct loss interval OR exceeded
			% 	max allowed epsilon factor
			rangeCond = @(k, loss, epsf) ( (k < obj.opt.maxNoiseIter) ...
				&& (abs(obj.opt.loss_inf - loss) < obj.opt.thrsLower || loss < obj.opt.thrsUpper)  ...
				&& epsf < obj.opt.epsHardCap);
			upper = obj.opt.epsHardCap;
			lower = 0;
			fprintf('%d: loss: %.4f \t epf: %.4f\n', k, loss, epsf)
            
			 % Store the intermediate values of epsilon for analysis
			stats.lossEpsBase = loss;
			stats.lossEps = loss;
			stats.epsf = epsf;
            
            % Iterate until heuristic conditions are met 
			while rangeCond(k, loss, epsf)
                % If epsilon is too small....
				if (abs(obj.opt.loss_inf - loss) < obj.opt.thrsLower)
					lower = epsf;
                    % It is either increased slowly
                    % OR if there is a known value where epsilon is large
                    %   we set epsf to the average of the too large and too
                    %   small value
					if upper == obj.opt.epsHardCap
						epsf = (1 + obj.opt.incRate) * epsf;
					else
						epsf = (upper + lower) / 2;
                    end
                % IF epsilon is to large it is decreased rapidly
				elseif (loss < obj.opt.thrsUpper)
					upper = epsf;
					epsf = (upper + lower) / 2;
				else
					warning('Something went wrong with rangeCond...')
				end
				obj.epsilon = epsf * obj.epsBase;
				[dh, nh] = obj.computeNoiseCEST(cest, layer, cest.dataHandle);
				loss = objFunNoise(dh, nh);
				k = k + 1;
				stats.lossEps = [stats.lossEps; loss];
				stats.epsf = [stats.epsf; epsf];
				fprintf('%d: loss: %.4f \t epf: %.4f\n', k, loss, epsf)
			end
			if k > (obj.opt.maxNoiseIter - 1), warning('maxNoiseIter exceeded.'), end
			obj.gradUx = [];
			stats.lossEps = [stats.lossEps; loss];
			stats.epsf = [stats.epsf; epsf];
			stats.noiseIters = k;
			fprintf('done!\n\n')
        end

		function [noiseHandle, stats] = generateNoiseInter(obj, dataHandle, nl, objFunNoise)
            %generateNoiseInter Generates noise in latent space (eg in 
            % between the 2nd and 3rd layer. For the noise generation used
            % in the ICML 2018 paper see generateNoiseExtra

			if isempty(dataHandle), error('No data X provided.'); end
			fprintf('Generating CNCE noise (Inter) \n')
			obj.epsilon = obj.epsBase;
			noiseHandle = obj.computeNoise(dataHandle, nl);
			loss = objFunNoise(dataHandle, noiseHandle);
			epsf = 1; k = 1;
			% Three part condition: exceeded max iterations OR in correct loss interval OR exceeded
			% 	max allowed epsilon factor
			rangeCond = @(k, loss, epsf) ( (k < obj.opt.maxNoiseIter) ...
				&& (abs(obj.opt.loss_inf - loss) < obj.opt.thrsLower || loss < obj.opt.thrsUpper)  ...
				&& epsf < obj.opt.epsHardCap);
			upper = obj.opt.epsHardCap;
			lower = 0;
			fprintf('%d: loss: %.4f \t epf: %.4f\n', k, loss, epsf)

            % Store the intermediate values of epsilon for analysis
			stats.lossEpsBase = loss;
			stats.lossEps = loss;
			stats.epsf = epsf;
            
            % Iterate until heuristic conditions are met 
			while rangeCond(k, loss, epsf)
				if (abs(obj.opt.loss_inf - loss) < obj.opt.thrsLower)
					lower = epsf;
					if upper == obj.opt.epsHardCap
						epsf = (1 + obj.opt.incRate) * epsf;
					else
						epsf = (upper + lower) / 2;
					end
				elseif (loss < obj.opt.thrsUpper)
					upper = epsf;
					epsf = (upper + lower) / 2;
				else
					warning('Something went wrong with rangeCond...')
				end
				obj.epsilon = epsf * obj.epsBase;
				noiseHandle = obj.computeNoise(dataHandle, nl); % Re-calculate noise
				loss = objFunNoise(dataHandle, noiseHandle);
				k = k + 1;
				stats.lossEps = [stats.lossEps; loss];
				stats.epsf = [stats.epsf; epsf];
				fprintf('%d: loss: %.4f \t epf: %.4f\n', k, loss, epsf)
			end
			if k > (obj.opt.maxNoiseIter - 1), warning('maxNoiseIter exceeded.'), end
			obj.gradUx = [];
			stats.lossEps = [stats.lossEps; loss];
			stats.epsf = [stats.epsf; epsf];
			stats.noiseIters = k;
			fprintf('done!\n\n')
        end
        
        function [dataHandleOut, noiseHandle] = computeNoiseCEST(obj, cest, layer, dataHandle)
            %computerNoiseCEST Used by generateNoiseExtra to compute the noise
            %variables for a given value of Epsilon and possibly given iid
            %normal noise base
			if nargin < 3, layer = length(cest.neuronLayers); end
			if nargin < 4, dataHandle = cest.dataHandle; end

			if isempty(obj.base)
				[N, D] = size(dataHandle.X);
				obj.generateNoiseBase(N, D);
			end
			noiseHandle = NoiseHandle();
			dataHandleOut = DataHandle();
			dataHandle.revertFromSphere();
			if (obj.aCNCE)
				if isempty(obj.gradUx)
					[~, obj.gradUx, varsX] = cest.evalModel(dataHandle.X, layer);
				elseif layer > 1
					[~, varsX] = cest.evalModelVars(dataHandle.X, layer);
				end
				mu = dataHandle.X + (obj.epsilon^2 / 2) * obj.gradUx;
				Y = repmat(mu, obj.kappa, 1) + obj.epsilon * obj.base;
				[~, gradUy, varsY] = cest.evalModel(Y, layer);
				Px_Py = repmat(obj.gradUx, obj.kappa, 1) + gradUy;
				noiseHandle.logP = (-1*obj.epsilon/2) * ...
					(dot(Px_Py, (((obj.epsilon/4)*Px_Py) + obj.base),2));
			else
				Y = repmat(dataHandle.X, obj.kappa, 1) + obj.epsilon * obj.base;
				noiseHandle.logP = zeros(size(dataHandle.X,1)*obj.kappa, 1);
				if layer > 1
					[~, varsX] = cest.evalModelVars(dataHandle.X, layer);
					[~, varsY] = cest.evalModelVars(Y, layer);
				end
			end
			
			if layer > 1
				% Set data to the PCA transform of the output of second to last layer
				% likewise for noise
				dataHandleOut.X	= cest.dataProcessors{layer}.toPCA( varsX{layer - 1}.V3 );
				noiseHandle.Y 	= cest.dataProcessors{layer}.toPCA( varsY{layer - 1}.V3 );
			else
				dataHandleOut.X	= dataHandle.X;
				noiseHandle. Y = Y;
				
			end
			dataHandleOut.isOnSphere = false;
			noiseHandle.isOnSphere = false;
        end
        
        function noiseHandle = computeNoise(obj, dataHandle, neuronLayer)
            %computerNoise Used by generateNoiseInter to compute the noise
            %variables for a given value of Epsilon and possibly given iid
            %normal noise base
			if isempty(obj.base)
				[N, D] = size(dataHandle.X);
				obj.generateNoiseBase(N, D);
			end
			noiseHandle = NoiseHandle();
			dataHandle.revertFromSphere();
			if (obj.aCNCE)
				if isempty(obj.gradUx)
					obj.gradUx = neuronLayer.gradU(dataHandle.X);
				end
				mu 		= dataHandle.X + (obj.epsilon^2 / 2) * obj.gradUx;
				noiseHandle.Y = repmat(mu, obj.kappa, 1) + obj.epsilon * obj.base;
				gradUy 	= neuronLayer.gradU(noiseHandle.Y);
				Px_Py 	= repmat(obj.gradUx, obj.kappa, 1) + gradUy;
				noiseHandle.logP = (-1*obj.epsilon/2) * ...
					(dot(Px_Py, (((obj.epsilon/4)*Px_Py) + obj.base),2));
			else
				noiseHandle.Y = repmat(dataHandle.X, obj.kappa, 1) + obj.epsilon * obj.base;
				noiseHandle.logP = zeros(size(dataHandle.X,1)*obj.kappa, 1);
			end
			noiseHandle.isOnSphere = false;
        end

	end
end