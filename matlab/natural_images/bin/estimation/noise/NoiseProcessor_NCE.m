classdef NoiseProcessor_NCE < handle
    %NOISEPROCESSOR_NCE Methods to sample noise batches for NCE.
    properties
		nu		% noise to data ratio
		opt		% options (nu)
	end
	
	methods
		function obj = NoiseProcessor_NCE(nceOpt)
			% NoiseProcessor_NCE nargin.nu only input needed
			if nargin == 0
			;
			else
				obj.nu = nceOpt.nu;
			end
			fprintf('NoiseProcessor initialized.\n')
		end

		function clearNoise(obj)
			% clearNoise Currently doesn't do anything, but would clear variables from memory
			; % placeholder
		end

		function [noiseHandle] = generateNoiseInter(obj, N, D)
			% generateNoise generates multivariate Gaussian noise
			fprintf('Generating noise (NCE - Inter) \n')
			noiseHandle = NoiseHandle();
			noiseHandle.Y =  randn(N * obj.nu, D);
			partFun = -(D/2) * log(2*pi);
			noiseHandle.logP = -0.5 * dot(noiseHandle.Y , noiseHandle.Y, 2); 
			noiseHandle.logP = bsxfun(@plus, noiseHandle.logP, partFun);% not currently used in loss
			fprintf('done!\n\n')
		end
		
		function [noiseHandle] = generateNoiseExtra(obj, nest, layer, N, D)
			%generateNoiseExtra Generates multivariate Gaussian noise in
			% in 
			fprintf('Generating noise (NCE - Extra) \n')
			noiseHandle = NoiseHandle();
			Y = randn(N * obj.nu, D);
			if layer > 1
				Y = nest.evalModelVars(Y, layer - 1, ':');
				Y = nest.dataProcessors{layer}.toPCA( Y );
			end
			noiseHandle.Y = Y;
			partFun = -(D/2) * log(2*pi);
			noiseHandle.logP = -0.5 * dot(noiseHandle.Y , noiseHandle.Y, 2); 
			noiseHandle.logP = bsxfun(@plus, noiseHandle.logP, partFun);% not currently used in loss
			
			fprintf('done!\n\n')
		end
	end
end