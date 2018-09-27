function [respMax, respMin] = response(I, cest, Lx, preproc_mode, intervention, activator)
%response(I, cest, Lx, preproc_mode, intervention, activator)
%       I:              input image (vectorized)
%       cest:           CNCEstimator contraining the estimated model
%       Lx:             Either 'L2', 'L3' or 'L4 depending on the layer
%                       visualisation desired
%       preproc_mode:   preprocessing mode. With 'relative_to_mean' the mean on the
%                       images in I are retained. With 'learning_pre_proc' the mean of the
%                       samples used to calculate the PCA is subtracted
%                       from I.
%       intervention:   0 (for nothing), 'noInhibition' to 
%                       remove negative 3rd layer weights
%       activator:      The desired activation function to use to calculate
%                       the response.
%       respMax:        The maximal response
%       respMin:        The minimal response
	if strcmp(Lx, 'L2')
		respMax = response_L2(I, cest, preproc_mode, activator);
		respMin = zeros(size(Y2)); % No minimal response for 2nd layer
	elseif strcmp(Lx, 'L3')
		[respMax, respMin] = response_L3(I, cest, intervention, preproc_mode, activator);
	elseif strcmp(Lx, 'L4')
		[respMax, respMin] = response_L4(I, cest, intervention, preproc_mode, activator);
	end
		

end

function [Y2] = response_L2(I, cest, preproc_mode, activator)
%response_L2 Get response(representation) from 2nd layer
    % Switch to desired activator
	if nargin > 3 && ~isempty(activator)
		tmpactivator = cest.neuronLayers{1}.activator;
		cest.neuronLayers{1}.activator = activator;
	end
	N = size(I, 1);
    % Transform image patches to PCA space
	if strcmp(preproc_mode,'relative_to_mean')
		Z = cest.dataProcessors{1}.toPCA(I, zeros(N, 1));
	elseif strcmp(preproc_mode,'learning_pre_proc')
		Z = bsxfun(@minus, I, cest.dataProcessors{1}.getSampleMean());
		Z = cest.dataProcessors{1}.toPCA(Z, zeros(N, 1));
    end
	
    % Evaluate model to get 2nd layer representation for each patch
	Y2 = cest.evalModel(Z, 1, ':');
    
    % Switch back to old activator (not exception save....)
	if nargin > 3 && ~isempty(activator)
		cest.neuronLayers{1}.activator = tmpactivator;
	end

end

function [Y3max, Y3min] = response_L3(I, cest, intervention, preproc_mode, activator)
%response_L3 Get response(representation) for each neuron in 3rd layer

	if nargin < 5 || isempty(activator), activator = cest.neuronLayers{2}.activator; end

    % Get ourput from 2nd layer
	[Y2] = response_L2(I, cest, preproc_mode);
    
    % Get whitening matrix for dimensionality reduction from 2nd to 3rd
    % layer
	Y2wh = cest.dataProcessors{2}.toPCA(Y2);
	
    % Calculate the 3rd layer response for different acitvators
	D = cest.neuronLayers{2}.layerDimensions(1);
	U3 = bsxfun(@minus, Y2wh, mean(Y2wh, 2)); % [N x D]
	s = sqrt(sum(U3.^2, 2)) / sqrt(D - 1) + 1e-6; % [N x 1]
	U3 = bsxfun(@rdivide, U3, s); % [N x D]
	
	W3 = cest.neuronLayers{2}.W;
	if strcmp(intervention, 'noInhibition')
		W3 = (W3 > 0) .* W3;
	end
	if strcmp(activator, 'LinearLog')
		Y3lin = U3 * W3';
		Y3max = log(Y3lin.^2 + 1);
		Y3min = zeros(size(Y3max));
	elseif strcmp(activator, 'Quad')
		Y3lin = U3 * W3';
		Y3max = Y3lin.^2;
		Y3min = zeros(size(Y3max));
	elseif strcmp(activator, 'QuadraticLog')
		Y3quad = U3.^2 * (W3.^2)';
		Y3max = log(Y3quad + 1);
		Y3min = zeros(size(Y3max));
	elseif strcmp(activator, 'Max')
		Y3lin = U3 * W3';
		Y3max = max(Y3lin, 0);
		Y3min = -min(Y3lin, 0);
	elseif strcmp(activator, 'Linear') 
		Y3max = U3 * W3';
		Y3min = -1 * Y3max;
	else
		Y3max = U3 * W3';
		Y3min = -1 * Y3max;
	end
end

function [Y4max, Y4min] = response_L4(I, cest, intervention, preproc_mode, activator)
%response_L4 Get response(representation) from 4th layer

	if nargin < 5 || isempty(activator), activator = cest.neuronLayers{2}.activator; end
	Q = cest.neuronLayers{2}.R.^2;
    
    % Get the representation depending on the activator
	if strcmp(activator, 'LinearQuadraticLog')
		[Y3max, ~] = response_L3(I, cest, intervention, preproc_mode, 'Linear');
		Y4quadMax = (Y3max.^2) * Q'; % [N x nNeurons2]
		Y4max = log(Y4quadMax + 1);
		Y4min = zeros(size(Y4max));
	elseif strcmp(activator, 'Quad')
		[Y3max, ~] = response_L3(I, cest, intervention, preproc_mode, 'Quad');
		Y4quadMax = Y3max * Q'; % [N x nNeurons2]
		Y4max = Y4quadMax;
		Y4min = zeros(size(Y4max));
	elseif strcmp(activator, 'MaxQuadraticLog')
		[Y3max, Y3min] = response_L3(I, cest, intervention, preproc_mode, 'Max');
		Y4quadMax = (Y3max.^2) * Q'; % [N x nNeurons2]
		Y4max = log(Y4quadMax + 1);
		Y4quadMin = (Y3min.^2) * Q'; % [N x nNeurons2]
		Y4min = log(Y4quadMin + 1);
	elseif strcmp(activator, 'Linear')
		[Y3max, Y3min] = response_L3(I, cest, intervention, preproc_mode, 'Linear');
		Y4max = Y3max * Q'; % [N x nNeurons2]
		Y4min = Y3min * Q'; % [N x nNeurons2]
	elseif strcmp(activator, 'LinearLinearLog')
		[Y3max, Y3min] = response_L3(I, cest, intervention, preproc_mode, 'Linear');
		Y4max = log( (Y3max * Q').^2 + 1 ); % [N x nNeurons2]
		Y4min = log( (Y3min * Q').^2 + 1 ); % [N x nNeurons2]
	elseif strcmp(activator, 'LinearLinearMaxLog')
		[Y3max, ~] = response_L3(I, cest, intervention, preproc_mode, 'Linear');
		Y4max = log( abs(Y3max * Q') + 1 ); % [N x nNeurons2]
		Y4min = log( abs(Y3max * Q')  + 1 ); % [N x nNeurons2]
	else
		warning('Running special LinearLog')
		[Y3max, ~] = response_L3(I, cest, intervention, preproc_mode, 'LinearLog');
		Y4quadMax = (Y3max.^2) * Q'; % [N x nNeurons2]
		Y4max = log(Y4quadMax + 1);
		Y4min = zeros(size(Y4max));
	end
end