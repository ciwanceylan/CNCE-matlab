classdef DoubleNeuronLayer < handle
    %NeuronLayer Contains two weight matrices, one layer of the neural image model.
	%   Inherits from handle so that we can pass it by reference

    properties
		W	% [nNeurons1 x D] The parameters (weights) of the first layer
		R	% [nNeurons2 x nNeurons1] The parameters (weights) of the second layer
		b	% The bias parameters of the second layer
		indices % {neuronIdx1, neuronIdx2} indices of variable neurons
		layerDimensions % [D, nNeurons1, nNeurons2]
		isInitialized % true if model has been initialized
		activator % Name of choice of activator function
    end % ------------------- End of properties-----------------

	properties (Hidden = true)
		hidden
		% U
		% Hstd
		% V1
		% V2
		% Q
	end

    methods
	%===================================================================
	% Class utilities
	%===================================================================
		function obj = DoubleNeuronLayer(init)
			% ImageModel constructor. Can be initialized with ImageModel object or a .mat file
			%	containing a struct named savedimagemodel which contains all of the ImageModel fields.
			% 	Copy constructor used is init is an ImageModel
			if nargin == 0
				obj.W = [];
				obj.R = [];
				obj.b	= [];
				obj.indices = {};
				obj.activator = [];
				obj.layerDimensions	= [];
			elseif isa(init, 'DoubleNeuronLayer')
				obj.W = init.W;
				obj.R = init.R;
				obj.b = init.b;
				obj.indices = init.indices;
				obj.activator = init.activator;
				obj.layerDimensions = init.layerDimensions;
			elseif isa(init, 'char') || isa(init, 'string')
				fileObj = load(init);
				obj.W = fileObj.savedimagemodel.W;
				obj.R = fileObj.savedimagemodel.R;
				obj.b = fileObj.savedimagemodel.b;
				obj.indices = fileObj.savedimagemodel.indices;
				obj.activator = fileObj.savedimagemodel.activator;
				obj.layerDimensions = fileObj.savedimagemodel.layerDimensions;
				clear fileObj
			elseif isa(init, 'cell')
				obj.initializeModel(init{1}, init{2});
				obj.activator = init{3};
				obj.initializeParameters();
			elseif isa(init, 'struct')
				obj.initializeModel(init.layerDimensions, init.indices);
				obj.initializeParameters();
			else
				error('Unknown input to DoubleNeuronLayer contructor.')
			end
			obj.isInitialized = obj.checkIfInitialized();
		end

		function isInitialized = checkIfInitialized(obj)
			% Checks that all fields are initialized.
			if isempty(obj.layerDimensions) || isempty(obj.W) || isempty(obj.b) ||...
				isempty(obj.indices) || isempty(obj.R)
				isInitialized = false;
			else
				isInitialized = true;
			end
		end

		function initializeModel(obj, layerDimensions, indices)
			% Initializes empty matrices for all layers
			if length(layerDimensions) ~= 3, error('DoubleNeuronLayer only supports 2 layers.'), end
			if length(indices) ~= 2
				error('Length of indices must be 2.')
			end
			obj.indices = indices;
			obj.layerDimensions = layerDimensions;
			obj.W = zeros(layerDimensions(2), layerDimensions(1));
			obj.R = zeros(layerDimensions(3), layerDimensions(2));
			for l = 1:2
				if size(indices{l}, 2) > 1,	error('indices must contain column vectors.'), end
				if ~(strcmp(indices{l}, ':') || isempty(indices{l})) ...
                        && max(indices{l}) > layerDimensions(l + 1)
					error(['indices for layer ', num2str(l), ' exceeds nNeurons'])
				end
			end
			obj.b = zeros(layerDimensions(end), 1);
			obj.isInitialized = true;
		end

		function initializeParameters(obj, type, layer)
			% Initializes the model with random parameters
			D = obj.layerDimensions(1);
			if ~obj.isInitialized, error('initializeParameters: ImageModel not initialized!'); end
			if nargin < 3 || isempty(layer), layer = 'all'; end
			if nargin < 2 || isempty(type), type = 'all'; end
			nNeurons = obj.getnNeurons(type);
			if strcmp(layer, 'all') || layer == 1
				obj.W(obj.getIdxW(type)) = (1/sqrt(D - 1)) * randn(nNeurons(1)*D, 1);
			end
			if strcmp(layer, 'all') || layer == 2
				% Different initialisation for different activators
				if (strcmp(obj.activator, 'LinearLinearLog') || ...
					strcmp(obj.activator, 'LinearLinearMaxLog') || ...
					strcmp(obj.activator, 'LinearLinearMax'))
					fprintf('LinearLinearLog init<--------------\n')
					obj.R(obj.getIdxR(type)) = ...
						(1/sqrt(D - 1)) * ( 1 +  0.1 * randn(nNeurons(2)*obj.layerDimensions(2), 1));
					obj.b(obj.getBiasInx(type)) = -ones(nNeurons(2) ,1)-rand(nNeurons(2) ,1);% b < 0
				elseif strcmp(obj.activator, 'LinearLinear')
					fprintf('LinearLinear init<--------------\n')
					obj.R(obj.getIdxR(type)) = ...
						(1/sqrt(D - 1)) * sqrt(0.01 * rand(nNeurons(2)*obj.layerDimensions(2), 1));
					obj.b(obj.getBiasInx(type)) = ones(nNeurons(2) ,1)-rand(nNeurons(2) ,1); % b > 0
				else
					obj.R(obj.getIdxR(type)) = ...
						(1/sqrt(D - 1)) * sqrt(0.01 * rand(nNeurons(2)*obj.layerDimensions(2), 1));
					obj.b(obj.getBiasInx(type)) = -ones(nNeurons(2) ,1)-rand(nNeurons(2) ,1);% b < 0
				end
			end
		end

		function clearHidden(obj)
			% clearHidden Clears intermediate variables to free memory
			obj.hidden.U = []; obj.hidden.Hstd = []; obj.hidden.V1 = [];
			obj.hidden.V2 = []; obj.hidden.Q = [];
		end

		function saveModel(savedimagemodel, filename)
			% Saves the ImageModel under the name 'savedimagemodel'
			save(filename, 'savedimagemodel')
		end

		function idx = getIdxW(obj, type)
			%getIdxW Get the indices of the W matrix.
			%	Type - if 'variable' only indices of parameters marked by indices vectors
			%		if 'all' returns ':'
			if ~ (strcmp(type, 'all') || strcmp(type, 'variable')), error('type not suported.'), end
			idx = obj.indices{1};
			if strcmp(type, 'all') || strcmp(idx, ':')
				idx = ':';
				return
			elseif isempty(idx)
				return
			elseif strcmp(type, 'variable')
				nNeurons = obj.getnNeurons(type, 1);
				nInputs = obj.getnInputs(1);
				idx = sub2ind(size(obj.W), ...
					repmat(idx, [nInputs, 1]), ...
					repelem((1:nInputs)', nNeurons));
			else
				error('Unknown type.');
			end
		end

		function idx = getIdxR(obj, type)
			%getIdxR Get the indices of the R matrix.
			%	Type - if 'variable' only indices of parameters marked by indices vectors
			%		if 'all' returns ':'
			if ~ (strcmp(type, 'all') || strcmp(type, 'variable')), error('type not suported.'), end
			idx = obj.indices{2};
			if strcmp(type, 'all') || strcmp(idx, ':')
				idx = ':';
				return
			elseif isempty(idx)
				return
			elseif strcmp(type, 'variable')
				nNeurons = obj.getnNeurons(type, 2);
				nInputs = obj.getnInputs(2);
				idx = sub2ind(size(obj.R), ...
					repmat(idx, [nInputs, 1]), ...
					repelem((1:nInputs)', nNeurons));
			else
				error('Unknown type.');
			end
		end

		function idx = getBiasInx(obj, type)
			%getBiasInx Returns the indices of the bias parameters, either ('variable' or 'all')
			idx = obj.indices{2};
			if strcmp(type, 'all') || strcmp(idx, ':')
				idx = ':';
			end
		end

		function nParameters = getnParameters(obj, type, layer)
			%nParameters Counts the number of parameters for a given layer and type ('variable', 'all')
			if nargin < 3 || isempty(layer), layer = 'all'; end
			if nargin < 2 || isempty(type), type = 'all'; end
			nInputs = obj.getnInputs(layer);
			nNeurons = obj.getnNeurons(type, layer);
			nParameters = nInputs * nNeurons; % nInputs is row, nNeurons is column
			if strcmp(layer, 'all') || layer == 2
				nParameters = nParameters + nNeurons(end);
			end
		end

		function nNeurons = getnNeurons(obj, type, layer)
			%nNeurons Counts the number of neurons for a given layer and type ('variable', 'all')
			if nargin < 3 || isempty(layer), layer = 'all'; end
			if nargin < 2 || isempty(type), type = 'all'; end
			nNeurons = zeros(2, 1);
			for l = 1:2 % avoid l = nLayers
				neuIdx = obj.indices{l};
				if strcmp(type, 'all') || strcmp(neuIdx, ':')
					nNeurons(l) = obj.layerDimensions(l + 1);
				elseif strcmp(type, 'variable')
					nNeurons(l) = numel(obj.indices{l});
				else
					error('Unknown type');
				end
			end
			if ~strcmp(layer, 'all'), nNeurons = nNeurons(layer); end
		end

		function nInputs = getnInputs(obj, layer)
			if nargin < 2 || isempty(layer), layer = 'all'; end
			nInputs = obj.layerDimensions(1:2);
			if ~strcmp(layer, 'all'), nInputs = nInputs(layer); end
		end

		function theta = getTheta(obj, type, layer)
			% Returns parameters as a theta vector.
			% 	Type is either 'all' or 'variable'
			% 	layer is  (1, 2, ... or 'all').
			% 	'all' 1 will return all parameters in layer 1, 'layer' 'all' will rtn all parameters
			%	'variable' 1 will return the parameters in layer 1 corresponding to the numbers in indices.
			if nargin < 3 || isempty(layer), layer = 'all'; end
			if nargin < 2 || isempty(type), type = 'all'; end
			nInputs = obj.getnInputs();
			nNeurons = obj.getnNeurons(type);
			theta = zeros(obj.getnParameters(type, layer), 1);
			tend = 0; tbegin = 1;
			if strcmp(layer, 'all') || layer == 1
				tend = tend + nNeurons(1)*nInputs(1);
				theta(tbegin:tend) = obj.W(obj.getIdxW(type));
				tbegin = tbegin + nNeurons(1)*nInputs(1);
			end
			if strcmp(layer, 'all') || layer == 2
				tend = tend + nNeurons(2)*nInputs(2);
				theta(tbegin:tend) = obj.R(obj.getIdxR(type));
				tbegin = tbegin + nNeurons(2)*nInputs(2);
				tend = tend + nNeurons(2);
				theta(tbegin:tend) = obj.b(obj.getBiasInx(type));
			end
		end

		function extendR(obj, dnl)
			% extendR concatenate with another DoubleNeuronLayer to extend the second layer R
			nN1 = size(obj.R, 1);
			nN2 = size(dnl.R, 1);
			obj.R = cat(1, obj.R, dnl.R);
			obj.b = cat(1, obj.b, dnl.b);
			obj.layerDimensions(3) = size(obj.R, 1);
			% If not all units should be active
			if strcmp(obj.indices{2}, ':')
				if ~strcmp(dnl.indices{2}, ':')
					obj.indices{2} = cat(1, (1:nN1)', dnl.indices{2} + nN1);
				end % else keep ':'
			else
				if ~strcmp(dnl.indices{2}, ':')
					obj.indices{2} = cat(1, obj.indices{2}, dnl.indices{2} + nN1);
				else
					obj.indices{2} = cat(1, obj.indices{2}, (1:nN2)' + nN1);
				end
			end
		end

		function extendW(obj, dnl)
			% extendW concatenate with another DoubleNeuronLayer to extend the first layer W
			nN1 = size(obj.W, 1);
			nN2 = size(dnl.W, 1);
			obj.W = cat(1, obj.W, dnl.W);
			obj.R = cat(2, obj.R, dnl.R);
			obj.layerDimensions(2) = size(obj.W, 1);
			% If not all units should be active
			if strcmp(obj.indices{1}, ':')
				if ~strcmp(dnl.indices{1}, ':')
					obj.indices{1} = cat(1, (1:nN1)', dnl.indices{1} + nN1);
				end % else keep ':'
			else
				if ~strcmp(dnl.indices{1}, ':')
					obj.indices{1} = cat(1, obj.indices{1}, dnl.indices{1} + nN1);
				else
					obj.indices{1} = cat(1, obj.indices{1}, (1:nN2)' + nN1);
				end
			end
		end

		function updateParameters(obj, theta, type)
			%updateParameters The parameters W, R and b are updated accroding to
			%	theta = [W(:); R(:); b]
			%	type - if value 'variable' only the units marked by the indices attribute
			%			is updated and the rest are fixed.
			%			if value 'all' every parameter will be updated
			if nargin < 3 || isempty(type), type = 'variable'; end
			nNeurons = obj.getnNeurons(type);
			nInputs = obj.getnInputs();
			nWeights = nInputs .* nNeurons';
			% W
			obj.W(obj.getIdxW(type)) = theta(1:nWeights(1));
			theta = theta(nWeights(1) + 1:end);
			% R
			obj.R(obj.getIdxR(type)) = theta(1:nWeights(2));
			theta = theta(nWeights(2) + 1:end);
			% b
			obj.b(obj.getBiasInx(type)) = theta(1:nNeurons(2));
			theta = theta(nNeurons(2) + 1: end);
			if ~isempty(theta)
				warning([num2str(length(theta)), ' elements in theta were unused!'])
			end
		end
	%===================================================================
	%	EVALUATION AND GRADIENTS
	%===================================================================

		function V3 = forwardPass(obj, Z, inx)
			if nargin < 3, inx = ':'; end
			% Forward propagation
			% Normalise Z
			D = obj.layerDimensions(1);
			H = bsxfun(@minus, Z, mean(Z, 2)); % [N x D]
			obj.hidden.Hstd = sqrt(sum(H.^2, 2)) / sqrt(D - 1) + 1e-6; % [N x 1] / sqrt(D - 1)
			obj.hidden.U = bsxfun(@rdivide, H, obj.hidden.Hstd); % [N x D]
			obj.hidden.Q = obj.R(inx, :).^2; % [nNeurons2 x nNeurons1]
			if strcmp(obj.activator, 'LinearQuadraticLog')
				% Linear term
				obj.hidden.V1 = obj.hidden.U * obj.W'; % [N x nNeurons1]
				obj.hidden.V2 = (obj.hidden.V1.^2) * obj.hidden.Q'; % [N x nNeurons2]
				V3 = log(obj.hidden.V2 + 1); % [N x nNeurons2]
			elseif strcmp(obj.activator, 'MaxQuadraticLog')
				obj.hidden.V1 = max(obj.hidden.U*obj.W', 0); % [N x nNeurons1]
				obj.hidden.V2 = (obj.hidden.V1.^2) * obj.hidden.Q'; % [N x nNeurons2]
				V3 = log(obj.hidden.V2 + 1); % [N x nNeurons2]
			elseif strcmp(obj.activator, 'LinearLinearLog')
				obj.hidden.V1 = obj.hidden.U * obj.W';
				obj.hidden.V2 = obj.hidden.V1 * obj.hidden.Q';
				V3 = log(obj.hidden.V2.^2 + 1);
			elseif strcmp(obj.activator, 'LinearLinearMaxLog')
				obj.hidden.V1 = obj.hidden.U * obj.W';
				obj.hidden.V2 = obj.hidden.V1 * obj.hidden.Q';
				V3 = log(abs(obj.hidden.V2) + 1);
			elseif strcmp(obj.activator, 'LinearLinearMax')
				obj.hidden.V1 = obj.hidden.U * obj.W';
				obj.hidden.V2 = obj.hidden.V1 * obj.hidden.Q';
				V3 = max(obj.hidden.V2, 0);
			elseif strcmp(obj.activator, 'LinearLinear')
				obj.hidden.V1 = obj.hidden.U * obj.W';
				obj.hidden.V2 = obj.hidden.V1 * obj.hidden.Q';
				V3 = obj.hidden.V2;
			else
				error(['Unrecognized DoubleNeuronLayer activator: ', obj.activator])
			end
		end

		function [logPhi, V3] = pdf(obj, Z)
			V3 = bsxfun(@plus, obj.forwardPass(Z), obj.b');
			logPhi 	= sum(rectif(V3), 2); % [N x 1]
		end

		function [out1, out2] = gradTheta(obj, z, theta)
			%gradTheta Parameter gradient with forward pass. Updates the parameters of the layer
			%	z - [1 x D] vector with a point in state
			%	theta - [nParameters x 1] the parameter values for which to calculate the gradient
			%			weights and bias values are updated accroding to theta = [W(:); R(:); b]
			%	out1 - Scalar result of forward propagation
			%	out2 - [1 x nParameters] The state space gradient
			if ~isequal(size(z), [1, obj.layerDimensions(1)])
				error(['Can only handle one image in ', num2str(obj.layerDimensions(1)) , ' dimensions'])
			end
			if nargin > 2, obj.updateParameters(theta); end
			% Forward pass
			V3 = obj.forwardPass(z) + obj.b';
			if nargout > 1
				out1 = sum(rectif(V3), 2); % [1 x 1]
			end

			% Back prop.
			dJ_dfV	 = rectifDot(V3); % [1 x nNeurons1Var]
			if strcmp(obj.activator, 'LinearQuadraticLog') || strcmp(obj.activator, 'MaxQuadraticLog')
				dJ_dV2 	= dJ_dfV ./ (obj.hidden.V2 + 1); % [1 x nNeurons2Var]
				grad_R	= 2 * ((dJ_dV2(:, obj.indices{2}))' * obj.hidden.V1.^2) .* obj.R(obj.indices{2}, :);
				dJ_dV1 	= (dJ_dV2 * obj.hidden.Q(:, obj.indices{1})) .* (obj.hidden.V1(:, obj.indices{1})); % [1 x nNeurons1Var]
			elseif strcmp(obj.activator, 'LinearLinearLog')
				dJ_dV2 	= dJ_dfV .* ( obj.hidden.V2 ./ (obj.hidden.V2.^2 + 1)) ;
				grad_R  = 2 * 2 * ((dJ_dV2(:, obj.indices{2}))' * obj.hidden.V1) .* obj.R(obj.indices{2}, :);
				dJ_dV1 	= dJ_dV2 * obj.hidden.Q(:, obj.indices{1}); % [1 x nNeurons1Var]
			elseif strcmp(obj.activator, 'LinearLinearMaxLog')
				dJ_dV2 	= dJ_dfV .* ( sign(obj.hidden.V2) ./ (abs(obj.hidden.V2) + 1)) ;
				grad_R  = 2 * ((dJ_dV2(:, obj.indices{2}))' * obj.hidden.V1) .* obj.R(obj.indices{2}, :);
				dJ_dV1 	= 0.5 * dJ_dV2 * obj.hidden.Q(:, obj.indices{1}); % [1 x nNeurons1Var]
			elseif strcmp(obj.activator, 'LinearLinearMax')
				dJ_dV2 	= dJ_dfV .* (obj.hidden.V2 > 0);
				grad_R  = 2 * ((dJ_dV2(:, obj.indices{2}))' * obj.hidden.V1) .* obj.R(obj.indices{2}, :);
				dJ_dV1 	= 0.5 * dJ_dV2 * obj.hidden.Q(:, obj.indices{1}); % [1 x nNeurons1Var]
			elseif strcmp(obj.activator, 'LinearLinear')
				dJ_dV2 	= dJ_dfV;
				grad_R  = 2 * ((dJ_dV2(:, obj.indices{2}))' * obj.hidden.V1) .* obj.R(obj.indices{2}, :);
				dJ_dV1 	= 0.5 * dJ_dV2 * obj.hidden.Q(:, obj.indices{1}); % [1 x nNeurons1Var]
			else
				error('Unrecognized activator') % sanity check
			end
			grad_W	= 2 * dJ_dV1' * obj.hidden.U; %[nNeurons1Var x D]
			grad_b = dJ_dfV(:, obj.indices{2});	%[1 x nNeurons1Var]

			if nargout > 1
				out2 = [grad_W(:); grad_R(:); grad_b'];
			else
				out1 = [grad_W(:); grad_R(:); grad_b'];
			end
			obj.clearHidden();
		end

		function [out1, out2] = gradU(obj, Z)
			%gradU State gradient with forward pass
			%	Z - [N x D] matrix with N points in state space
			%	inxL - index for the units to calculate the gradient
			%	out1 - [N x 1] Result of forward propagation
			%	out2 - [N x D] The state space gradient

			D = obj.layerDimensions(1);
			% Forward propagation
			V3 = bsxfun(@plus, obj.forwardPass(Z), obj.b'); % [N x nNeurons2]
			if nargout > 1
				out1 = sum(rectif(V3), 2); % [N x 1]
			end

			% Back propagation
			if strcmp(obj.activator, 'LinearQuadraticLog') || strcmp(obj.activator, 'MaxQuadraticLog')
				fDotV2 	= 2 * (rectifDot(V3) ./ (obj.hidden.V2 + 1));  % [N x nNeurons2]
				fDotV1	= (fDotV2 * (obj.hidden.Q) ) .* obj.hidden.V1; % [N x nNeurons1]
			elseif strcmp(obj.activator, 'LinearLinearLog')
				fDotV2 	= 2 * rectifDot(V3) .* (obj.hidden.V2 ./ (obj.hidden.V2.^2 + 1));  % [N x nNeurons2]
				fDotV1  = fDotV2 * obj.hidden.Q;
			elseif strcmp(obj.activator, 'LinearLinearMaxLog')
				fDotV2 	= rectifDot(V3) .* (sign(obj.hidden.V2) ./ (abs(obj.hidden.V2) + 1));  % [N x nNeurons2]
				fDotV1  = fDotV2 * obj.hidden.Q;
			elseif strcmp(obj.activator, 'LinearLinearMax')
				fDotV2 	= rectifDot(V3) .* (obj.hidden.V2 > 0);  % [N x nNeurons2]
				fDotV1  = fDotV2 * obj.hidden.Q;
			elseif strcmp(obj.activator, 'LinearLinear')
				fDotV2 	= rectifDot(V3);  % [N x nNeurons2]
				fDotV1  = fDotV2 * obj.hidden.Q;
			else
				error('Unrecognized DoubleNeuronLayer activator')
			end
			fDotU1	= bsxfun(@times, dot(fDotV1, obj.hidden.V1, 2), obj.hidden.U / (D - 1)); % [N x D]
			fDotU2	= fDotV1 *  (obj.W * (eye(D) - (1/D)*ones(D)));
			if nargout > 1
				out2 = bsxfun(@rdivide, fDotU2 - fDotU1, obj.hidden.Hstd); % [N x D]
			else
				out1 = bsxfun(@rdivide, fDotU2 - fDotU1, obj.hidden.Hstd); % [N x D]
			end
			obj.clearHidden();
		end

		function gradU_ = gradV3(obj, V3, inxL)
			%gradV3 State gradient without forward pass
			%	V3 - [N x length(inxL)] forward output value to backrpropagate
			%	inxL - index for the units to calculate the gradient
			%	gradU_ - [N x D] The state space gradient
			D = obj.layerDimensions(1);
			if nargin < 3 || isempty(inxL)
				dJ = rectifDot(V3);
				inx = ':';
			else
				dJ = V3;
				inx = inxL;
			end

			% Back propagation
			if strcmp(obj.activator, 'LinearQuadraticLog') || strcmp(obj.activator, 'MaxQuadraticLog')
				fDotV2 	= 2 * (dJ ./ (obj.hidden.V2(:,inx) + 1));  % [N x nNeurons2]
				fDotV1	= (fDotV2 * (obj.hidden.Q(inx, :)) ) .* obj.hidden.V1; % [N x nNeurons1]
			elseif strcmp(obj.activator, 'LinearLinearLog')
				fDotV2 	= 2 * dJ .* (obj.hidden.V2(:,inx) ./ (obj.hidden.V2(:,inx).^2 + 1));  % [N x nNeurons2]
				fDotV1  = fDotV2 * obj.hidden.Q(inx, :);
			elseif strcmp(obj.activator, 'LinearLinearMaxLog')
				fDotV2 	= dJ .* (sign(obj.hidden.V2(:,inx)) ./ (abs(obj.hidden.V2(:,inx)) + 1));  % [N x nNeurons2]
				fDotV1  = fDotV2 * obj.hidden.Q(inx, :);
			elseif strcmp(obj.activator, 'LinearLinearMax')
				fDotV2 	= dJ .* (obj.hidden.V2(:,inx) > 0);  % [N x nNeurons2]
				fDotV1  = fDotV2 * obj.hidden.Q(inx, :);
			elseif strcmp(obj.activator, 'LinearLinear')
				fDotV2 	= dJ;  % [N x nNeurons2]
				fDotV1  = fDotV2 * obj.hidden.Q(inx, :);
			else
				error('Unrecognized DoubleNeuronLayer activator')
			end
			fDotU1	= bsxfun(@times, dot(fDotV1, obj.hidden.V1, 2), obj.hidden.U / (D - 1)); % [N x D]
			fDotU2	= fDotV1 *  (obj.W * (eye(D) - (1/D)*ones(D)));
			gradU_	= bsxfun(@rdivide, fDotU2 - fDotU1, obj.hidden.Hstd); % [N x D]
			obj.clearHidden();
		end

    end %------------------- End of methods-----------------

end % ------------------- End of ImageModel class-----------------
