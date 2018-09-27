classdef SingleNeuronLayer < handle
    %SingleNeuronLayer Contains two weight matrices, one layer of the neural image model.
	%   Inherits from handle so that we can pass it by reference
    
    properties
		W
		b
		indices % [] indices of variable neurons
		layerDimensions % [D, nNeurons1]
		isInitialized % true if model has been initialized
		activator 	% structure containing parameters for CNCE.
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
		function obj = SingleNeuronLayer(init)
			% ImageModel constructor. Can be initialized with ImageModel object or a .mat file
			%	containing a struct named savedimagemodel which contains all of the ImageModel fields.
			% 	Copy constructor used is init is an ImageModel
			if nargin == 0
				obj.W = [];
				obj.b	= [];
				obj.indices = [];
				obj.activator = '';
				obj.layerDimensions	= [];
			elseif isa(init, 'SingleNeuronLayer')
				obj.W = init.W;
				obj.b = init.b;
				obj.indices = init.indices;
				obj.activator = init.activator;
				obj.layerDimensions = init.layerDimensions;
			elseif isa(init, 'char') || isa(init, 'string')
				fileObj = load(init);
				obj.W = fileObj.savedimagemodel.W;
				obj.b = fileObj.savedimagemodel.b;
				obj.indices = fileObj.savedimagemodel.indices;
				obj.activator = fileObj.savedimagemodel.activator;
				obj.layerDimensions = fileObj.savedimagemodel.layerDimensions;
				clear fileObj
			elseif isa(init, 'cell')
				obj.initializeModel(init{1}, init{2});
				obj.initializeParameters();
				obj.activator = init{3};
			elseif isa(init, 'struct')
				obj.initializeModel(init.layerDimensions, init.indices);
				obj.initializeParameters();
			else
				error('Unknown input to SingleNeuronLayer contructor.')
			end
			obj.isInitialized = obj.checkIfInitialized();
		end

		function isInitialized = checkIfInitialized(obj)
			% Checks that all fields are initialized.
			if isempty(obj.layerDimensions) || isempty(obj.W) || isempty(obj.b) ||...
				isempty(obj.indices)
				isInitialized = false;
			else
				isInitialized = true;
			end
		end
		
		function initializeModel(obj, layerDimensions, indices)
			% Initializes empty matrices for all layers
			if length(layerDimensions) ~= 2, error('SingleNeuronLayer only supports 1 layers.'), end
			if size(indices, 2) ~= 1
				error('indices must be a column vector.')
			end
			obj.indices = indices;
			obj.layerDimensions = layerDimensions;
			obj.W = zeros(layerDimensions(2), layerDimensions(1));
			if ~(strcmp(indices, ':') || isempty(indices))...
                    && max(indices) > layerDimensions(2)
				error('indices exceeds nNeurons.')
			end
			obj.b = zeros(layerDimensions(2), 1);
			obj.isInitialized = true;
		end
		
		function initializeParameters(obj, type)
			% Initializes the model with random parameters
			if ~obj.isInitialized, error('SingleNeuronLayer not initialized!');end
			if nargin < 2 || isempty(type), type = 'all'; end
			nNeurons = obj.getnNeurons(type);
			idx = obj.getIdxW(type);
			D = obj.layerDimensions(1);
			obj.W(obj.getIdxW(type)) = (1/sqrt(D - 1)) * randn(nNeurons * D, 1);
			obj.b(obj.getBiasInx(type)) = -ones(nNeurons ,1)-rand(nNeurons ,1);
		end
		
		function clearHidden(obj)
			obj.hidden.U = []; obj.hidden.Hstd = []; obj.hidden.V1 = [];
			obj.hidden.V2 = []; obj.hidden.Q = [];
		end
		
		function saveModel(savedimagemodel, filename)
			% Saves the SingleNeuronLayer under the name 'savedimagemodel'
			save(filename, 'savedimagemodel')
		end
		
		function idx = getIdxW(obj, type)
			idx = obj.indices;
			if strcmp(type, 'all') || strcmp(idx, ':')
				idx = ':';
				return
			elseif isempty(idx)
				return
			elseif strcmp(type, 'variable')
				nNeurons = obj.getnNeurons(type);
				nInputs = obj.getnInputs();
				idx = sub2ind(size(obj.W), ...
					repmat(idx, [nInputs, 1]), ...
					repelem((1:nInputs)', nNeurons));
			else
				error('Unknown type.');
			end
		end
		
		function idx = getBiasInx(obj, type)
			idx = obj.indices;
			if strcmp(type, 'all') || strcmp(idx, ':')
				idx = ':'; 
			end
		end
		
		function nParameters = getnParameters(obj, type)
			if nargin < 2 || isempty(type), type = 'all'; end
			nInputs = obj.getnInputs();
			nNeurons = obj.getnNeurons(type);
			nParameters = nInputs * nNeurons + nNeurons; % nInputs is row, nNeurons is column
		end
		
		function nNeurons = getnNeurons(obj, type)
			if nargin < 2 || isempty(type), type = 'all'; end
			neuIdx = obj.indices;
			if strcmp(type, 'all') || strcmp(neuIdx, ':')
				nNeurons = obj.layerDimensions(2);
			elseif strcmp(type, 'variable')
				nNeurons = numel(neuIdx);
			else
				error('Unknown type');
			end
		end

		function nInputs = getnInputs(obj)
			nInputs = obj.layerDimensions(1);
		end
		
		function theta = getTheta(obj, type)
			% Returns parameters as a theta vector.
			% 	Type is either 'all' or 'variable'
			% 	layer is  (1, 2, ... or 'all').
			% 	'all' 1 will return all parameters in layer 1, 'layer' 'all' will rtn all parameters
			%	'variable' 1 will return the parameters in layer 1 corresponding to the numbers in indices.
			if nargin < 2 || isempty(type), type = 'all'; end
			nInputs = obj.getnInputs();
			nNeurons = obj.getnNeurons(type);
			theta = zeros(obj.getnParameters(type), 1);
			tend = 0; tbegin = 1;
			tend = tend + nNeurons*nInputs;
			theta(tbegin:tend) = obj.W(obj.getIdxW(type));
			tbegin = tbegin + nNeurons*nInputs;
			tend = tend + nNeurons;
			theta(tbegin:tend) = obj.b(obj.getBiasInx(type));
		end
		
		function extendW(obj, snl)
			% extendW concatenate with another SingleNeuronLayer to extend the first layer W
			nN1 = size(obj.W, 1);
			nN2 = size(snl.W, 1);
			obj.W = cat(1, obj.W, snl.W);
			obj.b = cat(1, obj.b, snl.b);
			obj.layerDimensions(2) = size(obj.W, 1);
			% If not all units should be active
			if strcmp(obj.indices, ':') 
				if ~strcmp(snl.indices, ':')
					obj.indices = cat(1, (1:nN1)', snl.indices + nN1);
				end % else keep ':'
			else
				if ~strcmp(snl.indices, ':')
					obj.indices = cat(1, obj.indices, snl.indices + nN1);
				else
					obj.indices = cat(1, obj.indices, (1:nN2)' + nN1);
				end
			end
		end

	%===================================================================
	% 
	%===================================================================
		
		function updateParameters(obj, theta, type)
			if nargin < 3 || isempty(type), type = 'variable'; end
			nNeurons = obj.getnNeurons(type);
			nInputs = obj.getnInputs();
			nWeights = nInputs * nNeurons;
			% W
			obj.W(obj.getIdxW(type)) = theta(1:nWeights);
			theta = theta(nWeights + 1:end);
			% b
			obj.b(obj.getBiasInx(type)) = theta(1:nNeurons);
			theta = theta(nNeurons + 1: end);
			if ~isempty(theta)
				warning([num2str(length(theta)), ' elements in theta were unused!'])
			end
		end
		
		function V3 = forwardPass(obj, Z, inx)
			if nargin < 3, inx = ':'; end
			% Forward propagation
			% Normalise Z
			D = obj.layerDimensions(1);
			H = bsxfun(@minus, Z, mean(Z, 2)); % [N x D]
			obj.hidden.Hstd = sqrt(sum(H.^2, 2)) / sqrt(D - 1) + 1e-6; % [N x 1]
			obj.hidden.U = bsxfun(@rdivide, H, obj.hidden.Hstd); % [N x D]
			if strcmp(obj.activator, 'LinearLog')
				% Linear term
				obj.hidden.V1 = obj.hidden.U * obj.W(inx, :)'; % [N x nNeurons1]
				obj.hidden.V2 = (obj.hidden.V1.^2); % [N x nNeurons1]
				V3 = log(obj.hidden.V2 + 1); % [N x nNeurons1]
			elseif strcmp(obj.activator, 'QuadraticLog')
				obj.hidden.Q = obj.W(inx, :).^2; % [N x D] V1 = Q
				obj.hidden.V2 = (obj.hidden.U.^2)*obj.hidden.Q'; % [N x nNeurons1]
				V3 = log(obj.hidden.V2 + 1); % [N x nNeurons1]
			elseif strcmp(obj.activator, 'Max')
				obj.hidden.V1 = max(obj.hidden.U*obj.W(inx,:)', 0);
				V3 = obj.hidden.V1;
			elseif strcmp(obj.activator, 'Linear')
				obj.hidden.V1 = obj.hidden.U*obj.W(inx,:)';
				V3 = obj.hidden.V1;
			else
				error('Unrecognized SingleNeuronLayer activator')
			end
		end
		
		function [logPhi, V3] = pdf(obj, Z)
			V3 = bsxfun(@plus, obj.forwardPass(Z), obj.b');
			logPhi 	= sum(rectif(V3), 2); % [N x 1]
			%obj.clearHidden();
		end
		
		function f = f_nonLinearity(obj, ut)
			D = obj.layerDimensions(1);
			Wnorm = sqrt(sum(obj.W.^2, 2)) .* sqrt(D - 1);
			
			if strcmp(obj.activator, 'LinearLog')
				y = bsxfun(@times, ut, Wnorm);
				y = y.^2;
				f = rectif(bsxfun(@plus, log(y + 1), obj.b));			
			elseif strcmp(obj.activator, 'QuadraticLog')
				y = bsxfun(@times, ut.^2, Wnorm.^2);
				f = rectif(bsxfun(@plus, log(y + 1), obj.b));	
			elseif strcmp(obj.activator, 'Max')
				y = bsxfun(@times, ut, Wnorm);
				y = max(y, 0);
				f = rectif(bsxfun(@plus, y, obj.b));
			else
				error('Unrecognized SingleNeuronLayer activator')
			end
		end
		
		function [out1, out2] = gradTheta(obj, z, theta)
			if ~isequal(size(z), [1, obj.layerDimensions(1)])
				error(['Can only handle one image in ', num2str(obj.layerDimensions(1)) , ' dimensions'])
			end
			if nargin > 2, obj.updateParameters(theta); end
			% Forward pass
			V3 = obj.forwardPass(z) + obj.b';
			if nargout > 1
				out1 = sum(rectif(V3), 2); % [1 x 1]
			end

			if isempty(obj.indices)
				out2 = [];
				obj.clearHidden();
				return
			end
			
			% Back prop.
			dJ_dfV	 = rectifDot(V3(:, obj.indices)); % [1 x nNeurons1Var]
			if strcmp(obj.activator, 'LinearLog')
				dJ_dV2 	= dJ_dfV./(obj.hidden.V2(:, obj.indices) + 1); % [1 x nNeurons1Var]
				dJ_dV1	= dJ_dV2 .* obj.hidden.V1(:, obj.indices); % [1 x nNeurons1Var]
				grad_W 	= 2 * (dJ_dV1' * obj.hidden.U ); 	% [nNeurons1Var x D]
			elseif strcmp(obj.activator, 'QuadraticLog')
				dJ_dV2 	= dJ_dfV./(obj.hidden.V2(:, obj.indices) + 1); %[1 x nNeurons1Var]
				grad_W	= 2 * (dJ_dV2' * (obj.hidden.U.^2)) .* obj.W(obj.indices, :); %[nNeurons1Var x D]
			elseif strcmp(obj.activator, 'Max')
				grad_W 	= (dJ_dfV .* (obj.hidden.V1(:, obj.indices) > 0))' * obj.hidden.U;
			elseif strcmp(obj.activator, 'Linear')
				grad_W 	= dJ_dfV' * obj.hidden.U;
			else
				error(['Unrecognized SingleNeuronLayer: ', obj.activator])
			end
			grad_b = dJ_dfV;	%[1 x nNeurons1Var]
			if nargout > 1
				out2 = [grad_W(:); grad_b'];
			else
				out1 = [grad_W(:); grad_b'];
			end
			obj.clearHidden();
		end
		
		function [out1, out2] = gradU(obj, Z)
			D = obj.layerDimensions(1);
			% Forward pass
			V3 = bsxfun(@plus, obj.forwardPass(Z), obj.b');
			if nargout > 1
				out1 = sum(rectif(V3), 2); % [N x 1]
			end
			
			% Back propagation
			if strcmp(obj.activator, 'LinearLog')
				fDotV2 	= 2 * (rectifDot(V3) ./ (obj.hidden.V2 + 1));  % [N x nNeurons1]
				fDotV1	= fDotV2 .* obj.hidden.V1; % [N x nNeurons1]
				fDotU1	= bsxfun(@times, dot(fDotV1, obj.hidden.V1, 2), obj.hidden.U / (D - 1)); % [N x D]
				%fDotU1	= bsxfun(@times, dot(fDotV1, obj.hidden.V1, 2), obj.hidden.U); % [N x D]
				fDotU2	= fDotV1 * ( obj.W * (eye(D) - (1/D)*ones(D)) );
			elseif strcmp(obj.activator, 'QuadraticLog')
				fDotV2 	= (rectifDot(V3) ./ (obj.hidden.V2 + 1));  % [N x nNeurons1]
				fDotV1 	= 2 * fDotV2 * obj.hidden.Q;
				fDotV1 	= fDotV1 .* obj.hidden.U; % [N x D]
				fDotU1 	= bsxfun(@times, dot(fDotV1, obj.hidden.U, 2), obj.hidden.U / (D - 1)); % [N x D]
				fDotU2	= fDotV1 * ( eye(D) - (1/D)*ones(D)); % [N x D]
			elseif strcmp(obj.activator, 'Max')
				fDotV3 	= rectifDot(V3); % [N x nNeurons]
				fDotV1 	= (fDotV3 .* (obj.hidden.V1 > 0)) * obj.W; % [N x D]
				fDotU1 	= bsxfun(@times, dot(fDotV1, obj.hidden.U, 2), obj.hidden.U / (D - 1)); % [N x D]
				fDotU2	= fDotV1 * (eye(D) - (1/D)*ones(D));
			elseif strcmp(obj.activator, 'Linear')
				fDotV3 	= rectifDot(V3); % [N x nNeurons]
				fDotV1 	= fDotV3 * obj.W; % [N x D]
				fDotU1 	= bsxfun(@times, dot(fDotV1, obj.hidden.U, 2), obj.hidden.U / (D - 1)); % [N x D]
				fDotU2	= fDotV1 * (eye(D) - (1/D)*ones(D));
			else
				error(['Unrecognized SingleNeuronLayer: ', obj.activator])
			end
			if nargout > 1
				out2 = bsxfun(@rdivide, fDotU2 - fDotU1, obj.hidden.Hstd); % [N x D]
			else
				out1 = bsxfun(@rdivide, fDotU2 - fDotU1, obj.hidden.Hstd); % [N x D]
			end
			obj.clearHidden();
		end
		
		function gradU_ = gradV3(obj, V3, inxL)
			D = obj.layerDimensions(1);
			if nargin < 3 || isempty(inxL)
				dJ = rectifDot(V3);
				inx = ':';
			else
				dJ = V3;
				inx = inxL;
			end
			
			% Back propagation
			if strcmp(obj.activator, 'LinearLog')
				fDotV2 	= 2 * dJ ./ (obj.hidden.V2(:,inx) + 1);  % [N x nNeurons1]
				fDotV1	= fDotV2 .* obj.hidden.V1(:, inx); % [N x nNeurons1]
				fDotU1	= bsxfun(@times, dot(fDotV1, obj.hidden.V1(:, inx), 2) , obj.hidden.U / (D - 1)); % [N x D]
				fDotU2	= fDotV1 * ( obj.W(inx, :) * (eye(D) - (1/D)*ones(D)) );
			elseif strcmp(obj.activator, 'QuadraticLog')
				fDotV2 	= dJ ./ (obj.hidden.V2(:,inx) + 1);  % [N x nNeurons1]
				fDotV1 	= 2 * fDotV2 * obj.hidden.Q(inx, :);
				fDotV1 	= fDotV1 .* obj.hidden.U; % [N x D]
				fDotU1 	= bsxfun(@times, dot(fDotV1, obj.hidden.U, 2), obj.hidden.U / (D - 1)); % [N x D]
				fDotU2	= fDotV1 * ( eye(D) - (1/D)*ones(D)); % [N x D]
			elseif strcmp(obj.activator, 'Max')
				fDotV3 	= dJ; % [N x nNeurons]
				fDotV1 	= (fDotV3 .* (obj.hidden.V1(:,inx) > 0)) * obj.W(inx, :); % [N x D]
				fDotU1 	= bsxfun(@times, dot(fDotV1, obj.hidden.U, 2), obj.hidden.U / (D - 1)); % [N x D]
				fDotU2	= fDotV1 * (eye(D) - (1/D)*ones(D));
			elseif strcmp(obj.activator, 'Linear')
				fDotV3 	= dJ; % [N x nNeurons]
				fDotV1 	= fDotV3 * obj.W(inx, :); % [N x D]
				fDotU1 	= bsxfun(@times, dot(fDotV1, obj.hidden.U, 2), obj.hidden.U / (D - 1)); % [N x D]
				fDotU2	= fDotV1 * (eye(D) - (1/D)*ones(D));
			else
				error(['Unrecognized SingleNeuronLayer: ', obj.activator])
			end
			gradU_	= bsxfun(@rdivide, fDotU2 - fDotU1, obj.hidden.Hstd); % [N x D]
			obj.clearHidden();
		end
		
		
	
    end %------------------- End of methods-----------------
    
end % ------------------- End of ImageModel class-----------------

