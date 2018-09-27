classdef DataHandle < handle   
	%DATAHANDLE For holding image patch data    
    properties
		X			% [N x D] matrix with image patch data
		meanX		% [N x 1] The local mean (DC-component) of the data
		stdX		% [N x 1] The standard deviation of the data 
		isOnSphere	% If the mean has been subtracted and the data rescaled or not
    end

    methods

		function obj = DataHandle()
			obj.isOnSphere = false;
		end

		function moveToSphere(obj)
			% Subtract the local mean and rescale the data. Save the mean and std to be able 
			%		to restore the original data later.
			D = size(obj.X,2);
			obj.meanX = mean(obj.X, 2); % [N x 1]
			obj.X = bsxfun(@minus, obj.X, obj.meanX); % [N x D]
			obj.stdX = sqrt(sum(obj.X.^2, 2)) / sqrt(D - 1) + 1e-6; % [N x 1]
			obj.X = bsxfun(@rdivide, obj.X, obj.stdX); % [N x D]
			obj.isOnSphere = true;
		end
		
		function revertFromSphere(obj)
			%revertFromSphere Restore the original data by reverting mean subtraction and scaling
			if obj.isOnSphere
				obj.X = bsxfun(@times, obj.X, obj.stdX);
				obj.X = bsxfun(@plus, obj.X, obj.meanX);
				obj.isOnSphere = false;
				fprintf('dataHandle: X reverted from sphere\n')
			else
				fprintf('dataHandle: X is not on sphere\n')
			end
		end

		function clearX(obj)
			obj.X = [];
			obj.meanX = [];
			obj.stdX = [];
			obj.isOnSphere = false;
		end
    end

end
