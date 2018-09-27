classdef NoiseHandle < handle
    %NOISEHANDLE Summary of this class goes here
    %   Detailed explanation goes here

    properties
		Y
		logP
		meanY
		stdY
		isOnSphere
    end

    methods

		function obj = NoiseHandle()
			obj.isOnSphere = false;
		end

		function moveToSphere(obj)
			D = size(obj.Y,2);
			obj.meanY = mean(obj.Y, 2);
			obj.Y = bsxfun(@minus, obj.Y, obj.meanY); % [N x D]
			obj.stdY = sqrt(sum(obj.Y.^2, 2)) / sqrt(D - 1) + 1e-6; % [N x 1] / sqrt(D - 1)
			obj.Y = bsxfun(@rdivide, obj.Y, obj.stdY); % [N x D]
			obj.isOnSphere = true;
		end
		
		function revertFromSphere(obj)
			if obj.isOnSphere
				obj.Y = bsxfun(@times, obj.Y, obj.stdY);
				obj.Y = bsxfun(@plus, obj.Y, obj.meanY);
				obj.isOnSphere = false;
			else
				warning('Y is not on sphere')
			end
		end

		function clearNoise(obj)
			obj.Y = [];
			obj.logP = [];
			obj.meanY = [];
			obj.stdY = [];
			obj.isOnSphere = false;
		end
    end

end

