function X = getRawData(dataset, winsize, batchSize, batchNr, overSample)
%GETRAWDATA Get image samples from one dataset: 'VanHateren' or 'Tiff'
%   The patches are normalised by the largest pixel value among all the
%   patches and the mean of each patch is subtracted.
%   Input:
%       dataset     name of the dataset, either 'VanHateren' or 'Tiff'
%       winsize     patch window size (usually 32)
%       batchSise   The number of desired patches
%       batchNr     Useful if pre-extracted image patches are used to avoid
%                   sampling same patches muliple times. Currently not used.
%       overSample  If true (default) 2*batchSize patches are sampled and
%                   the half with the highest variance is selected

% 	if nargin > 5 && rerandomize
% 		rng('shuffle');
% 	else
% 		rseed = randi(20020) + batchNr;
% 		fprintf('rngseed for batch = %d\n', rseed);
% 		rng(rseed);
% 	end
	if nargin < 5 || isempty(overSample) || overSample
		overSample = true;
		sampleFactor = 2;
	else
		sampleFactor = 1;
	end
	
	if strcmp(dataset, 'VanHateren') || strcmp(dataset, 'Tiff')
		if strcmp(dataset, 'VanHateren')
			Xraw = getVanHaterenData(sampleFactor * batchSize, winsize);
		else
			Xraw = getTiffPatches(sampleFactor * batchSize, winsize);
		end
		maxPix = max(Xraw(:));
		X = Xraw / maxPix;
		X = bsxfun(@minus, X, mean(X, 2));
		variance = sum(X.^2, 2);
        if overSample 
            % More patch than needed are sampled and those with the highest
            % variance are used
			[~, index] = sort(variance, 1, 'descend');
			X = X(index(1:batchSize), :);
        end
	elseif strcmp(dataset, 'testset')
		X = randn(batchSize, winsize^2);
		X = bsxfun(@minus, X, mean(X, 2));
	else
		error('Unrecognized image dataset.')
	end
end

