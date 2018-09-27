function [xmean, C] = covOnline(getDataFun, D, batchSize, nBatches)
	%covOnline Calculates the sample mean and sample covariance matrix in batches (for large data)
	%	getDataFun - function handle to a function which returns a data batchSize
	% 				Should take to parameters: batchSize and iteration number
	%				Should return a matrix [batchSize x D] 
	xmean = zeros(1, D);
	C = zeros(D);
	N = 0;
	for i = 1:nBatches
		N = N + batchSize;
		X = getDataFun(batchSize, i);
		dx = bsxfun(@minus, X, xmean);
		xmean = xmean + (batchSize/N)*(mean(dx, 1));
		dx2 = bsxfun(@minus, X, xmean);
		C = C + dx' * dx2;
	end
	C = C/(N-1);
end

