function [ U, logPyx_ratio ]  = berCalcNoise( X, epsilon, noiseBase)
%BERCALCNOISE  Calculates the noise and the log ratio of the noise pdf.
%   [ U, logPyx_ratio, varargout ] = berCalcNoise( X, theModel, theta, epsilon, noiseBase)
%	Input:
%		X - [N x D] matrix of the data.
%		epsilon - the hyperparameter epsilon
%       noiseBase - uniform noise generated using berNoiseBase


[N, ~, kappa] = size(noiseBase);

% X is either 0 or 1. If noiseBase > epsilon Y has same value as X.
% If noiseBase < epsilon Y = ~X
Y = bsxfun(@times, X, (noiseBase > epsilon)) + ...
	bsxfun(@times, ~X, (noiseBase < epsilon)) ;

U = cat(3, X, Y);

logPyx_ratio = zeros(N, kappa);

end

