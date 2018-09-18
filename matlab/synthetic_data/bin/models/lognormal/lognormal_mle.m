function [ theta ] = lognormal_mle( X )
%LOGNORMAL_MLE Calculates the maximum likelihood estimation of theta for log-normal data.
%   Input:
%		X - [N x 1] matrix with log-normal distributed data
%	Output:
%		theta - [1 x 2] = [1 / var(log(X)), 0];

theta = [1 / var(log(X)), 0];

end

