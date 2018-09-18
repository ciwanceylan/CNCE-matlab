function [ Lambda ] = thetatoLambda(theta, D)
%THETATOLAMBDA Maps the [1 x M] parameter vector theta to a [D x D] precision matrix Lambda.
%   [ Lambda ] = thetatoLambda(theta, D)
%		Input:
%			theta - [1 x M] vector containing the parameters of the Gauss model.
%			D - the dimention of Lambda. M = D(D + 1)/2 must hold.
%		output:
%			Lambda - the precision matrix for the Gaussian model.

M = size(theta, 2);
Lambda = tril(ones(D));
Lambda(Lambda == 1) = theta;
Lambda = Lambda + tril(Lambda, -1)';

end

