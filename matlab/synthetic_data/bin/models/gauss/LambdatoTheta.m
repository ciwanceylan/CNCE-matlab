function [ theta ] = LambdatoTheta( Lambda )
%LAMBDATOTHETA Maps the [D x D] precision matrix Lambda to a [1 x M] parameter vector theta.
%   [ Lambda ] = thetatoLambda(theta, D)
%		Input:
%			Lambda - the [D x D] precision matrix for the Gaussian model.
%		output:
%			theta - [1 x M] vector containing the parameters of the Gauss model.
theta = Lambda(tril(ones(size(Lambda)) == 1))';

end

