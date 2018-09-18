function [ U, logPU ]  = berNCEnoise( X, nu)
%berNCEnoise Generate discrete noise for NCE of the bernoulli model
%	Input:
%		X - [N x D] matrix of the data.
%       nu - number of noise samples per data sample

[N, D] = size(X);
theNoise = rand(N*nu, D);
U = cat(1, X, theNoise > 0.5);
logPU = 0.5*ones(N*(nu+1), 1);
end

