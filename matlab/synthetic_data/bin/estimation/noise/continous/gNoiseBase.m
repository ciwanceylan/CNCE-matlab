function [ noiseBase ] = gNoiseBase( N, D, kappa, varargin)
%GNOISEBASE Generate a noise base, iid normal noise.
%   Generates N * kappa samples of iid normal noise in D dimensions.
% 	varargin is seed.

if nargin > 3
	seed = varargin{1};
    rng(seed);
end

noiseBase = randn(N, D, kappa);

end

