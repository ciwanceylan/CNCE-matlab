function [ noiseBase ] = berNoiseBase( N, D, kappa )
%BERNOISEBASE Generate a noise base, iid uniform noise.
%   Generates N * kappa samples of iid normal noise in D dimensions.

noiseBase = rand(N, D, kappa);

end

