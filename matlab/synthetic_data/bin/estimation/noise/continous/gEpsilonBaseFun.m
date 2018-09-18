function [ epsilonBase ] = gEpsilonBaseFun( X )
%GEPSILONBASEFUN Generates a base value of the epsilon parameter.
%   A reasonable starting point for epsilon is the mean std of the data

epsilonBase = mean(std(X, [], 1));

end

