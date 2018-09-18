function [d, dy, dh]  = checkgrad_nce(U, theta, theModel, e, logPU, nu, varargin)
%CHECKGRADNCE calculates the difference between nummerical and analytical gradient.
%	dh: finite difference
%	dy: implemented in nce_loss
%	d = norm(dh-dy)/norm(dh+dy);

M = size(theta,2);
dh = zeros(1, M);

% Added special case for ring model where only gradient for precision
% is calculated
if nargin > 6
    prmtrInd = varargin{1};
else
    prmtrInd = 1:M;
end

[~, dy] = nce_loss( theModel, theta, U, logPU, nu);
for m=prmtrInd
    pert = zeros(1, M);
    pert(1,m) = e;
	
	pertTheta = theta + pert;
	[loss1] = nce_loss( theModel, pertTheta, U, logPU, nu);
	
	pertTheta = theta - pert;
	[loss2] = nce_loss( theModel, pertTheta, U, logPU, nu);
	
    dh(m) = (loss1-loss2)/(2*e);
    
end


%disp([dy dh])                                          % print the two vectors
d = norm(dh-dy)/norm(dh+dy);       % return norm of diff divided by norm of sum
end
