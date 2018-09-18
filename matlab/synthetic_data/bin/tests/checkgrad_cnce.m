function [d, dy, dh] = checkgrad_cnce(U, theta, theModel, e, logPyx_ratio, varargin)
%CHECKGRAD calculates the difference between nummerical and analytical gradient.
%	dh: finite difference
%	dy: implemented in cnce_loss
%	d = norm(dh-dy)/norm(dh+dy);

M = size(theta,2);
dh = zeros(1, M);

% Added special case for ring model where only gradient for precision
% is calculated
if nargin > 5
    prmtrInd = varargin{1};
else
    prmtrInd = 1:M;
end


[~, dy] = cnce_loss( theModel, theta, U, logPyx_ratio);
for m=prmtrInd
    pert = zeros(1, M);
    pert(1,m) = e;
	
	pertTheta = theta + pert;
	[loss1] = cnce_loss( theModel, pertTheta, U, logPyx_ratio);
	
	pertTheta = theta - pert;
	[loss2] = cnce_loss( theModel, pertTheta, U, logPyx_ratio);
	
    dh(m) = (loss1-loss2)/(2*e);
    
end


%disp([dy dh])                                          % print the two vectors
d = norm(dh-dy)/norm(dh+dy);       % return norm of diff divided by norm of sum
end