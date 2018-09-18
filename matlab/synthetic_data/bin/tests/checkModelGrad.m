function [d_t, d_u]  = checkModelGrad(x, theta, theModel, e, varargin)
%GMCHECKGRAD calculates the difference between nummerical and analytical gradient.
%	d_t: The normalised difference between analytical and numerical calculation 
%		wrt theta.
%	d_u: The normalised difference between analytical and numerical calculation 
%		wrt u (the state).

M = size(theta,2);
[~, D] = size(x);
dh_t = zeros(1, M);
dh_u = zeros(1, D);

% Added special case for ring model where only gradient for precision
% is calculated
if nargin > 4
    prmtrInd = varargin{1};
else
    prmtrInd = 1:M;
end


[~, grad_t, grad_u] = theModel(x, theta);
for m=prmtrInd
    pert = zeros(1, M);
    pert(1,m) = e;
	
	pertTheta = theta + pert;
	[logPhi1] = theModel(x, pertTheta);
	
	pertTheta = theta - pert;
	[logPhi2] = theModel(x, pertTheta);
	
    dh_t(m) = (logPhi1-logPhi2)/(2*e);
    
end

for m=1:D
    pert = zeros(1, D);
    pert(1,m) = e;
	
	pertX = x + pert;
	[logPhi1] = theModel(pertX, theta);
	
	pertX = x - pert;
	[logPhi2] = theModel(pertX, theta);
	
    dh_u(m) = (logPhi1-logPhi2)/(2*e);
    
end


d_t = norm(dh_t-grad_t)/norm(dh_t+grad_t);       % return norm of diff divided by norm of sum
d_u = norm(dh_u-grad_u)/norm(dh_u+grad_u);   
end

