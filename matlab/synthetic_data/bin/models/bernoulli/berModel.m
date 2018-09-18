function [ out1, varargout ] = berModel( X, theta, varargin )
%BERMODEL A unnomrmalised Bernoulli model parameterized by 2 scalars to be used by CNCE.
%   [ out1, varargout ] = model(X, theta, varargin )
%	varargin{1} can be used to choose the output for out1:
%		1: gradient of logPhi wrt theta
%		2: gradient of logPhi wrt u
%		ow: logPhi
% 	if nargout > 1  
%   	grad_t_Phi and grad_u_Phi will also be calculated.
%
%   logPhi - a [N x 1] matrix, the log models value at X.
%
%   grad_t_logPhi - a [N x M] matrix, the value of the gradient w.r.t 
%   	theta of the log of the model.
%
%   grad_u_logPhi - a [N x D] matrix, the value of the gradient w.r.t 
%   	u of the log of the model.

nargoutchk(1, 3);
narginchk(2, 3);
outputflag = 0;
if (nargin > 2)
	outputflag = varargin{1};
end

[N, D] = size(X);
M = size(theta, 2);

Ifalse = find(~X);
Nfalse = size(Ifalse, 1);
Itrue = find(X);
Ntrue = size(Itrue, 1);

% Calculate Phi
if (outputflag ~= 1 && outputflag ~= 2)
	logPhi = zeros(N, 1);
	logPhi(Ifalse) = log(theta(1));
	logPhi(Itrue) = log(theta(2));
end

if (nargout > 1 || outputflag == 1)
    % Calculate log-gradient w.r.t theta
	logGrad_t = zeros(N, M);
	logGrad_t(Ifalse, :) = repmat([1/theta(1), 0], Nfalse, 1);
	logGrad_t(Itrue, :) = repmat([0, 1/theta(2)], Ntrue, 1);
    varargout{1} = logGrad_t;
end
if (nargout > 2 || outputflag == 2)
    % Calculate log-gradient w.r.t the data
	% NOT USED FOR DISCRETE DISTRIBUTIONS
	logGrad_u = zeros(N, 1);
    varargout{2} = logGrad_u;
end

switch outputflag
	case 0
		out1 = logPhi; % is log Phi
	case 1
		out1 = logGrad_t; % gradient wrt theta
	case 2
		out1 = logGrad_u; % gradient wrt data
	otherwise
		out1 = logPhi;
end

end

