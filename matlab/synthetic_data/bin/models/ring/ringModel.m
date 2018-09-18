function [ out1, varargout ]  = ringModel(X, theta, varargin)
%RingModel A unnomrmalised model on a lower-dimensional manifold.
%   The model is Gaussian in the radial direction with a non-zero mean.
%   The mean is given and the precision is a parameter to be learned.
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

r 		= sqrt(sum(X.^2, 2));
mu 		= theta(1);
prec 	= theta(2);

% Calculate Phi
arg = r - mu;
argSq = arg.^2;

% Calculate Phi
if (outputflag ~= 1 && outputflag ~= 2)
	logPhi = -0.5 .* prec .* argSq;
end

if (nargout > 1 || outputflag == 1)
    % Calculate log-gradient w.r.t theta

    grad_mu = zeros(size(r,1), 1);
    grad_prec = -0.5 .* argSq;
 
	logGrad_t = [grad_mu, grad_prec];
    varargout{1} = logGrad_t;
end
if (nargout > 2 || outputflag == 2)
    tmp = bsxfun(@rdivide, X, r);
    logGrad_u = bsxfun(@times, -1 .* prec .* arg, tmp);
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

