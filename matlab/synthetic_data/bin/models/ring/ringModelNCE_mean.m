function [ out1, varargout ]  = ringModelNCE_mean(X, theta, varargin )
%RINGMODELNCE Summary of this function goes here
%   Detailed explanation goes here

nargoutchk(1, 3);
narginchk(2, 3);
outputflag = 0;
if (nargin > 2)
	outputflag = varargin{1};
end

r 		= sqrt(sum(X.^2, 2));
mu 		= theta(1);
prec 	= theta(2);
c 		= theta(3);

% Calculate Phi
arg = r - mu;
argSq = arg.^2;

% Calculate Phi
if (outputflag ~= 1 && outputflag ~= 2)
	logPhi = -0.5 .* prec .* argSq + c ;
end

if (nargout > 1 || outputflag == 1)
    % Calculate log-gradient w.r.t theta
	grad_mu = -1 * prec.* arg;
	grad_prec = -0.5 .* argSq;
	logGrad_t = [grad_mu, grad_prec, ones(size(r,1), 1)];
    varargout{1} = logGrad_t;
end
if (nargout > 2 || outputflag == 2)
    tmp = bsxfun(@rdivide, X, r);
    logGrad_u = bsxfun(@times, -1 .* prec .* arg, tmp);
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

