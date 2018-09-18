function [ out1, varargout ] = lognormalModelNCE( X, theta, varargin )
%LOGNORMALMODEL  A unnomrmalised log-normal model, parameterized by the precision.
% The model is extendet to x <= 0 to which a value c is assigned.
%   [logPhi, varargout] = model(X, theta)
%   where  nargout is between 0 and 2.  
%   returns  grad_t_Phi, grad_u_Phi.
%
%   logPhi - a [N x 1] matrix, the models value at X.
%
%   varargout_1(grad_t_logPhi) - the value of the gradient w.r.t 
%   theta of the log of the model, \grad_\theta \log \Phi.
%
%   varargout_2(grad_u_logPhi) - the value of the gradient w.r.t 
%   the data of the log of the model, \grad_\x \log \Phi.



nargoutchk(1, 3);
narginchk(2, 3);
outputflag = 0;
if (nargin > 2)
	outputflag = varargin{1};
end

[N, D] = size(X);
M = length(theta);

inxPos = find(X > 0);
inxNeg = find(X <= 0);
% Calculate Phi
logX = log(X(inxPos));

% Calculate Phi
if (outputflag ~= 1 && outputflag ~= 2)
	logPhi = zeros(N, 1);
	logPhi(inxPos) = -0.5 * theta(1) * logX.^2 - logX - theta(3);
	logPhi(inxNeg) = theta(2);
end

if (nargout > 1 || outputflag == 1)
    % Calculate log-gradient w.r.t theta
	logGrad_t = zeros(N, 2);
	logGrad_t(inxPos, 1) = -0.5 * logX.^2;
	logGrad_t(inxNeg, 2) = 1;
	logGrad_t(inxPos, 3) = -1;
    varargout{1} = logGrad_t;
end
if (nargout > 2 || outputflag == 2)
    % Calculate log-gradient w.r.t the data
	logGrad_u = zeros(N, 1);
	logGrad_u(inxPos) = -1 * (1./ X(inxPos)) .* ( theta(1) * logX + 1);
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


