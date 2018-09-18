function [ out1, varargout ]= icaModel( X, theta, varargin )
%ICAMODElNCE The unnormalised ICA model of laplacian sources.
%	The model is parameterised by the mixture matrix which is not restricted in any way.
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
c = theta(end);
theta = theta(1:end -1);

[N, D] = size(X);
M = size(theta, 2);
nrSources = M / D;

% Calculate Phi
B = reshape(theta, nrSources, D); % B is "long"
Shat = X * B'; % [N x D] x [D x nrSources] = [N x nrSources]
if (outputflag ~= 1 && outputflag ~= 2)
	logPhi = -1 * sqrt(2) * sum(abs(Shat), 2) + c;
end

if (nargout > 1 || outputflag == 1)
    % Calculate log-gradient w.r.t theta
	g = -1*sqrt(2)*sign(Shat); %
	grad = bsxfun(@times, g, permute(X, [1, 3, 2])); % [N x nrSources x D] 
	logGrad_t = reshape(grad, N, D*nrSources);
	logGrad_t = cat(2, logGrad_t, ones(N, 1));
    varargout{1} = logGrad_t;
end
if (nargout > 2 || outputflag == 2)
    % Calculate log-gradient w.r.t the data
	if (outputflag == 2)
		g = -1*sqrt(2)*sign(Shat);
	end
	grad = bsxfun(@times, g, permute(B, [3, 1, 2])); %Reshape B to [1 x nrSources x D]
	grad = sum(grad, 2);
	logGrad_u = reshape(grad, N, D);
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

