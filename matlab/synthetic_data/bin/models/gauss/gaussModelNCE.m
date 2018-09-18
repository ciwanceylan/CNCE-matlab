function [ out1, varargout ] = gaussModelNCE( X, theta, varargin )
%gaussModelNCE A unnomrmalised gaussian parameterized by the precision matrix to be used by NCE.
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
theta = theta(1:end-1);
[N, D] = size(X);
M = size(theta, 2);

% Calculate Phi
XLambda = X * (thetatoLambda(theta, D));

% Calculate Phi
if (outputflag ~= 1 && outputflag ~= 2)
	logPhi = -0.5 * (dot(XLambda, X, 2));
	logPhi = bsxfun(@plus, logPhi, c);
end

if (nargout > 1 || outputflag == 1)
    % Calculate log-gradient w.r.t theta
	logGrad_t = -0.5 *(innerDerivative(X));
	logGrad_t = cat(2, logGrad_t, ones(N, 1));
    varargout{1} = logGrad_t;
end
if (nargout > 2 || outputflag == 2)
    % Calculate log-gradient w.r.t the data
	logGrad_u = -1*(XLambda);
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

function [ inder ] = innerDerivative( X )
%INNERDERIVATIVE Calculates the log gradient w.r.t theta for the Gauss model.
%   [ res ] = innerDerivative( X )
% 		Input:
%			X - [N x D] matrix of data points
%		Output:
%			inder - [N x M] matrix containing the derivative of x*Lambda*x' w.r.t to theta.
%		inder is calculated with matrix operations by first calculating a [N x D x D] tensor
%		containing the outer product x' * x for all N data points. The tensor is reshaped into
%		a [N x D^2] matrix which is transformed into the [N x M] inder matrix by a mapping 
%		given by D2toMmap.

[N, D] = size(X);
Xt = repmat(X, 1, 1, D);
%XtT = permute(Xt, [1, 3, 2]);
outerProduct = reshape(Xt .* permute(Xt, [1, 3, 2]), [N, D^2]);
inder = full(outerProduct * D2toMmap(D));

end

function [map] = D2toMmap( D )
%D2TOMMAP Returns a matrix which maps [1 x D^2] vectors to [1 x M] vectors.
%   [map] = D2toMmap( D )
%	Input:
%		D - the dimension of the data.
%	Output:
%		map - a [D^2 x M] sparse matrix which maps data from a symmetric [D x D] matrix
%		onto a M-dimensional space. M = D*(D+1)/2
% 	Because the maping is determined by D, the map matrix is declared persistent so it 
%	does not have to be recalculated for the same D.
persistent prevD
persistent mapStore
if (D == prevD)
	map = mapStore;
	return;
else
	prevD = D;
	M = D*(D+1)/2;
	lowerMap = @(r, c, D) D*(c-1) - c*(c-1)/2 + r;
	upperMap = @(r, c, D) D*(r-1) - r*(r-1)/2 + c;
	map = zeros(D, D, M);

	for r = 1:D
		for c = 1:D
			if (r < c)
				m = upperMap(r, c, D);
			else
				m = lowerMap(r, c, D);
			end
			map(r, c, m) = 1;
		end
	end
	map = sparse(reshape(map, [D^2, M]));
	mapStore = map;
end
end
