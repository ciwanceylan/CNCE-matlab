function [ U, logPyx_ratio, varargout ] =...
    gCalcNoise( X, theModel, theta, epsilon, noiseBase, mIter)
%gCalcNoise Calculates the noise and the log ratio of the noise pdf.
%   [ U, logPyx_ratio, varargout ] = ...
%       gCalcNoise( X, theModel, theta, epsilon, noiseBase, mIter)
%	Input:
%		X - [N x D] matrix of the data.
%		theta - [1 x M] vector of the model parameters
%		theModel - function handle of the model
%		epsilon - [eps_mu, eps_sigma] the hyperparameter epsilon 
% 		varargin{1} - if set to 'noGrad' the gradient information is 
%					not used for generastion of the noise.
nargoutchk(2, 3);

[N, D, kappa] = size(noiseBase);

% If mIter > 0 then we use the gradient for the noise
% These results were not presented in the ICML paper but are available
% in http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-213847
if (mIter)
	[grad_u_x] = theModel(X, theta, 2);
else
	grad_u_x = zeros(N, D);
end

mu  = X + (epsilon^2 / 2) * grad_u_x;
Y = bsxfun(@plus, mu, epsilon*noiseBase);

% calculate the gradient w.r.t the data
grad_u_y = zeros(N, D, kappa);
if (mIter)
	for i = 1:kappa
		[grad_u_y(:, :, i)] = theModel(Y(:,:,i), theta, 2);
	end
end

% Concatenate data, noise and data gradients for calculation of conditional noise pdf
U = cat(3, X, Y);
grad_u = cat(3, grad_u_x, grad_u_y);

epsilonFactor = -1/(2*epsilon^2);
c =  (epsilon^2 / 2) .* grad_u_x;
b = (epsilon^2 / 2) .* grad_u_y;
a =  bsxfun(@minus, X, Y);
logPyx_ratio = epsilonFactor * sum( bsxfun(@minus, b.^2, c.^2) - ...
    2 * (bsxfun(@times, a, bsxfun(@plus,b, c))) , 2);
logPyx_ratio = squeeze(logPyx_ratio);


if (nargout > 2)
	aux.grad_u = grad_u;
	aux.mu = mu;
	varargout{1} = aux;
end
end



