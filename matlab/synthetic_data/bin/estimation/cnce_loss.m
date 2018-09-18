function [loss, varargout] = cnce_loss(theModel, theta, U, logPyx_ratio)
%cnce_loss Calculates the negative objective function and gradient for CNCE
%   [loss, varargout] = cnce_loss(model, theta, U, logPyx_ratio)
%   Input:
%       theModel - a function handle to the model. The model should take
%       	return outputs and take inputs on the form presented in the Model template file.
%		theta - parameter values.
%       U - [X Y1 Y2 ... Ykappa] [N, D, kappa + 1] matrix containing data and noise
%		logPyx_ratio - [N x kappa] the log ratio of the conditional noise pdf.
%   Output:
%       loss - the scalar value of the loss function for given data and
%       	parameters
%       gradient - [1 x M] vector of the gradient.  
%       
narginchk(4, 4);
nargoutchk(1, 2);
approxFactor = 30;

[N, D, K] = size(U);
kappa = K - 1;
X = squeeze(U(:,:,1));
Y = reshape( permute(U(:,:,2:end), [1, 3, 2]), [N* kappa, D]); % concat all noise data to a [ N*kappa x D] matrix

% Decide if gradients should be evaluated or not based on nargout.
if (nargout == 1)	
	logPhi_X = theModel(X, theta);
	logPhi_Y = theModel(Y, theta);
elseif (nargout > 1)
	[logPhi_X, logGrad_X] = theModel(X, theta);
	[logPhi_Y, logGrad_Y] = theModel(Y, theta);
end

% Calculate the loss
% G is a [kappa*N x 1] vector
% NOTE: sign of X opposite to the version in the ICML paper
G = logPhi_Y - repmat(logPhi_X, kappa, 1) + reshape(logPyx_ratio, [N*kappa, 1]);

% Separate Gs in different areas:
%	small G does not contriubute to the loss (G < -approxFactor)
%	large G are linear (G > approxFactor), large negative G -> 0
mask_large = G > approxFactor;
mask_inter = abs(G) < approxFactor;

% Large G
loss = sum( log(1 + exp( G(mask_inter)))) + sum(G(mask_large));
loss = loss / (0.5*kappa*N);

if (nargout > 2)
	aux.G = -1 * G; % Change sign of G to agree with theory in paper
	varargout{2} = aux;
end	

% Calculations for the gradient
if (nargout > 1)

	grad_t = logGrad_Y - repmat(logGrad_X, kappa, 1);
	% when G is very small, better to not divide by Inf.
	mask_small = ~mask_large & ~mask_inter;
	grad_t(mask_small, :) = bsxfun(@times, exp( G(mask_small) ),  grad_t(mask_small, :)  );
	grad_t(mask_inter, :) = bsxfun(@rdivide, grad_t(mask_inter, :) , ( 1 + exp( - G(mask_inter) )));
	varargout{1} = sum(grad_t, 1) / (0.5*kappa*N);
end


end

