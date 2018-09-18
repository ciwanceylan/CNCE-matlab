function [loss, varargout] = nce_loss(theModel, theta, U, logPU, nu)
%nce_loss Calculates the negative objective function and gradient for NCE
%   [loss, varargout] = nce_loss(model, theta, U, logPyx_ratio)
%   Input:
%       theModel - a function handle to the model. The model should take
%       	return outputs and take inputs on the form presented in the Model template file.
%		theta - parameter values.
%       U - [X Y1 Y2 ... Ykappa] [N, D, nu + 1] matrix containing data and noise
%		logPyx_ratio - [N x nu] the log ratio of the conditional noise pdf.
%   Output:
%       loss - the scalar value of the loss function for given data and
%       	parameters
%       gradient - [1 x M] vector of the gradient.

narginchk(5, 5);
nargoutchk(1, 2);
approxFactor = 30;

[K, ~] = size(U);
N = K / (nu + 1);
X = U(1:N,:);
Y = U(N+1:end, :); % concat all noise data to a [ N*nu x D] matrix
% Decide if gradients should be evaluated or not based on nargout.

if (nargout == 1)	
	logPhi_X = theModel(X, theta);
	logPhi_Y = theModel(Y, theta);
elseif (nargout > 1)

	[logPhi_X, logGrad_X] = theModel(X, theta);
	[logPhi_Y, logGrad_Y] = theModel(Y, theta);

end

% Calculate the loss
% G is a [nu*N x 1] vector
Gx = logPhi_X - logPU(1:N);
Gy = logPhi_Y - logPU(N+1:end);

% Separate Gs in different areas:
%	approxFactor = 30
%	small G does not contriubute to the loss (G < -30)
%	large G are linear (G > 30)
mask_largeNegX = Gx < -1 * approxFactor;
mask_interX = abs(Gx) < approxFactor;
mask_largeY = Gy > approxFactor;
mask_interY = abs(Gy) < approxFactor;

% X term - large G -> 0, large negative G -> linear
loss = sum( log(1 + nu*exp( -1*Gx(mask_interX)))) - log(nu)*sum( Gx(mask_largeNegX));
% Y term - large negative G -> 0, large G -> linear
loss = loss + sum( log(1 + (1/nu)*exp( Gy(mask_interY)))) + log(1/nu)*sum( Gy(mask_largeY));
loss = loss / N;
if (nargout > 2)
	aux.Gx = Gx; 
	aux.Gy = Gy; 
	varargout{2} = aux;
end	

% Calculations for the gradient
if (nargout > 1)

    % Gradient for X term
	mask_largeX = ~mask_largeNegX & ~mask_interX;
	
	grad_t_x = -1 * logGrad_X;
	grad_t_x(mask_largeX, :) = ....
		bsxfun(@times, grad_t_x(mask_largeX, :), nu *exp( -1 * Gx(mask_largeX) ) ); 
	grad_t_x(mask_interX, :) = ... 
		bsxfun(@rdivide, grad_t_x(mask_interX, :) , ( 1 + (1/nu)*exp( Gx(mask_interX) )));
	grad_t = sum(grad_t_x, 1);
	
    % Gradient for Y term
	mask_largeNegY = ~mask_largeY & ~mask_interY;
	
	grad_t_y = logGrad_Y;
	grad_t_y(mask_largeNegY , :) = ...
		bsxfun(@times, grad_t_y(mask_largeNegY , :), (1/nu) *exp( Gy(mask_largeNegY) ));
	grad_t_y(mask_interY, :) = ... 
		bsxfun(@rdivide, grad_t_y(mask_interY, :) , ( 1 + nu*exp( -1*Gy(mask_interY) )));
		
	grad_t = grad_t + sum(grad_t_y, 1);
	
	varargout{1} = grad_t/ N;
end


end

