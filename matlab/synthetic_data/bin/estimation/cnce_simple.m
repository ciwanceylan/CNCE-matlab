function [ theta, varargout ] =...
    cnce_simple( X, theModel, thetaInit, epsilonBase, noiseBase, noiseFun)
%cnce_simple The CNCE estimation algorithm without state-space gradient information.
%	Only performs one meta iteration.
%	Input:
%		X - [N x D] matrix with the data
%		theModel - function handel for a model on the form presented in the model file.
%		thetaInit - Initial value for the parameters.
%		epsilonBase - initial value for epsilon ( see CNCE paper).
%		noiseBase 	- [N x D x kappa] normal distributed iid noise.
%		noiseFun - function handle for the conditional noise function
% 		varargin:
%			options structre for CNCE.
%			option.verbose - 0 or 1, turns verbose estimation off and on.
%	Output:
%		theta - [1 x M] the final estimate of theta.
%		varargout:
%			1. aux.theta - the theta for every iteration.
%				aux.loss - the loss for every iteration
%				aux.epsilon - the value for epsilon at every iteration
%			2. noise
%				{mIterMax + 1, 1} cell, the ajusted noise for every iteration.

nargoutchk(0, 3);

[ U, logPyx_ratio, epsilon] = noiseFun(X, theModel, thetaInit, epsilonBase, noiseBase, 0);
[theta, aux.history] = optimisation_whistory(U, thetaInit, theModel, logPyx_ratio);

if (nargout > 1)
	aux.epsilon 	= epsilon;
	varargout{1}	= aux;
end
if (nargout > 2)
	noise = U(:,:,2:end);
	varargout{2} = noise;
end

end

