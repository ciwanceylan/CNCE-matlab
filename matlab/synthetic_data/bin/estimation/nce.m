function [ theta, varargout ] = nce( X, theModel, thetaInit, noiseFun, nu, varargin)
%NCE The NCE estimation algorithm.
%	Performes meta iterations until either mIterMax is reached or 
%	the change in theta falls below a threshold.
%	Input:
%		X - [N x D] matrix with the data
%		theModel - function handel for a model on the form presented in the model file.
%		thetaInit - Initial value for the parameters.
%       noiseFun - function handle to noise generation function
%       nu - number of noise samples per data sample
%       varargin{1}: optimisation options
%	Output:
%		theta - [1 x M] the final estimate of theta.
%		varargout:
%			1. aux.theta - the theta for every iteration.
%				aux.loss - the loss for every iteration
%				aux.epsilon - the value for epsilon at every iteration
%			2. noise
%				{mIterMax + 1, 1} cell, the ajusted noise for every iteration.

nargoutchk(0, 3);
narginchk(5, 6);

if (nargin > 5)
	options = varargin{1};
else
	options.verbose = 0;
	options.alg = 'minimize';
	options.maxIter = 400;
end

% Run optimisation and measure time
tic;
[ U, logPU ] = noiseFun( X, nu );
objFun= @(theta) nce_loss(theModel, theta, U, logPU, nu);
[theta, opt] = optimise_main(objFun, thetaInit, options);
aux.time = toc;

if (nargout > 1)
	aux.theta 	= theta;
	aux.loss 	= opt.loss;
	aux.alg 	= opt.alg;
	varargout{1} = aux;
end

if (nargout > 2)
	varargout{2} = U;
end

end

