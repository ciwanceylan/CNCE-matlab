function [ theta, varargout ] = ...
    cnce( X, theModel, thetaInit, epsilonBase, noiseBase, noiseFun,...
        mIterMax, varargin)
%CNCE The CNCE estimation algorithm.
%	Performes meta iterations until either mIterMax is reached or 
%	the change in theta falls below a threshold.
%	Input:
%		X - [N x D] matrix with the data
%		theModel - function handel for a model on the form presented in the model file.
%		thetaInit - Initial value for the parameters.
%		epsilonBase - initial value for epsilon ( see CNCE paper).
%		noiseBase 	- [N x D x kappa] normal distributed iid noise.
%		noiseFun - function handle for the conditional noise function
%		mIterMax - the maximum number of meta iterations
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
narginchk(7, 8);

if (nargin > 7)
	options = varargin{1};
else
	options.verbose = 0;
	options.alg = 'fminunc';
	options.maxIter = 400;
end
metaPrecision = 1e-4;  % Stop if relative change in parameter values
                        % fall below this value
M 	= length(thetaInit);

% Auxiliary measurements
aux.time = zeros(mIterMax + 1, 1);
if (nargout > 1)
	aux.theta 		= zeros(mIterMax + 1, M);
	aux.epsilon 	= zeros(mIterMax + 1, 1);
	aux.loss		= zeros(mIterMax + 1, 1);
end
if (nargout > 2)
	noise		= cell(mIterMax + 1, 1);
end

if (options.verbose)
	fprintf('.')
end

% Run initial estimation
tic;
[ U, logPyx_ratio, epsilon] = noiseFun(X, theModel, thetaInit, epsilonBase, noiseBase, 0);
objFun = @(theta) cnce_loss(theModel, theta, U, logPyx_ratio);
[theta] = optimise_main(objFun, thetaInit, options);
aux.time(1)		= toc;

if (nargout > 1)
	aux.theta(1,:) 	= theta;
	aux.epsilon(1) 	= epsilon;
	aux.loss(1)		= objFun(theta);
end
if (nargout > 2)
	noise{1} = U(:,:,2:end);
end

% Run additional meta-iterations if mIterMax > 0
for mIter = 1 : mIterMax
	if (options.verbose)
		fprintf('.')
	end
	thetaOld = theta;
	tic;
	[ U, logPyx_ratio, epsilon] = noiseFun(X, theModel, theta, epsilonBase, noiseBase, mIter);
	objFun = @(theta) cnce_loss(theModel, theta, U, logPyx_ratio);
	[theta] = optimise_main(objFun, theta, options);
	aux.time(mIter + 1)		= toc;
	
    if (nargout > 1)
		aux.theta(mIter + 1,:) 	= theta;
		aux.epsilon(mIter + 1) 	= epsilon;
		aux.loss(mIter + 1)		= objFun(theta);
    end
    
    if (nargout > 2)
		noise{mIter + 1} = U(:,:,2:end);
    end
	
    % Stop if theta is not changing
	if (sqrt(sum((theta - thetaOld).^2)) < sqrt(sum(theta.^2)) * metaPrecision )
		break;
	end
	
end

% If mIterMax was 0
if isempty(mIter), mIter = 0; end

% Output statistics
if (nargout > 1)
	exIter = mIterMax - mIter;
	if exIter > 0
		aux.theta(mIter + 2:end, :) = repmat(aux.theta(mIter + 1, :), exIter, 1);
	end
	aux.epsilon(mIter + 2:end) = aux.epsilon(mIter + 1);
	aux.loss(mIter + 2:end) = aux.loss(mIter + 1);
	aux.mIter 	= mIter + 1;
	varargout{1}	= aux;
end
if (nargout > 2)
	varargout{2} = noise;
end
if (options.verbose)
	fprintf('\n');
end

end

