function [ U, logPyx_ratio, epsilon, varargout ] = ...
    gNoise(X, theModel, thetaInit, epsilonBase, noiseBase, mIter, varargin)
%GNOISE Function which generates noise based on Gaussian iid noise for CNCE.
%   Epsilon is adjusted using a heuristic to not be too small nor too large. 

% Epsilon adjustment settings
loss_inf 	= 2*log(2);  % Loss as epsilon -> 0
lossinf		= 0;  % Loss as epsilon -> inf
thrsLower 	= 0.1;  % Epsilon is too large is loss falls below this value
thrsUpper 	= 0.5 * loss_inf;  % Epsilon is too small is loss goes above 
incRate 	= 0.2;  % Epsilon increase rate
decRate 	= 0.5;  % Epsilon decrease rate
maxIter 	= 500;  % Maximum allowed iterations
epsilonFactor = 0.5;  % Start value (mutliplied with epsilonBase)
epsHardCap = 1000;  % Epsilon is not allowed to increase above this value
                    % (in case epsilon goes to 0 very slowly)
if (nargin > 6)
	epsHardCap = varargin{1};
end

if (nargout > 3)
	aux.epsilonFactors = zeros(maxIter, 1);  % storage for all iterations
end

% Calculate initial noise and loss
[ U, logPyx_ratio] = gCalcNoise( X, theModel, thetaInit, ...
    epsilonFactor * epsilonBase, noiseBase, mIter);
[ loss] = cnce_loss(theModel, thetaInit, U, logPyx_ratio);

if (nargout > 3)
aux.epsilonFactors(1) = epsilonFactor;
end

% Iterate until conditions are met 
k = 1;
while ( k < maxIter ...
        && (abs(1 - (loss/loss_inf)) < thrsLower || loss < thrsUpper ) ...
        &&  epsilonFactor < epsHardCap	)

	if (abs(1 - (loss/loss_inf)) < thrsLower)
		epsilonFactor = (1 + incRate) * epsilonFactor;
	elseif (loss < thrsUpper)
		epsilonFactor = (1 - decRate) * epsilonFactor;
	end
	[ U, logPyx_ratio] = gCalcNoise( X, theModel, thetaInit, ...
        epsilonFactor * epsilonBase, noiseBase, mIter);
	[ loss] = cnce_loss(theModel, thetaInit, U, logPyx_ratio);
	if (nargout > 3)
	aux.epsilonFactors(k+1) = epsilonFactor;
	end
 	k = k + 1;
end

% Calculate final epsilon
epsilon = epsilonFactor * epsilonBase;

if (nargout > 3)
	aux.epsilonFactors = aux.epsilonFactors( ~~aux.epsilonFactors );
	aux.epsilonBase = epsilonBase;
	varargout{1} = aux;
end
end

