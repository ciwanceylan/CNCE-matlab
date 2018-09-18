function [ U, logPyx_ratio, epsilon, varargout] = ...
    berNoise( X, theModel, thetaInit, epsilonBase, noiseBase, varargin)
%BERNOISE Function which generates noise based on uniform iid noise for discrete CNCE.
%   Use of symmetric conidtional noise sets logPyx_ratio to 0.


loss_inf 	= 2*log(2);  % Loss as epsilon -> 0
lossinf		= 0;  % Loss as epsilon -> inf
thrsLower 	= 0.1;  % Epsilon is too large is loss falls below this value
thrsUpper 	= 0.5 * loss_inf;  % Epsilon is too small is loss goes above 
incRate 	= 0.1;  % Epsilon increase rate
decRate 	= 0.1;  % Epsilon decrease rate
maxIter 	= 50;   % Maximum allowed iterations
epsilonFactor = 1;  % Start value (mutliplied with epsilonBase)

if (nargout > 3)
	aux.epsilonFactors = zeros(maxIter, 1);
end

% Calculate initial noise and loss
[ U, logPyx_ratio] = berCalcNoise( X, epsilonFactor * epsilonBase, noiseBase);
[ loss] = cnce_loss(theModel, thetaInit, U, logPyx_ratio);

if (nargout > 3)
    aux.epsilonFactors(1) = epsilonFactor;
end

% Iterate until conditions are met 
k = 1;
while ( k < maxIter ...
    && (abs(1 - (loss/loss_inf)) < thrsLower || loss < thrsUpper ) ...
	&& (abs(epsilonFactor * epsilonBase) < 1.1))

	if (abs(1 - (loss/loss_inf)) < thrsLower)
		epsilonFactor = (1 + incRate) * epsilonFactor;
	elseif (loss < thrsUpper)
		epsilonFactor = (1 - decRate) * epsilonFactor;
	end
	[ U, logPyx_ratio] = berCalcNoise( X, epsilonFactor * epsilonBase, noiseBase);
	[ loss] = cnce_loss(theModel, thetaInit, U, logPyx_ratio);
	if (nargout > 3)
        aux.epsilonFactors(k+1) = epsilonFactor;
	end
 	k = k + 1;
end

epsilon = epsilonFactor * epsilonBase;

if (nargout > 3)
	aux.epsilonFactors = aux.epsilonFactors( ~aux.epsilonFactors );
	aux.epsilonBase = epsilonBase;
	varargout{1} = aux;
end

end

