function [ thetaMLE, varargout ] = getMLE( X, thetaIni, dataType)
%CALCULATEMLE Function to calculate MLE for different models.

if (strcmp(dataType, 'gauss')) 
	thetaMLE = LambdatoTheta(inv(cov(X)));
elseif (strcmp(dataType, 'gauss_NCE'))
	D = size(X, 2);
	Xcov = cov(X);
	thetaMLE = LambdatoTheta(inv(Xcov));
	c = -(D/2) * log(2*pi*det(Xcov));
	thetaMLE = [ thetaMLE , c];
elseif (strcmp(dataType, 'ICA'))
	[thetaMLE, ~, aux] = ica_fast_mle(X, thetaIni);
	if (nargout > 1)
		varargout{1} = aux;
	end
elseif (strcmp(dataType, 'lognormal'))
	thetaMLE = lognormal_mle(X);
elseif (strcmp(dataType, 'bernoulli'))
	theta2 = sum(X, 1) / size(X, 1);
	thetaMLE = [1- theta2, theta2];
elseif (strcmp(dataType, 'ring'))
    R = sqrt(sum(X.^2, 2));
    thetaMLE = [thetaIni(1), 1 ./ std(R).^2];
elseif (strcmp(dataType, 'ring_NCE'))
    R = sqrt(sum(X.^2, 2));
    sig = std(R);
    c = -0.5 * log(2*pi) - log(sig);
    thetaMLE = [thetaIni(1), 1 ./ sig.^2, c];
else
	thetaMLE = thetaIni;
end

end

