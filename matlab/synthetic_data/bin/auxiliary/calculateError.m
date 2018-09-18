function [ sqrErr ] = calculateError( thetaTrue, thetaHat, modelName)
%CALCULATEERROR Wraper to calculate the sqErr for different test models.
% Supported models: gauss, lognormal, ICA, bernoulli, ring

K = size(thetaHat, 1);
sqrErr 	= zeros(K, 1);	 
if (strcmp(modelName, 'ICA'))
	D = sqrt(size(thetaTrue,2));
	Btrue 	= reshape(thetaTrue, D, D);
	for k = 1:K
		Bhat = reshape(thetaHat(k,:), D, D);
		sqrErr(k) = calcErr_ica(Btrue, Bhat);
	end
elseif (strcmp(modelName, 'lognormal'))
	sqrErr = (bsxfun(@minus, thetaHat(:,1), thetaTrue(1))).^2;
elseif (strcmp(modelName, 'bernoulli'))
	theta = bsxfun(@rdivide, thetaHat, sum(thetaHat, 2) );
	sqrErr = sum ( (bsxfun(@minus, theta, thetaTrue)).^2, 2);
else
	sqrErr = sum ( (bsxfun(@minus, thetaHat, thetaTrue)).^2, 2);
end

end

