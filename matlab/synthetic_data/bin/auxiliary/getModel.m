function [ theModel, M] = getModel( dataType, D)
%GETMODEL Help function for CNCE which returns a fuction handle.

if (strcmp(dataType, 'gauss'))
	theModel = @gaussModel;
	M = D*(D+1)/2;
elseif (strcmp(dataType, 'gauss_NCE'))
	theModel = @gaussModelNCE;
	M = D*(D+1)/2 + 1;
elseif (strcmp(dataType, 'ICA'))
	theModel = @icaModel;
	M = D^2;
elseif (strcmp(dataType, 'ICA_NCE'))
	theModel = @icaModelNCE;
	M = D^2 + 1;
elseif (strcmp(dataType, 'lognormal'))
	theModel = @lognormalModel;
	M = 2;
elseif (strcmp(dataType, 'lognormal_NCE'))
	theModel = @lognormalModelNCE;
	M = 3;
elseif (strcmp(dataType, 'bernoulli'))
	theModel = @berModel;
	M = 2;
elseif (strcmp(dataType, 'bernoulli_NCE'))
	theModel = @berModelNCE;
	M = 3;
elseif (strcmp(dataType, 'ring'))
    theModel = @ringModel;
    M = 2;
elseif (strcmp(dataType, 'ring_NCE'))
    theModel = @ringModelNCE;
    M = 3;
else
	error('Could not find model of desired type and dimension')
end

end

